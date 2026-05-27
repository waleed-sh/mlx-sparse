# Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import time

import numpy as np
import pytest

import mlx_sparse as ms
import mlx_sparse._fallback as fallback
import mlx_sparse._native as native
from mlx_sparse import linalg
from mlx_sparse._host import to_numpy


def _bench_ms(mx, fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        mx.eval(fn())
        samples.append(1000.0 * (time.perf_counter() - start))
    return float(np.median(samples))


def _random_csr(mx, scipy_sparse, *, rows: int, cols: int, density: float):
    rng = np.random.default_rng(8675309)
    scipy_csr = scipy_sparse.random(
        rows,
        cols,
        density=density,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    scipy_csr.sum_duplicates()
    scipy_csr.sort_indices()
    csr = ms.csr_array(
        (
            mx.array(scipy_csr.data.astype(np.float32)),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
        canonical=True,
    )
    return csr, scipy_csr, rng


def _random_coo(mx, scipy_sparse, *, rows: int, cols: int, density: float, seed: int):
    rng = np.random.default_rng(seed)
    scipy_coo = scipy_sparse.random(
        rows,
        cols,
        density=density,
        format="coo",
        dtype=np.float32,
        random_state=rng,
    )
    coo = ms.coo_array(
        (
            mx.array(scipy_coo.data.astype(np.float32)),
            (
                mx.array(scipy_coo.row.astype(np.int32)),
                mx.array(scipy_coo.col.astype(np.int32)),
            ),
        ),
        shape=scipy_coo.shape,
    )
    return coo, scipy_coo


def _random_csc(mx, scipy_sparse, *, rows: int, cols: int, density: float, seed: int):
    rng = np.random.default_rng(seed)
    scipy_csc = scipy_sparse.random(
        rows,
        cols,
        density=density,
        format="csc",
        dtype=np.float32,
        random_state=rng,
    )
    scipy_csc.sum_duplicates()
    scipy_csc.sort_indices()
    csc = ms.csc_array(
        (
            mx.array(scipy_csc.data.astype(np.float32)),
            mx.array(scipy_csc.indices.astype(np.int32)),
            mx.array(scipy_csc.indptr.astype(np.int32)),
        ),
        shape=scipy_csc.shape,
        sorted_indices=True,
        canonical=True,
    )
    return csc, scipy_csc


def _reduction_workload(mx, csr, coo, csc):
    return (
        mx.sum(csr.row_sums())
        + mx.sum(csr.col_sums())
        + mx.sum(csr.row_norms())
        + csr.trace()
        + mx.sum(coo.row_sums())
        + mx.sum(coo.col_sums())
        + mx.sum(coo.row_norms())
        + mx.sum(coo.col_norms())
        + coo.trace()
        + mx.sum(csc.row_sums())
        + mx.sum(csc.col_sums())
        + mx.sum(csc.row_norms())
        + mx.sum(csc.col_norms())
        + csc.trace()
    )


def _fallback_reduction_workload(mx, csr, coo, csc):
    return (
        mx.sum(fallback.csr_row_sums(csr.data, csr.indices, csr.indptr, csr.shape))
        + mx.sum(fallback.csr_col_sums(csr.data, csr.indices, csr.indptr, csr.shape))
        + mx.sum(fallback.csr_row_norms(csr.data, csr.indices, csr.indptr, csr.shape))
        + fallback.csr_trace(csr.data, csr.indices, csr.indptr, csr.shape)
        + mx.sum(fallback.coo_row_sums(coo.data, coo.row, coo.col, coo.shape))
        + mx.sum(fallback.coo_col_sums(coo.data, coo.row, coo.col, coo.shape))
        + mx.sum(fallback.coo_row_norms(coo.data, coo.row, coo.col, coo.shape))
        + mx.sum(fallback.coo_col_norms(coo.data, coo.row, coo.col, coo.shape))
        + fallback.coo_trace(coo.data, coo.row, coo.col, coo.shape)
        + mx.sum(fallback.csc_row_sums(csc.data, csc.indices, csc.indptr, csc.shape))
        + mx.sum(fallback.csc_col_sums(csc.data, csc.indices, csc.indptr, csc.shape))
        + mx.sum(fallback.csc_row_norms(csc.data, csc.indices, csc.indptr, csc.shape))
        + mx.sum(fallback.csc_col_norms(csc.data, csc.indices, csc.indptr, csc.shape))
        + fallback.csc_trace(csc.data, csc.indices, csc.indptr, csc.shape)
    )


def _eval_coo_product(mx, coo):
    mx.eval(coo.data, coo.row, coo.col)
    return coo.data


def _eval_csc_product(mx, csc):
    mx.eval(csc.data, csc.indices, csc.indptr)
    return csc.data


@pytest.mark.performance
def test_csr_matvec_native_performance_regression(mx, scipy_sparse):
    if not ms.is_available():
        pytest.skip("native extension is required for performance regression checks")

    rows = int(os.environ.get("MLX_SPARSE_PERF_ROWS", "2048"))
    cols = int(os.environ.get("MLX_SPARSE_PERF_COLS", "2048"))
    density = float(os.environ.get("MLX_SPARSE_PERF_DENSITY", "0.002"))
    warmup = int(os.environ.get("MLX_SPARSE_PERF_WARMUP", "3"))
    iters = int(os.environ.get("MLX_SPARSE_PERF_ITERS", "5"))

    csr, scipy_csr, rng = _random_csr(
        mx,
        scipy_sparse,
        rows=rows,
        cols=cols,
        density=density,
    )
    x_np = rng.standard_normal(cols).astype(np.float32)
    x = mx.array(x_np)
    dense = csr.todense()

    sparse_out = csr @ x
    np.testing.assert_allclose(
        to_numpy(sparse_out),
        scipy_csr @ x_np,
        rtol=1e-5,
        atol=1e-5,
    )

    native_ms = _bench_ms(mx, lambda: csr @ x, warmup=warmup, iters=iters)
    dense_ms = _bench_ms(mx, lambda: dense @ x, warmup=warmup, iters=iters)
    fallback_ms = _bench_ms(
        mx,
        lambda: fallback.csr_matvec(csr.data, csr.indices, csr.indptr, x, csr.shape),
        warmup=1,
        iters=max(2, min(iters, 3)),
    )

    dense_factor = float(os.environ.get("MLX_SPARSE_PERF_DENSE_FACTOR", "20.0"))
    fallback_factor = float(os.environ.get("MLX_SPARSE_PERF_FALLBACK_FACTOR", "1.25"))
    absolute_ms = float(os.environ.get("MLX_SPARSE_PERF_ABSOLUTE_MS", "75.0"))

    assert native_ms <= max(absolute_ms, dense_factor * dense_ms)
    assert native_ms <= fallback_factor * fallback_ms


@pytest.mark.performance
def test_csr_matmul_native_performance_regression(mx, scipy_sparse):
    if not ms.is_available():
        pytest.skip("native extension is required for performance regression checks")

    csr, scipy_csr, rng = _random_csr(
        mx,
        scipy_sparse,
        rows=1024,
        cols=1024,
        density=0.004,
    )
    rhs_np = rng.standard_normal((1024, 8)).astype(np.float32)
    rhs = mx.array(rhs_np)
    dense = csr.todense()

    np.testing.assert_allclose(
        to_numpy(native.csr_matmul(csr.data, csr.indices, csr.indptr, rhs, csr.shape)),
        scipy_csr @ rhs_np,
        rtol=1e-5,
        atol=1e-5,
    )

    native_ms = _bench_ms(mx, lambda: csr @ rhs, warmup=2, iters=4)
    dense_ms = _bench_ms(mx, lambda: dense @ rhs, warmup=2, iters=4)

    assert native_ms <= max(100.0, 25.0 * dense_ms)


@pytest.mark.performance
def test_coo_spgemm_native_performance_regression(mx, scipy_sparse):
    if not ms.is_available():
        pytest.skip("native extension is required for performance regression checks")

    lhs, scipy_lhs = _random_coo(
        mx, scipy_sparse, rows=128, cols=160, density=0.025, seed=1234
    )
    rhs, scipy_rhs = _random_coo(
        mx, scipy_sparse, rows=160, cols=96, density=0.025, seed=5678
    )
    dense_lhs = lhs.todense()
    dense_rhs = rhs.todense()

    out = lhs @ rhs
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        (scipy_lhs @ scipy_rhs).toarray(),
        rtol=1e-5,
        atol=1e-5,
    )

    native_ms = _bench_ms(
        mx, lambda: _eval_coo_product(mx, lhs @ rhs), warmup=2, iters=4
    )
    dense_ms = _bench_ms(mx, lambda: dense_lhs @ dense_rhs, warmup=2, iters=4)

    def fallback_product():
        data, row, col = fallback.coo_matmat(lhs, rhs)
        mx.eval(data, row, col)
        return data

    fallback_ms = _bench_ms(mx, fallback_product, warmup=1, iters=3)

    assert native_ms <= max(150.0, 50.0 * dense_ms)
    assert native_ms <= 2.5 * fallback_ms


@pytest.mark.performance
def test_csc_spgemm_native_performance_regression(mx, scipy_sparse):
    if not ms.is_available():
        pytest.skip("native extension is required for performance regression checks")

    lhs, scipy_lhs = _random_csc(
        mx, scipy_sparse, rows=128, cols=160, density=0.025, seed=2468
    )
    rhs, scipy_rhs = _random_csc(
        mx, scipy_sparse, rows=160, cols=96, density=0.025, seed=1357
    )
    dense_lhs = lhs.todense()
    dense_rhs = rhs.todense()

    out = lhs @ rhs
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        (scipy_lhs @ scipy_rhs).toarray(),
        rtol=1e-5,
        atol=1e-5,
    )

    native_ms = _bench_ms(
        mx, lambda: _eval_csc_product(mx, lhs @ rhs), warmup=2, iters=4
    )
    dense_ms = _bench_ms(mx, lambda: dense_lhs @ dense_rhs, warmup=2, iters=4)

    def fallback_product():
        data, indices, indptr = fallback.csc_matmat(lhs, rhs)
        mx.eval(data, indices, indptr)
        return data

    fallback_ms = _bench_ms(mx, fallback_product, warmup=1, iters=3)

    assert native_ms <= max(150.0, 50.0 * dense_ms)
    assert native_ms <= 2.5 * fallback_ms


@pytest.mark.performance
def test_csr_transpose_matmul_native_performance_regression(mx, scipy_sparse):
    if not ms.is_available():
        pytest.skip("native extension is required for performance regression checks")

    csr, scipy_csr, rng = _random_csr(
        mx,
        scipy_sparse,
        rows=768,
        cols=640,
        density=0.004,
    )
    rhs_np = rng.standard_normal((768, 4)).astype(np.float32)
    rhs = mx.array(rhs_np)
    dense = csr.todense()

    np.testing.assert_allclose(
        to_numpy(
            native.csr_matmul_transpose(
                csr.data, csr.indices, csr.indptr, rhs, csr.shape
            )
        ),
        scipy_csr.T @ rhs_np,
        rtol=1e-5,
        atol=1e-5,
    )

    native_ms = _bench_ms(
        mx,
        lambda: native.csr_matmul_transpose(
            csr.data, csr.indices, csr.indptr, rhs, csr.shape
        ),
        warmup=2,
        iters=4,
    )
    dense_ms = _bench_ms(mx, lambda: mx.transpose(dense) @ rhs, warmup=2, iters=4)

    assert native_ms <= max(100.0, 30.0 * dense_ms)


@pytest.mark.performance
def test_csr_batched_matmul_native_performance_regression(mx, scipy_sparse):
    if not ms.is_available():
        pytest.skip("native extension is required for performance regression checks")

    csr, scipy_csr, rng = _random_csr(
        mx,
        scipy_sparse,
        rows=512,
        cols=512,
        density=0.006,
    )
    rhs_np = rng.standard_normal((4, 512, 6)).astype(np.float32)
    rhs = mx.array(rhs_np)
    dense = csr.todense()

    np.testing.assert_allclose(
        to_numpy(ms.csr_batched_matmul(csr, rhs)),
        np.asarray([scipy_csr @ rhs_np[i] for i in range(4)]),
        rtol=1e-5,
        atol=1e-5,
    )

    native_ms = _bench_ms(
        mx, lambda: ms.csr_batched_matmul(csr, rhs), warmup=2, iters=4
    )
    dense_ms = _bench_ms(mx, lambda: dense[None, :, :] @ rhs, warmup=2, iters=4)

    assert native_ms <= max(100.0, 30.0 * dense_ms)


@pytest.mark.performance
def test_reduction_heavy_native_performance_regression(mx, scipy_sparse):
    if not ms.is_available():
        pytest.skip("native extension is required for performance regression checks")

    rows = int(os.environ.get("MLX_SPARSE_PERF_REDUCTION_ROWS", "640"))
    cols = int(os.environ.get("MLX_SPARSE_PERF_REDUCTION_COLS", "640"))
    density = float(os.environ.get("MLX_SPARSE_PERF_REDUCTION_DENSITY", "0.025"))
    warmup = int(os.environ.get("MLX_SPARSE_PERF_WARMUP", "2"))
    iters = int(os.environ.get("MLX_SPARSE_PERF_ITERS", "4"))

    csr, scipy_csr, _ = _random_csr(
        mx, scipy_sparse, rows=rows, cols=cols, density=density
    )
    scipy_coo = scipy_csr.tocoo(copy=True)
    coo = ms.coo_array(
        (
            mx.array(scipy_coo.data.astype(np.float32)),
            (
                mx.array(scipy_coo.row.astype(np.int32)),
                mx.array(scipy_coo.col.astype(np.int32)),
            ),
        ),
        shape=scipy_coo.shape,
        canonical=True,
    )
    scipy_csc = scipy_csr.tocsc(copy=True)
    csc = ms.csc_array(
        (
            mx.array(scipy_csc.data.astype(np.float32)),
            mx.array(scipy_csc.indices.astype(np.int32)),
            mx.array(scipy_csc.indptr.astype(np.int32)),
        ),
        shape=scipy_csc.shape,
        sorted_indices=True,
        canonical=True,
    )

    dense = scipy_csr.toarray()
    expected = (
        dense.sum(axis=1).sum()
        + dense.sum(axis=0).sum()
        + np.linalg.norm(dense, axis=1).sum()
        + np.trace(dense)
        + dense.sum(axis=1).sum()
        + dense.sum(axis=0).sum()
        + np.linalg.norm(dense, axis=1).sum()
        + np.linalg.norm(dense, axis=0).sum()
        + np.trace(dense)
        + dense.sum(axis=1).sum()
        + dense.sum(axis=0).sum()
        + np.linalg.norm(dense, axis=1).sum()
        + np.linalg.norm(dense, axis=0).sum()
        + np.trace(dense)
    )
    np.testing.assert_allclose(
        to_numpy(_reduction_workload(mx, csr, coo, csc)),
        np.asarray(expected, dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
    )

    native_ms = _bench_ms(
        mx, lambda: _reduction_workload(mx, csr, coo, csc), warmup=warmup, iters=iters
    )
    fallback_ms = _bench_ms(
        mx,
        lambda: _fallback_reduction_workload(mx, csr, coo, csc),
        warmup=1,
        iters=max(2, min(iters, 3)),
    )

    absolute_ms = float(
        os.environ.get("MLX_SPARSE_PERF_REDUCTION_ABSOLUTE_MS", "250.0")
    )
    fallback_factor = float(
        os.environ.get("MLX_SPARSE_PERF_REDUCTION_FALLBACK_FACTOR", "2.5")
    )
    assert native_ms <= max(absolute_ms, fallback_factor * fallback_ms)


@pytest.mark.performance
def test_linalg_cg_native_performance_regression(mx, scipy_sparse):
    if not ms.is_available():
        pytest.skip("native extension is required for performance regression checks")

    n = int(os.environ.get("MLX_SPARSE_PERF_LINALG_N", "512"))
    diagonals = [
        -np.ones(n - 1, dtype=np.float32),
        4.0 * np.ones(n, dtype=np.float32),
        -np.ones(n - 1, dtype=np.float32),
    ]
    scipy_csr = scipy_sparse.diags(diagonals, offsets=[-1, 0, 1], format="csr")
    csr = ms.from_scipy(scipy_csr)
    b_np = np.ones(n, dtype=np.float32)
    b = mx.array(b_np)

    x, info = linalg.cg(csr, b, rtol=1e-5, atol=0.0, maxiter=2 * n)
    assert info == 0
    residual = np.linalg.norm(scipy_csr @ to_numpy(x) - b_np)
    assert residual <= 1e-3 * np.linalg.norm(b_np)

    native_ms = _bench_ms(
        mx,
        lambda: native.csr_cg(
            csr.data,
            csr.indices,
            csr.indptr,
            b,
            mx.zeros((n,), dtype=mx.float32),
            csr.shape,
            rtol=1e-5,
            atol=0.0,
            maxiter=2 * n,
        )[0],
        warmup=1,
        iters=3,
    )
    absolute_ms = float(os.environ.get("MLX_SPARSE_PERF_CG_ABSOLUTE_MS", "250.0"))
    assert native_ms <= absolute_ms
