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
from conftest import to_numpy

import mlx_sparse as ms
import mlx_sparse._fallback as fallback
import mlx_sparse._native as native


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
