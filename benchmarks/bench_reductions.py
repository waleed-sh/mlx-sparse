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

"""Microbenchmark COO/CSC sparse reductions.

Run as a script when MLX devices are directly available::

    python benchmarks/bench_reductions.py

The file also exposes a pytest entrypoint so the same benchmark can be run in
environments where the project test fixture is responsible for selecting the
usable MLX device::

    python -m pytest benchmarks/bench_reductions.py -q -s
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest
import scipy.sparse as sp

import mlx_sparse as ms


@pytest.fixture
def mlx_benchmark_device():
    failures = []
    for name, kind in (("gpu", mx.gpu), ("cpu", mx.cpu)):
        try:
            device = mx.Device(kind, 0)
            if not mx.is_available(device):
                failures.append(f"{name}: mx.is_available returned False")
                continue
            mx.set_default_device(device)
            probe = mx.array(np.array([0], dtype=np.float32))
            mx.eval(probe)
            return device
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    pytest.skip("No usable MLX device found. " + " | ".join(failures))


@pytest.fixture
def mx_module(mlx_benchmark_device):
    mx.set_default_device(mlx_benchmark_device)
    return mx


def _select_device():
    for name, kind in (("gpu", mx.gpu), ("cpu", mx.cpu)):
        try:
            device = mx.Device(kind, 0)
            if not mx.is_available(device):
                continue
            mx.set_default_device(device)
            probe = mx.array(np.array([0], dtype=np.float32))
            mx.eval(probe)
            return name
        except Exception:
            continue
    raise RuntimeError("No usable MLX device available for reductions benchmark.")


def _make_matrix(*, n_rows=4096, n_cols=4096, nnz=32768, seed=20260525):
    rng = np.random.default_rng(seed)
    row = rng.integers(0, n_rows, size=nnz, dtype=np.int32)
    col = rng.integers(0, n_cols, size=nnz, dtype=np.int32)
    data = rng.normal(size=nnz).astype(np.float32)
    scipy_coo = sp.coo_matrix((data, (row, col)), shape=(n_rows, n_cols))
    scipy_csc = scipy_coo.tocsc(copy=True)
    coo = ms.coo_array(
        (mx.array(data), (mx.array(row), mx.array(col))),
        shape=(n_rows, n_cols),
        canonical=False,
    )
    csc = coo.tocsc(canonical=False)

    unique_nnz = min(nnz, n_rows * 8)
    canonical_row = np.repeat(np.arange(n_rows, dtype=np.int32), 8)[:unique_nnz]
    canonical_col = np.tile(np.arange(8, dtype=np.int32), n_rows)[:unique_nnz]
    canonical_data = rng.normal(size=unique_nnz).astype(np.float32)
    canonical_scipy_coo = sp.coo_matrix(
        (canonical_data, (canonical_row, canonical_col)),
        shape=(n_rows, n_cols),
    )
    canonical_scipy_csc = canonical_scipy_coo.tocsc(copy=True)
    canonical_coo = ms.coo_array(
        (
            mx.array(canonical_data),
            (mx.array(canonical_row), mx.array(canonical_col)),
        ),
        shape=(n_rows, n_cols),
        canonical=True,
    )
    canonical_csc = canonical_coo.tocsc(canonical=True)
    mx.eval(coo.data, coo.row, coo.col, csc.data, csc.indices, csc.indptr)
    mx.eval(
        canonical_coo.data,
        canonical_coo.row,
        canonical_coo.col,
        canonical_csc.data,
        canonical_csc.indices,
        canonical_csc.indptr,
    )
    return (
        coo,
        csc,
        canonical_coo,
        canonical_csc,
        scipy_coo,
        scipy_csc,
        canonical_scipy_coo,
        canonical_scipy_csc,
    )


def _eval_result(result):
    if hasattr(result, "data") and hasattr(result, "indices"):
        mx.eval(result.data, result.indices, result.indptr)
    else:
        mx.eval(result)


def _time_ms(fn, *, warmups=2, repeats=7):
    for _ in range(warmups):
        _eval_result(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _eval_result(fn())
        times.append((time.perf_counter() - start) * 1_000.0)
    return min(times)


def _time_scipy_ms(fn, *, warmups=2, repeats=7):
    for _ in range(warmups):
        _force_scipy(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _force_scipy(fn())
        times.append((time.perf_counter() - start) * 1_000.0)
    return min(times)


def _force_scipy(result):
    if sp.issparse(result):
        _ = result.data
        if hasattr(result, "indices"):
            _ = result.indices
        if hasattr(result, "indptr"):
            _ = result.indptr
        if hasattr(result, "row"):
            _ = result.row
        if hasattr(result, "col"):
            _ = result.col
        return result
    np.asarray(result)
    return result


def _scipy_row_sums(matrix):
    return np.asarray(matrix.sum(axis=1)).ravel()


def _scipy_col_sums(matrix):
    return np.asarray(matrix.sum(axis=0)).ravel()


def _scipy_row_norms(matrix):
    return np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())


def _scipy_col_norms(matrix):
    return np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())


def run_benchmark():
    (
        coo,
        csc,
        canonical_coo,
        canonical_csc,
        scipy_coo,
        scipy_csc,
        canonical_scipy_coo,
        canonical_scipy_csc,
    ) = _make_matrix()
    ops = [
        (
            "coo_row_sums",
            lambda: coo.tocsr(canonical=False).row_sums(),
            lambda: coo.row_sums(),
            lambda: _scipy_row_sums(scipy_coo),
        ),
        (
            "coo_col_sums",
            lambda: coo.tocsr(canonical=False).col_sums(),
            lambda: coo.col_sums(),
            lambda: _scipy_col_sums(scipy_coo),
        ),
        (
            "coo_row_norms_canonical",
            lambda: canonical_coo.tocsr(canonical=True).row_norms(),
            lambda: ms.coo_row_norms(canonical_coo),
            lambda: _scipy_row_norms(canonical_scipy_coo),
        ),
        (
            "coo_col_norms_canonical",
            lambda: canonical_coo.tocsc(canonical=True).col_norms(),
            lambda: ms.coo_col_norms(canonical_coo),
            lambda: _scipy_col_norms(canonical_scipy_coo),
        ),
        (
            "coo_diagonal",
            lambda: coo.tocsr(canonical=False).diagonal(),
            lambda: coo.diagonal(),
            lambda: scipy_coo.diagonal(),
        ),
        (
            "coo_trace",
            lambda: coo.tocsr(canonical=False).trace(),
            lambda: coo.trace(),
            lambda: scipy_coo.diagonal().sum(),
        ),
        (
            "csc_row_sums",
            lambda: csc.tocsr(canonical=False).row_sums(),
            lambda: csc.row_sums(),
            lambda: _scipy_row_sums(scipy_csc),
        ),
        (
            "csc_col_sums",
            lambda: csc.tocsr(canonical=False).col_sums(),
            lambda: csc.col_sums(),
            lambda: _scipy_col_sums(scipy_csc),
        ),
        (
            "csc_row_norms_canonical",
            lambda: canonical_csc.tocsr(canonical=True).row_norms(),
            lambda: canonical_csc.row_norms(),
            lambda: _scipy_row_norms(canonical_scipy_csc),
        ),
        (
            "csc_col_norms_canonical",
            lambda: canonical_csc.tocsr(canonical=True)
            .tocsc(canonical=True)
            .col_norms(),
            lambda: canonical_csc.col_norms(),
            lambda: _scipy_col_norms(canonical_scipy_csc),
        ),
        (
            "csc_diagonal",
            lambda: csc.tocsr(canonical=False).diagonal(),
            lambda: csc.diagonal(),
            lambda: scipy_csc.diagonal(),
        ),
        (
            "csc_trace",
            lambda: csc.tocsr(canonical=False).trace(),
            lambda: csc.trace(),
            lambda: scipy_csc.diagonal().sum(),
        ),
    ]
    rows = []
    for name, legacy_fn, native_fn, scipy_fn in ops:
        legacy_ms = _time_ms(legacy_fn)
        native_ms = _time_ms(native_fn)
        scipy_ms = _time_scipy_ms(scipy_fn)
        rows.append(
            (
                name,
                legacy_ms,
                native_ms,
                scipy_ms,
                legacy_ms / native_ms,
                scipy_ms / native_ms,
            )
        )
    return rows


def _format_table(rows, device_name):
    lines = [
        f"device: {device_name}",
        "matrix: 4096 x 4096, nnz=32768, dtype=float32",
        "",
        "| operation | legacy conversion ms | native ms | SciPy ms | native vs legacy | native vs SciPy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, legacy_ms, native_ms, scipy_ms, legacy_speedup, scipy_speedup in rows:
        lines.append(
            f"| {name} | {legacy_ms:.3f} | {native_ms:.3f} | {scipy_ms:.3f} | "
            f"{legacy_speedup:.2f}x | {scipy_speedup:.2f}x |"
        )
    return "\n".join(lines)


@pytest.mark.performance
def test_reductions_benchmark(mx_module):
    device_name = str(mx_module.default_device())
    print("\n" + _format_table(run_benchmark(), device_name))


if __name__ == "__main__":
    device = _select_device()
    print(_format_table(run_benchmark(), device))
