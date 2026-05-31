"""Microbenchmark COO/CSC sparse reductions.

Run as a script when MLX devices are directly available::

    python benchmarks/bench_reductions.py

The file also exposes a pytest entrypoint so the same benchmark can be run in
environments where the project test fixture is responsible for selecting the
usable MLX device::

    python -m pytest benchmarks/bench_reductions.py -q -s
"""

from __future__ import annotations

import os
import time

import mlx.core as mx
import numpy as np
import pytest
import scipy.sparse as sp

import mlx_sparse as ms


def _device_candidates():
    requested = os.environ.get("MLX_SPARSE_TEST_DEVICE", "auto").lower()
    if requested == "cpu":
        return (("cpu", mx.cpu),)
    if requested == "gpu":
        return (("gpu", mx.gpu),)
    if requested != "auto":
        raise ValueError(
            "MLX_SPARSE_TEST_DEVICE must be 'auto', 'cpu', or 'gpu', "
            f"got {requested!r}."
        )
    return (("gpu", mx.gpu), ("cpu", mx.cpu))


@pytest.fixture
def mlx_benchmark_device():
    failures = []
    for name, kind in _device_candidates():
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
    for name, kind in _device_candidates():
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


def _make_matrix(*, n_rows=None, n_cols=None, nnz=None, seed=20260525):
    n_rows = int(os.environ.get("MLX_SPARSE_REDUCTIONS_N_ROWS", n_rows or 4096))
    n_cols = int(os.environ.get("MLX_SPARSE_REDUCTIONS_N_COLS", n_cols or 4096))
    nnz = int(os.environ.get("MLX_SPARSE_REDUCTIONS_NNZ", nnz or 32768))
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
    csr = coo.tocsr(canonical=False)
    csc = coo.tocsc(canonical=False)
    scipy_csr = scipy_coo.tocsr(copy=True)

    canonical_per_row = int(
        os.environ.get("MLX_SPARSE_REDUCTIONS_CANONICAL_PER_ROW", 8)
    )
    canonical_per_row = max(1, min(canonical_per_row, n_cols))
    unique_nnz = min(nnz, n_rows * canonical_per_row)
    canonical_row = np.repeat(np.arange(n_rows, dtype=np.int32), canonical_per_row)[
        :unique_nnz
    ]
    canonical_col = np.tile(np.arange(canonical_per_row, dtype=np.int32), n_rows)[
        :unique_nnz
    ]
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
    canonical_csr = canonical_coo.tocsr(canonical=True)
    canonical_csc = canonical_coo.tocsc(canonical=True)
    canonical_scipy_csr = canonical_scipy_coo.tocsr(copy=True)
    mx.eval(
        coo.data,
        coo.row,
        coo.col,
        csr.data,
        csr.indices,
        csr.indptr,
        csc.data,
        csc.indices,
        csc.indptr,
    )
    mx.eval(
        canonical_coo.data,
        canonical_coo.row,
        canonical_coo.col,
        canonical_csr.data,
        canonical_csr.indices,
        canonical_csr.indptr,
        canonical_csc.data,
        canonical_csc.indices,
        canonical_csc.indptr,
    )
    return (
        coo,
        csr,
        csc,
        canonical_coo,
        canonical_csr,
        canonical_csc,
        scipy_coo,
        scipy_csr,
        scipy_csc,
        canonical_scipy_coo,
        canonical_scipy_csr,
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


def _mlx_dense_trace(matrix):
    dense = matrix.todense()
    eye = mx.eye(matrix.shape[0], matrix.shape[1], dtype=dense.dtype)
    return mx.sum(dense * eye)


def _mlx_dense_dot(lhs, rhs, *, conjugate_lhs=False):
    lhs_dense = lhs.todense()
    rhs_dense = rhs.todense()
    if conjugate_lhs:
        lhs_dense = mx.conjugate(lhs_dense)
    return mx.sum(lhs_dense * rhs_dense)


def _scipy_sparse_dot(lhs, rhs, *, conjugate_lhs=False):
    left = lhs.conjugate() if conjugate_lhs else lhs
    return left.multiply(rhs).sum()


def run_benchmark():
    (
        coo,
        csr,
        csc,
        canonical_coo,
        canonical_csr,
        canonical_csc,
        scipy_coo,
        scipy_csr,
        scipy_csc,
        canonical_scipy_coo,
        canonical_scipy_csr,
        canonical_scipy_csc,
    ) = _make_matrix()
    ops = [
        (
            "csr_row_sums",
            lambda: coo.row_sums(),
            lambda: csr.row_sums(),
            lambda: _scipy_row_sums(scipy_csr),
        ),
        (
            "csr_row_norms_canonical",
            lambda: canonical_coo.row_norms(),
            lambda: canonical_csr.row_norms(),
            lambda: _scipy_row_norms(canonical_scipy_csr),
        ),
        (
            "csr_diagonal",
            lambda: coo.diagonal(),
            lambda: csr.diagonal(),
            lambda: scipy_csr.diagonal(),
        ),
        (
            "csr_todense",
            lambda: coo.todense(),
            lambda: csr.todense(),
            lambda: scipy_csr.toarray(),
        ),
        (
            "csr_trace",
            lambda: _mlx_dense_trace(csr),
            lambda: csr.trace(),
            lambda: scipy_csr.diagonal().sum(),
        ),
        (
            "csr_dot_canonical",
            lambda: _mlx_dense_dot(canonical_csr, canonical_csr),
            lambda: canonical_csr.dot(canonical_csr),
            lambda: _scipy_sparse_dot(canonical_scipy_csr, canonical_scipy_csr),
        ),
        (
            "csr_vdot_canonical",
            lambda: _mlx_dense_dot(canonical_csr, canonical_csr, conjugate_lhs=True),
            lambda: canonical_csr.vdot(canonical_csr),
            lambda: _scipy_sparse_dot(
                canonical_scipy_csr, canonical_scipy_csr, conjugate_lhs=True
            ),
        ),
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
        (
            "csc_todense",
            lambda: csc.tocsr(canonical=False).todense(),
            lambda: csc.todense(),
            lambda: scipy_csc.toarray(),
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
    n_rows = int(os.environ.get("MLX_SPARSE_REDUCTIONS_N_ROWS", 4096))
    n_cols = int(os.environ.get("MLX_SPARSE_REDUCTIONS_N_COLS", 4096))
    nnz = int(os.environ.get("MLX_SPARSE_REDUCTIONS_NNZ", 32768))
    lines = [
        f"device: {device_name}",
        f"matrix: {n_rows} x {n_cols}, nnz={nnz}, dtype=float32",
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
