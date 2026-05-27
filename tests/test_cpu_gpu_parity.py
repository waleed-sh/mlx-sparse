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

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse._host import to_numpy

_REDUCTION_TOLERANCES = {
    "float32": (2e-5, 2e-5),
    "float16": (8e-3, 8e-3),
    "bfloat16": (5e-2, 5e-2),
    "complex64": (2e-5, 2e-5),
}


def _available_devices(mx):
    devices = []
    for name, kind in (("cpu", mx.cpu), ("gpu", mx.gpu)):
        try:
            device = mx.Device(kind, 0)
            if not mx.is_available(device):
                continue
            mx.set_default_device(device)
            probe = mx.array(np.array([0.0], dtype=np.float32))
            mx.eval(probe)
            devices.append((name, device))
        except Exception:
            continue
    if not devices:
        pytest.skip("No usable MLX CPU or GPU device found.")
    return devices


def _sample_csr(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    return ms.csr_array(
        (data, indices, indptr),
        shape=(3, 4),
        sorted_indices=True,
        canonical=True,
    )


def _reduction_stress_arrays(mx, dtype_name: str):
    n = 640
    nnz_per_row = 36
    rows = []
    cols = []
    indptr = [0]
    values = []

    for row in range(n):
        row_cols = np.sort((row + np.arange(nnz_per_row) * 17) % n).astype(np.int32)
        rows.extend(np.full(nnz_per_row, row, dtype=np.int32))
        cols.extend(row_cols)
        base = np.arange(nnz_per_row, dtype=np.float32)
        values.extend(np.sin(0.011 * (row + 1) * (base + 1)) / 4.0)
        indptr.append(len(cols))

    values = np.asarray(values, dtype=np.float32)
    if dtype_name == "complex64":
        values = values.astype(np.complex64) + 1j * (
            np.cos(np.arange(values.size, dtype=np.float32) * 0.003) / 6.0
        )

    dtype = getattr(mx, dtype_name)
    data = mx.array(values).astype(dtype)
    row = mx.array(np.asarray(rows, dtype=np.int32))
    col = mx.array(np.asarray(cols, dtype=np.int32))
    indptr = mx.array(np.asarray(indptr, dtype=np.int32))
    csr = ms.csr_array(
        (data, col, indptr),
        shape=(n, n),
        sorted_indices=True,
        canonical=True,
    )
    coo = ms.coo_array((data, (row, col)), shape=(n, n), canonical=True)
    csc = csr.tocsc(canonical=True)
    return csr, coo, csc


def _reduction_results(csr, coo, csc):
    return {
        "csr_row_sums": to_numpy(csr.row_sums()),
        "csr_col_sums": to_numpy(csr.col_sums()),
        "csr_row_norms": to_numpy(csr.row_norms()),
        "csr_diagonal": to_numpy(csr.diagonal()),
        "csr_trace": to_numpy(csr.trace()),
        "coo_row_sums": to_numpy(coo.row_sums()),
        "coo_col_sums": to_numpy(coo.col_sums()),
        "coo_row_norms": to_numpy(coo.row_norms()),
        "coo_col_norms": to_numpy(coo.col_norms()),
        "coo_diagonal": to_numpy(coo.diagonal()),
        "coo_trace": to_numpy(coo.trace()),
        "csc_row_sums": to_numpy(csc.row_sums()),
        "csc_col_sums": to_numpy(csc.col_sums()),
        "csc_row_norms": to_numpy(csc.row_norms()),
        "csc_col_norms": to_numpy(csc.col_norms()),
        "csc_diagonal": to_numpy(csc.diagonal()),
        "csc_trace": to_numpy(csc.trace()),
    }


@pytest.mark.gpu
def test_sparse_dense_ops_match_dense_mlx_on_available_devices():
    mx = pytest.importorskip("mlx.core")
    x_np = np.array([3.0, 10.0, 7.0, -2.0], dtype=np.float32)
    rhs_np = np.arange(20, dtype=np.float32).reshape(4, 5) / 10.0

    for _, device in _available_devices(mx):
        mx.set_default_device(device)
        csr = _sample_csr(mx)
        dense = csr.todense()
        x = mx.array(x_np)
        rhs = mx.array(rhs_np)

        np.testing.assert_allclose(np.array(csr @ x), np.array(dense @ x))
        np.testing.assert_allclose(np.array(csr @ rhs), np.array(dense @ rhs))
        np.testing.assert_allclose(np.array(csr.T.todense()), np.array(dense.T))


@pytest.mark.gpu
def test_coo_conversion_matches_dense_on_available_devices():
    mx = pytest.importorskip("mlx.core")
    for _, device in _available_devices(mx):
        mx.set_default_device(device)
        coo = ms.coo_array(
            (
                mx.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
                (
                    mx.array(np.array([0, 0, 2, 1], dtype=np.int32)),
                    mx.array(np.array([2, 0, 1, 3], dtype=np.int32)),
                ),
            ),
            shape=(3, 4),
        )
        csr = coo.tocsr()
        expected = np.array(
            [[2.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 4.0], [0.0, 3.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.array(csr.todense()), expected)


@pytest.mark.gpu
def test_fromdense_and_sum_duplicates_match_on_available_devices():
    mx = pytest.importorskip("mlx.core")
    dense_np = np.array(
        [[0.0, 2.0, 1e-4], [3.0, 0.0, -4.0], [0.0, 5e-4, 0.0]],
        dtype=np.float32,
    )
    reference = None

    for _, device in _available_devices(mx):
        mx.set_default_device(device)
        from_dense = ms.fromdense(mx.array(dense_np), threshold=1e-3)
        duplicate = ms.csr_array(
            (
                mx.array(np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)),
                mx.array(np.array([2, 0, 2, 0], dtype=np.int32)),
                mx.array(np.array([0, 4], dtype=np.int32)),
            ),
            shape=(1, 3),
        ).canonicalize()
        result = {
            "fromdense": np.array(from_dense.todense()),
            "duplicate_data": np.array(duplicate.data),
            "duplicate_indices": np.array(duplicate.indices),
            "duplicate_indptr": np.array(duplicate.indptr),
        }
        if reference is None:
            reference = result
            continue
        np.testing.assert_allclose(result["fromdense"], reference["fromdense"])
        np.testing.assert_allclose(
            result["duplicate_data"], reference["duplicate_data"]
        )
        np.testing.assert_array_equal(
            result["duplicate_indices"], reference["duplicate_indices"]
        )
        np.testing.assert_array_equal(
            result["duplicate_indptr"], reference["duplicate_indptr"]
        )


@pytest.mark.gpu
def test_sparse_linalg_native_ops_match_on_available_devices():
    mx = pytest.importorskip("mlx.core")
    reference = None
    for _, device in _available_devices(mx):
        mx.set_default_device(device)
        spd = ms.csr_array(
            (
                mx.array(np.array([4.0, -1.0, -1.0, 4.0], dtype=np.float32)),
                mx.array(np.array([0, 1, 0, 1], dtype=np.int32)),
                mx.array(np.array([0, 2, 4], dtype=np.int32)),
            ),
            shape=(2, 2),
            validate="full",
            canonical=True,
        )
        rhs = mx.array(np.array([1.0, 2.0], dtype=np.float32))
        x, info = linalg.cg(spd, rhs, rtol=1e-6, maxiter=32)
        result = {
            "dot": np.array(spd.dot(spd)),
            "vdot": np.array(spd.vdot(spd)),
            "cg_info": info,
            "cg": np.array(x),
        }
        if reference is None:
            reference = result
            continue
        assert result["cg_info"] == reference["cg_info"] == 0
        np.testing.assert_allclose(result["dot"], reference["dot"], rtol=1e-6)
        np.testing.assert_allclose(result["vdot"], reference["vdot"], rtol=1e-6)
        np.testing.assert_allclose(result["cg"], reference["cg"], rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype_name", list(_REDUCTION_TOLERANCES))
def test_reduction_heavy_cpu_gpu_parity(dtype_name):
    mx = pytest.importorskip("mlx.core")
    devices = _available_devices(mx)
    if len(devices) < 2:
        pytest.skip("CPU/GPU reduction parity requires both devices.")

    reference = None
    for name, device in devices:
        mx.set_default_device(device)
        csr, coo, csc = _reduction_stress_arrays(mx, dtype_name)
        result = _reduction_results(csr, coo, csc)
        if reference is None:
            reference = result
            continue

        rtol, atol = _REDUCTION_TOLERANCES[dtype_name]
        for key, expected in reference.items():
            np.testing.assert_allclose(
                result[key],
                expected,
                rtol=rtol,
                atol=atol,
                err_msg=f"{key} differed on {name}",
            )
