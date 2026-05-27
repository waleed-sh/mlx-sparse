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
from mlx_sparse._host import to_numpy

_DTYPE_TOLERANCES = {
    "float32": (2e-5, 2e-5),
    "float16": (8e-3, 8e-3),
    "bfloat16": (5e-2, 5e-2),
    "complex64": (2e-5, 2e-5),
}


def _typed_values(mx, values: np.ndarray, dtype_name: str):
    dtype = getattr(mx, dtype_name)
    data = mx.array(values).astype(dtype)
    return data, to_numpy(data)


def _dense_from_csr(data, indices, indptr, shape):
    dense = np.zeros(shape, dtype=data.dtype)
    for row in range(shape[0]):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        np.add.at(dense[row], indices[start:end], data[start:end])
    return dense


def _long_segment_csr(mx, dtype_name: str, index_dtype):
    n_rows = 72
    n_cols = 96
    nnz_per_row = 40
    indices = []
    indptr = [0]
    values = []

    for row in range(n_rows):
        cols = (row * 5 + np.arange(nnz_per_row) * 7) % n_cols
        cols[0] = row
        cols[1] = row
        indices.extend(cols.astype(index_dtype))
        base = np.arange(nnz_per_row, dtype=np.float32)
        values.extend(np.sin(0.17 * (row + 1) * (base + 1)) / 3.0)
        indptr.append(len(indices))

    values = np.asarray(values, dtype=np.float32)
    if dtype_name == "complex64":
        values = values.astype(np.complex64) + 1j * (
            np.cos(np.arange(values.size, dtype=np.float32) * 0.11) / 5.0
        )

    data, stored = _typed_values(mx, values, dtype_name)
    indices = np.asarray(indices, dtype=index_dtype)
    indptr = np.asarray(indptr, dtype=index_dtype)
    csr = ms.csr_array(
        (data, mx.array(indices), mx.array(indptr)),
        shape=(n_rows, n_cols),
        sorted_indices=False,
        canonical=False,
    )
    return csr, _dense_from_csr(stored, indices, indptr, csr.shape)


def _large_trace_arrays(mx, dtype_name: str, index_dtype):
    n = 768
    entries_per_row = 4
    row = np.repeat(np.arange(n, dtype=index_dtype), entries_per_row)
    col_chunks = []
    values = []
    indptr = [0]

    for r in range(n):
        cols = np.array([r, (r + 17) % n, (r * 3 + 11) % n, r], dtype=index_dtype)
        col_chunks.append(cols)
        values.extend(
            np.array(
                [
                    np.sin(r * 0.013) + 0.25,
                    np.cos(r * 0.017) * 0.125,
                    np.sin(r * 0.019) * 0.0625,
                    np.cos(r * 0.023) - 0.5,
                ],
                dtype=np.float32,
            )
        )
        indptr.append(len(values))

    col = np.concatenate(col_chunks).astype(index_dtype, copy=False)
    values = np.asarray(values, dtype=np.float32)
    if dtype_name == "complex64":
        values = values.astype(np.complex64) + 1j * (
            np.sin(np.arange(values.size, dtype=np.float32) * 0.07) / 7.0
        )

    data, stored = _typed_values(mx, values, dtype_name)
    indptr = np.asarray(indptr, dtype=index_dtype)
    shape = (n, n)
    csr = ms.csr_array(
        (data, mx.array(col), mx.array(indptr)),
        shape=shape,
        sorted_indices=False,
        canonical=False,
    )
    coo = ms.coo_array(
        (data, (mx.array(row), mx.array(col))),
        shape=shape,
        canonical=False,
    )
    dense = _dense_from_csr(stored, col, indptr, shape)
    return csr, coo, dense


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype_name", list(_DTYPE_TOLERANCES))
def test_dtype_specific_tolerances_for_long_reduction_segments(
    mx, dtype_name, index_dtype
):
    rtol, atol = _DTYPE_TOLERANCES[dtype_name]
    csr, dense = _long_segment_csr(mx, dtype_name, index_dtype)
    csc = csr.tocsc(canonical=False)

    np.testing.assert_allclose(
        to_numpy(csr.row_sums()), dense.sum(axis=1), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(csr.col_sums()), dense.sum(axis=0), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(csr.diagonal()), np.diag(dense), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(csr.trace()), np.trace(dense), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(csr.row_norms()),
        np.linalg.norm(dense, axis=1).astype(np.float32),
        rtol=rtol,
        atol=atol,
    )

    np.testing.assert_allclose(
        to_numpy(csc.row_sums()), dense.sum(axis=1), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(csc.col_sums()), dense.sum(axis=0), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(csc.row_norms()),
        np.linalg.norm(dense, axis=1).astype(np.float32),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(csc.col_norms()),
        np.linalg.norm(dense, axis=0).astype(np.float32),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(csc.diagonal()), np.diag(dense), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(csc.trace()), np.trace(dense), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype_name", list(_DTYPE_TOLERANCES))
def test_dtype_specific_tolerances_for_staged_trace_reductions(
    mx, dtype_name, index_dtype
):
    rtol, atol = _DTYPE_TOLERANCES[dtype_name]
    csr, coo, dense = _large_trace_arrays(mx, dtype_name, index_dtype)
    csc = csr.tocsc(canonical=False)
    expected = np.trace(dense)

    np.testing.assert_allclose(to_numpy(csr.trace()), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(to_numpy(csc.trace()), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(to_numpy(coo.trace()), expected, rtol=rtol, atol=atol)
