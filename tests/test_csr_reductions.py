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
import mlx_sparse._native as native
from mlx_sparse._host import to_numpy


def _dense_from_csr(data_np, indices_np, indptr_np, shape):
    dense = np.zeros(shape, dtype=data_np.dtype)
    for row in range(shape[0]):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        np.add.at(dense[row], indices_np[start:end], data_np[start:end])
    return dense


def _sample_duplicate_csr(mx, dtype_name: str, index_dtype):
    shape = (5, 6)
    values = np.array(
        [
            2.0,
            -1.0,
            0.5,
            3.0,
            -4.0,
            1.5,
            -2.0,
            2.25,
            5.0,
            -0.75,
            1.25,
            -3.0,
            4.0,
        ],
        dtype=np.float32,
    )
    if dtype_name == "complex64":
        values = values.astype(np.complex64) + 1j * np.array(
            [
                0.25,
                -0.5,
                0.75,
                0.0,
                -1.25,
                0.5,
                -0.25,
                1.5,
                0.0,
                2.0,
                -1.0,
                0.25,
                -0.75,
            ],
            dtype=np.float32,
        )
    indices = np.array(
        [0, 2, 2, 5, 1, 0, 2, 4, 0, 3, 3, 4, 5],
        dtype=index_dtype,
    )
    indptr = np.array([0, 4, 5, 8, 8, 13], dtype=index_dtype)
    dtype = getattr(mx, dtype_name)
    csr = ms.csr_array(
        (
            mx.array(values).astype(dtype),
            mx.array(indices),
            mx.array(indptr),
        ),
        shape=shape,
        sorted_indices=False,
        canonical=False,
    )
    return csr, values, indices, indptr


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    ("dtype_name", "rtol", "atol"),
    [
        ("float32", 1e-5, 1e-5),
        ("float16", 6e-3, 6e-3),
        ("bfloat16", 4e-2, 4e-2),
        ("complex64", 1e-5, 1e-5),
    ],
)
def test_csr_reductions_match_dense_and_scipy(
    mx, scipy_sparse, dtype_name, rtol, atol, index_dtype
):
    csr, _, indices_np, indptr_np = _sample_duplicate_csr(mx, dtype_name, index_dtype)
    data_np = to_numpy(csr.data)
    dense = _dense_from_csr(data_np, indices_np, indptr_np, csr.shape)
    scipy_csr = scipy_sparse.csr_matrix(
        (data_np, indices_np, indptr_np), shape=csr.shape
    )

    row_sums = ms.csr_row_sums(csr)
    col_sums = ms.csr_col_sums(csr)
    row_norms = ms.csr_row_norms(csr)
    diagonal = ms.csr_diagonal(csr)
    trace = ms.csr_trace(csr)

    dense_mx = csr.todense()
    dense_row_sums = mx.sum(dense_mx, axis=1)
    dense_col_sums = mx.sum(dense_mx, axis=0)
    dense_row_norms = mx.sqrt(mx.sum(mx.abs(dense_mx) * mx.abs(dense_mx), axis=1))

    expected_row_sums = np.asarray(scipy_csr.sum(axis=1)).reshape(-1)
    expected_col_sums = np.asarray(scipy_csr.sum(axis=0)).reshape(-1)
    norm_dtype = np.complex64 if np.iscomplexobj(data_np) else np.float32
    expected_row_norms = np.linalg.norm(
        scipy_csr.astype(norm_dtype).toarray(), axis=1
    ).astype(np.float32)
    expected_diagonal = scipy_csr.diagonal()
    expected_trace = np.asarray(scipy_csr.trace())

    np.testing.assert_allclose(
        to_numpy(row_sums), expected_row_sums, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(col_sums), expected_col_sums, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(row_norms), expected_row_norms, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(diagonal), expected_diagonal, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(to_numpy(trace), expected_trace, rtol=rtol, atol=atol)

    np.testing.assert_allclose(
        to_numpy(row_sums), to_numpy(dense_row_sums), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(col_sums), to_numpy(dense_col_sums), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(row_norms), to_numpy(dense_row_norms), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(csr.sum(axis=1)), expected_row_sums, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(csr.sum(axis=0)), expected_col_sums, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(to_numpy(csr.sum()), dense.sum(), rtol=rtol, atol=atol)


def test_csr_reduction_methods_and_aliases(mx):
    csr, _, indices_np, indptr_np = _sample_duplicate_csr(mx, "float32", np.int32)
    dense = _dense_from_csr(to_numpy(csr.data), indices_np, indptr_np, csr.shape)

    np.testing.assert_allclose(to_numpy(csr.row_sums()), dense.sum(axis=1))
    np.testing.assert_allclose(to_numpy(csr.col_sums()), dense.sum(axis=0))
    np.testing.assert_allclose(to_numpy(csr.column_sums()), dense.sum(axis=0))
    np.testing.assert_allclose(to_numpy(ms.csr_column_sums(csr)), dense.sum(axis=0))
    np.testing.assert_allclose(to_numpy(csr.row_norms()), np.linalg.norm(dense, axis=1))
    np.testing.assert_allclose(to_numpy(csr.diagonal()), np.diag(dense))
    np.testing.assert_allclose(to_numpy(csr.trace()), np.trace(dense))

    with pytest.raises(ValueError, match="axis must be"):
        csr.sum(axis=2)
    with pytest.raises(TypeError, match="expects CSRArray"):
        ms.csr_row_sums(mx.array(np.eye(2, dtype=np.float32)))


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_native_canonical_reductions_hit_long_row_kernels(mx, index_dtype):
    n_rows = 4
    n_cols = 96
    nnz_per_row = 64
    indices_np = np.tile(np.arange(nnz_per_row, dtype=index_dtype), n_rows)
    indptr_np = np.arange(0, (n_rows + 1) * nnz_per_row, nnz_per_row, dtype=index_dtype)
    data_np = (np.arange(n_rows * nnz_per_row, dtype=np.float32) / 37.0) - 3.0
    csr = ms.csr_array(
        (
            mx.array(data_np),
            mx.array(indices_np),
            mx.array(indptr_np),
        ),
        shape=(n_rows, n_cols),
        sorted_indices=True,
        canonical=True,
    )
    dense = _dense_from_csr(data_np, indices_np, indptr_np, csr.shape)

    np.testing.assert_allclose(
        to_numpy(native.csr_row_sums(csr.data, csr.indices, csr.indptr, csr.shape)),
        dense.sum(axis=1),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(native.csr_row_norms(csr.data, csr.indices, csr.indptr, csr.shape)),
        np.linalg.norm(dense, axis=1),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(native.csr_col_sums(csr.data, csr.indices, csr.indptr, csr.shape)),
        dense.sum(axis=0),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(native.csr_diagonal(csr.data, csr.indices, csr.indptr, csr.shape)),
        np.diag(dense),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(native.csr_trace(csr.data, csr.indices, csr.indptr, csr.shape)),
        np.trace(dense),
        rtol=1e-5,
        atol=1e-5,
    )


def test_reductions_handle_empty_shapes(mx):
    empty = ms.csr_array(
        (
            mx.array(np.array([], dtype=np.float32)),
            mx.array(np.array([], dtype=np.int32)),
            mx.array(np.array([0, 0, 0], dtype=np.int32)),
        ),
        shape=(2, 0),
        sorted_indices=True,
        canonical=True,
    )
    np.testing.assert_allclose(
        to_numpy(empty.row_sums()), np.zeros(2, dtype=np.float32)
    )
    assert empty.col_sums().shape == (0,)
    assert empty.diagonal().shape == (0,)
    np.testing.assert_allclose(to_numpy(empty.trace()), np.array(0.0, dtype=np.float32))
