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
import mlx_sparse._fallback as fallback
import mlx_sparse._native as native
from mlx_sparse._host import to_numpy


def _dense_from_coo(data, row, col, shape):
    dense = np.zeros(shape, dtype=data.dtype)
    np.add.at(dense, (row, col), data)
    return dense


def _sample_duplicate_coo(mx, dtype_name: str, index_dtype):
    shape = (5, 6)
    real = np.array(
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
            -1.5,
        ],
        dtype=np.float32,
    )
    if dtype_name == "complex64":
        imag = np.array(
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
                0.5,
            ],
            dtype=np.float32,
        )
        values = real.astype(np.complex64) + 1j * imag
    else:
        values = real
    row = np.array([0, 0, 0, 0, 1, 2, 2, 2, 4, 4, 4, 4, 4, 2], dtype=index_dtype)
    col = np.array([0, 2, 2, 5, 1, 0, 2, 4, 0, 3, 3, 4, 5, 0], dtype=index_dtype)
    dtype = getattr(mx, dtype_name)
    coo = ms.coo_array(
        (mx.array(values).astype(dtype), (mx.array(row), mx.array(col))),
        shape=shape,
        canonical=False,
    )
    return coo, values, row, col


def _assert_reductions_match_dense(array, dense, rtol, atol):
    expected_row_sums = dense.sum(axis=1)
    expected_col_sums = dense.sum(axis=0)
    expected_row_norms = np.linalg.norm(dense, axis=1).astype(np.float32)
    expected_col_norms = np.linalg.norm(dense, axis=0).astype(np.float32)

    np.testing.assert_allclose(
        to_numpy(array.row_sums()), expected_row_sums, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(array.col_sums()), expected_col_sums, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(array.column_sums()), expected_col_sums, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(array.row_norms()), expected_row_norms, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(array.col_norms()), expected_col_norms, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(array.column_norms()), expected_col_norms, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(array.diagonal()), np.diag(dense), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(array.trace()), np.trace(dense), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(array.sum(axis=1)), expected_row_sums, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        to_numpy(array.sum(axis=0)), expected_col_sums, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(to_numpy(array.sum()), dense.sum(), rtol=rtol, atol=atol)


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
def test_coo_and_csc_reductions_match_dense_and_scipy(
    mx, scipy_sparse, dtype_name, rtol, atol, index_dtype
):
    coo, _, row, col = _sample_duplicate_coo(mx, dtype_name, index_dtype)
    data_np = to_numpy(coo.data)
    dense = _dense_from_coo(data_np, row, col, coo.shape)
    scipy_data = data_np.astype(np.float32) if data_np.dtype == np.float16 else data_np
    scipy_coo = scipy_sparse.coo_matrix((scipy_data, (row, col)), shape=coo.shape)
    scipy_csc = scipy_coo.tocsc()

    _assert_reductions_match_dense(coo, dense, rtol, atol)
    _assert_reductions_match_dense(coo.tocsc(canonical=False), dense, rtol, atol)

    np.testing.assert_allclose(
        to_numpy(ms.coo_row_sums(coo)),
        np.asarray(scipy_coo.sum(axis=1)).reshape(-1),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(ms.csc_col_sums(coo.tocsc(canonical=False))),
        np.asarray(scipy_csc.sum(axis=0)).reshape(-1),
        rtol=rtol,
        atol=atol,
    )


def test_reduction_methods_and_errors(mx):
    coo, _, row, col = _sample_duplicate_coo(mx, "float32", np.int32)
    dense = _dense_from_coo(to_numpy(coo.data), row, col, coo.shape)
    csc = coo.tocsc(canonical=False)

    np.testing.assert_allclose(to_numpy(ms.coo_col_sums(coo)), dense.sum(axis=0))
    np.testing.assert_allclose(to_numpy(ms.coo_column_sums(coo)), dense.sum(axis=0))
    np.testing.assert_allclose(to_numpy(ms.coo_diagonal(coo)), np.diag(dense))
    np.testing.assert_allclose(to_numpy(ms.coo_trace(coo)), np.trace(dense))
    np.testing.assert_allclose(to_numpy(ms.csc_row_sums(csc)), dense.sum(axis=1))
    np.testing.assert_allclose(to_numpy(ms.csc_column_sums(csc)), dense.sum(axis=0))
    np.testing.assert_allclose(
        to_numpy(ms.csc_row_norms(csc)), np.linalg.norm(dense, axis=1)
    )
    np.testing.assert_allclose(
        to_numpy(ms.csc_column_norms(csc)), np.linalg.norm(dense, axis=0)
    )
    np.testing.assert_allclose(to_numpy(ms.csc_diagonal(csc)), np.diag(dense))
    np.testing.assert_allclose(to_numpy(ms.csc_trace(csc)), np.trace(dense))

    with pytest.raises(ValueError, match="axis must be"):
        coo.sum(axis=2)
    with pytest.raises(ValueError, match="axis must be"):
        csc.sum(axis=2)
    with pytest.raises(TypeError, match="expects COOArray"):
        ms.coo_row_sums(csc)
    with pytest.raises(TypeError, match="expects CSCArray"):
        ms.csc_col_sums(coo)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_native_canonical_coo_and_csc_reduction_kernels(mx, index_dtype):
    shape = (4, 96)
    rows = np.repeat(np.arange(shape[0], dtype=index_dtype), 32)
    cols = np.tile(np.arange(32, dtype=index_dtype), shape[0])
    data = (np.arange(rows.size, dtype=np.float32) / 17.0) - 4.0
    coo = ms.coo_array(
        (mx.array(data), (mx.array(rows), mx.array(cols))),
        shape=shape,
        canonical=True,
    )
    csc = coo.tocsc(canonical=True)
    dense = _dense_from_coo(data, rows, cols, shape)

    np.testing.assert_allclose(
        to_numpy(native.coo_row_sums(coo.data, coo.row, coo.col, coo.shape)),
        dense.sum(axis=1),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(native.coo_col_norms(coo.data, coo.row, coo.col, coo.shape)),
        np.linalg.norm(dense, axis=0),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(native.csc_col_sums(csc.data, csc.indices, csc.indptr, csc.shape)),
        dense.sum(axis=0),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(native.csc_col_norms(csc.data, csc.indices, csc.indptr, csc.shape)),
        np.linalg.norm(dense, axis=0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_coo_and_csc_reductions_handle_empty_shapes(mx):
    data = mx.array(np.array([], dtype=np.float32))
    row = mx.array(np.array([], dtype=np.int32))
    col = mx.array(np.array([], dtype=np.int32))
    coo = ms.coo_array((data, (row, col)), shape=(2, 0), canonical=True)
    csc = coo.tocsc(canonical=True)

    np.testing.assert_allclose(to_numpy(coo.row_sums()), np.zeros(2, dtype=np.float32))
    assert coo.col_sums().shape == (0,)
    assert coo.col_norms().shape == (0,)
    assert coo.diagonal().shape == (0,)
    np.testing.assert_allclose(to_numpy(coo.trace()), np.array(0.0, dtype=np.float32))

    np.testing.assert_allclose(to_numpy(csc.row_sums()), np.zeros(2, dtype=np.float32))
    assert csc.col_sums().shape == (0,)
    assert csc.col_norms().shape == (0,)
    assert csc.diagonal().shape == (0,)
    np.testing.assert_allclose(to_numpy(csc.trace()), np.array(0.0, dtype=np.float32))


def test_reduction_fallbacks_are_duplicate_correct(mx):
    coo, _, row, col = _sample_duplicate_coo(mx, "float32", np.int32)
    data_np = to_numpy(coo.data)
    dense = _dense_from_coo(data_np, row, col, coo.shape)
    csc = coo.tocsc(canonical=False)

    np.testing.assert_allclose(
        to_numpy(fallback.coo_row_norms(coo.data, coo.row, coo.col, coo.shape)),
        np.linalg.norm(dense, axis=1),
    )
    np.testing.assert_allclose(
        to_numpy(fallback.coo_col_norms(coo.data, coo.row, coo.col, coo.shape)),
        np.linalg.norm(dense, axis=0),
    )
    np.testing.assert_allclose(
        to_numpy(fallback.csc_row_norms(csc.data, csc.indices, csc.indptr, csc.shape)),
        np.linalg.norm(dense, axis=1),
    )
    np.testing.assert_allclose(
        to_numpy(fallback.csc_col_norms(csc.data, csc.indices, csc.indptr, csc.shape)),
        np.linalg.norm(dense, axis=0),
    )
