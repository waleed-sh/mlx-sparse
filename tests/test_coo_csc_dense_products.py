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


def _sample_arrays(dtype=np.float32, index_dtype=np.int32):
    row = np.array([0, 0, 1, 3, 3, 2, 1], dtype=index_dtype)
    col = np.array([0, 2, 1, 0, 3, 2, 3], dtype=index_dtype)
    data = np.array([2.0, -1.0, 0.5, 3.0, -2.0, 4.0, 1.5], dtype=np.float32)
    if dtype == np.complex64:
        data = data.astype(np.complex64) + 1j * np.array(
            [0.25, -0.5, 1.0, -0.25, 0.75, 0.5, -1.5], dtype=np.float32
        )
    else:
        data = data.astype(dtype)
    dense = np.zeros((4, 5), dtype=data.dtype)
    np.add.at(dense, (row, col), data)
    return data, row, col, dense


def _assert_canonical_coo(coo):
    data = to_numpy(coo.data)
    row = to_numpy(coo.row)
    col = to_numpy(coo.col)
    assert coo.has_canonical_format
    assert np.all(data != 0)
    if row.size:
        order = np.lexsort((col, row))
        np.testing.assert_array_equal(order, np.arange(row.size))
        pairs = np.stack([row, col], axis=1)
        assert np.unique(pairs, axis=0).shape[0] == pairs.shape[0]


def _assert_canonical_csc(csc):
    data = to_numpy(csc.data)
    indices = to_numpy(csc.indices)
    indptr = to_numpy(csc.indptr)
    assert csc.sorted_indices
    assert csc.has_canonical_format
    assert np.all(data != 0)
    assert np.all(np.diff(indptr) >= 0)
    for col in range(csc.shape[1]):
        start, end = indptr[col], indptr[col + 1]
        rows = indices[start:end]
        if rows.size:
            np.testing.assert_array_equal(rows, np.sort(rows))
            assert np.unique(rows).size == rows.size


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (np.float32, 1e-5, 1e-5),
        (np.float16, 6e-3, 6e-3),
        (np.complex64, 1e-5, 1e-5),
    ],
)
def test_coo_dense_products_match_dense_and_scipy(
    mx, scipy_sparse, dtype, rtol, atol, index_dtype
):
    data_np, row_np, col_np, dense_np = _sample_arrays(dtype, index_dtype)
    coo = ms.coo_array(
        (
            mx.array(data_np),
            (mx.array(row_np), mx.array(col_np)),
        ),
        shape=dense_np.shape,
    )
    scipy_data_np = data_np.astype(np.float32) if dtype == np.float16 else data_np
    scipy_coo = scipy_sparse.coo_matrix(
        (scipy_data_np, (row_np, col_np)), dense_np.shape
    )

    x_np = np.linspace(-1.0, 1.0, dense_np.shape[1]).astype(data_np.dtype)
    rhs_np = (
        np.arange(dense_np.shape[1] * 3, dtype=np.float32).reshape(-1, 3) / 7.0
    ).astype(data_np.dtype)
    batched_vec_np = (
        np.arange(2 * dense_np.shape[1], dtype=np.float32).reshape(2, -1) / 5.0
    ).astype(data_np.dtype)
    batched_rhs_np = (
        np.arange(2 * 3 * dense_np.shape[1] * 2, dtype=np.float32).reshape(
            2, 3, dense_np.shape[1], 2
        )
        / 11.0
    ).astype(data_np.dtype)

    np.testing.assert_allclose(
        to_numpy(coo @ mx.array(x_np)),
        (scipy_coo @ x_np.astype(scipy_data_np.dtype)).astype(data_np.dtype),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(coo @ mx.array(rhs_np)),
        dense_np @ rhs_np,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(ms.coo_batched_matvec(coo, mx.array(batched_vec_np))),
        batched_vec_np @ dense_np.T,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(coo @ mx.array(batched_rhs_np)),
        dense_np @ batched_rhs_np,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (np.float32, 1e-5, 1e-5),
        (np.float16, 6e-3, 6e-3),
        (np.complex64, 1e-5, 1e-5),
    ],
)
def test_csc_dense_products_match_dense_and_scipy(
    mx, scipy_sparse, dtype, rtol, atol, index_dtype
):
    data_np, row_np, col_np, dense_np = _sample_arrays(dtype, index_dtype)
    scipy_data_np = data_np.astype(np.float32) if dtype == np.float16 else data_np
    scipy_csc = scipy_sparse.coo_matrix(
        (scipy_data_np, (row_np, col_np)), dense_np.shape
    ).tocsc()
    csc = ms.csc_array(
        (
            mx.array(scipy_csc.data.astype(data_np.dtype, copy=False)),
            mx.array(scipy_csc.indices.astype(index_dtype, copy=False)),
            mx.array(scipy_csc.indptr.astype(index_dtype, copy=False)),
        ),
        shape=dense_np.shape,
        sorted_indices=True,
    )

    x_np = np.linspace(-1.0, 1.0, dense_np.shape[1]).astype(data_np.dtype)
    rhs_np = (
        np.arange(dense_np.shape[1] * 4, dtype=np.float32).reshape(-1, 4) / 9.0
    ).astype(data_np.dtype)
    batched_vec_np = (
        np.arange(3 * dense_np.shape[1], dtype=np.float32).reshape(3, -1) / 4.0
    ).astype(data_np.dtype)
    batched_rhs_np = (
        np.arange(2 * dense_np.shape[1] * 3, dtype=np.float32).reshape(
            2, dense_np.shape[1], 3
        )
        / 13.0
    ).astype(data_np.dtype)

    np.testing.assert_allclose(
        to_numpy(csc @ mx.array(x_np)),
        scipy_csc @ x_np,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(csc @ mx.array(rhs_np)),
        scipy_csc @ rhs_np,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(ms.csc_batched_matvec(csc, mx.array(batched_vec_np))),
        batched_vec_np @ dense_np.T,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(ms.csc_batched_matmul(csc, mx.array(batched_rhs_np))),
        dense_np @ batched_rhs_np,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.csc_matmul_transpose(
                csc.data,
                csc.indices,
                csc.indptr,
                mx.array(np.ones((dense_np.shape[0], 2), dtype=data_np.dtype)),
                csc.shape,
            )
        ),
        dense_np.T @ np.ones((dense_np.shape[0], 2), dtype=data_np.dtype),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (np.float32, 1e-5, 1e-5),
        (np.float16, 6e-3, 6e-3),
        (np.complex64, 1e-5, 1e-5),
    ],
)
def test_coo_and_csc_sparse_sparse_matmat_match_scipy(
    mx, scipy_sparse, dtype, rtol, atol, index_dtype
):
    data_np, row_np, col_np, _ = _sample_arrays(np.float32, np.int32)
    data_np = data_np.astype(np.float32)
    if dtype == np.complex64:
        data_np = data_np.astype(np.complex64) + 1j * np.linspace(
            -0.5, 0.5, data_np.size, dtype=np.float32
        )
    else:
        data_np = data_np.astype(dtype)
    row_np = row_np.astype(index_dtype)
    col_np = col_np.astype(index_dtype)
    rhs_row_np = np.array([0, 1, 2, 2, 3, 4, 4, 2], dtype=index_dtype)
    rhs_col_np = np.array([1, 0, 0, 0, 2, 1, 1, 2], dtype=index_dtype)
    rhs_data_np = np.array([1.0, 2.0, -2.0, 0.5, 3.0, -1.0, 1.0, -4.0])
    if dtype == np.complex64:
        rhs_data_np = rhs_data_np.astype(np.complex64) + 1j * np.linspace(
            0.25, -0.25, rhs_data_np.size, dtype=np.float32
        )
    else:
        rhs_data_np = rhs_data_np.astype(dtype)

    coo = ms.coo_array(
        (mx.array(data_np), (mx.array(row_np), mx.array(col_np))),
        shape=(4, 5),
    )
    rhs_coo = ms.coo_array(
        (mx.array(rhs_data_np), (mx.array(rhs_row_np), mx.array(rhs_col_np))),
        shape=(5, 3),
    )

    scipy_dtype = np.float32 if dtype == np.float16 else dtype
    scipy_a = scipy_sparse.coo_matrix(
        (data_np.astype(scipy_dtype), (row_np, col_np)), shape=(4, 5)
    )
    scipy_b = scipy_sparse.coo_matrix(
        (rhs_data_np.astype(scipy_dtype), (rhs_row_np, rhs_col_np)), shape=(5, 3)
    )
    expected = (scipy_a @ scipy_b).toarray().astype(dtype)

    coo_out = coo @ rhs_coo
    assert isinstance(coo_out, ms.COOArray)
    assert coo_out.has_canonical_format
    np.testing.assert_allclose(
        to_numpy(coo_out.todense()), expected, rtol=rtol, atol=atol
    )
    row_col = np.stack([to_numpy(coo_out.row), to_numpy(coo_out.col)], axis=1)
    if row_col.size:
        order = np.lexsort((row_col[:, 1], row_col[:, 0]))
        np.testing.assert_array_equal(order, np.arange(row_col.shape[0]))

    csc = coo.tocsc(canonical=True)
    rhs_csc = rhs_coo.tocsc(canonical=True)
    csc_out = csc @ rhs_csc
    assert isinstance(csc_out, ms.CSCArray)
    assert csc_out.sorted_indices
    assert csc_out.has_canonical_format
    np.testing.assert_allclose(
        to_numpy(csc_out.todense()), expected, rtol=rtol, atol=atol
    )

    with pytest.raises(NotImplementedError, match="Mixed-format COO"):
        _ = coo @ rhs_csc
    with pytest.raises(NotImplementedError, match="Mixed-format CSC"):
        _ = csc @ rhs_coo


def test_coo_spgemm_duplicate_cancellation_and_rectangular_output(mx, scipy_sparse):
    lhs_row = np.array([0, 0, 0, 1, 1, 2, 3, 3], dtype=np.int32)
    lhs_col = np.array([1, 1, 3, 0, 2, 4, 1, 4], dtype=np.int32)
    lhs_data = np.array([1.0, -1.0, 2.0, 3.0, -2.0, 5.0, 4.0, -5.0], dtype=np.float32)
    rhs_row = np.array([1, 1, 3, 3, 4, 0, 2, 2, 4], dtype=np.int32)
    rhs_col = np.array([0, 0, 2, 3, 1, 1, 0, 3, 1], dtype=np.int32)
    rhs_data = np.array(
        [7.0, -7.0, 2.0, 0.0, 1.0, 2.0, -3.0, 4.0, -1.0], dtype=np.float32
    )

    lhs = ms.coo_array(
        (mx.array(lhs_data), (mx.array(lhs_row), mx.array(lhs_col))),
        shape=(4, 5),
    )
    rhs = ms.coo_array(
        (mx.array(rhs_data), (mx.array(rhs_row), mx.array(rhs_col))),
        shape=(5, 4),
    )

    expected = (
        scipy_sparse.coo_matrix((lhs_data, (lhs_row, lhs_col)), shape=lhs.shape)
        @ scipy_sparse.coo_matrix((rhs_data, (rhs_row, rhs_col)), shape=rhs.shape)
    ).toarray()
    out = lhs @ rhs

    assert out.shape == (4, 4)
    _assert_canonical_coo(out)
    np.testing.assert_allclose(to_numpy(out.todense()), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.cpu_only
def test_coo_csc_spgemm_fixed_parallel_matches_serial_and_scipy(mx, scipy_sparse):
    lhs_row = np.array([0, 0, 1, 1, 2, 2, 3, 4, 4, 4], dtype=np.int32)
    lhs_col = np.array([0, 1, 1, 4, 2, 3, 5, 0, 4, 5], dtype=np.int32)
    lhs_data = np.array(
        [1.0, 1.0, 3.0, -2.0, 4.0, -1.0, 0.5, 2.5, -3.0, 1.5],
        dtype=np.float32,
    )
    rhs_row = np.array([0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5], dtype=np.int32)
    rhs_col = np.array([0, 0, 2, 5, 1, 3, 3, 0, 4, 5, 2], dtype=np.int32)
    rhs_data = np.array(
        [2.0, -2.0, 1.5, -0.5, 3.0, 4.0, -4.0, 2.0, -1.0, 0.25, 5.0],
        dtype=np.float32,
    )
    scipy_lhs = scipy_sparse.coo_matrix((lhs_data, (lhs_row, lhs_col)), shape=(5, 6))
    scipy_rhs = scipy_sparse.coo_matrix((rhs_data, (rhs_row, rhs_col)), shape=(6, 6))
    expected = scipy_lhs @ scipy_rhs
    expected.eliminate_zeros()

    lhs_coo = ms.from_scipy(scipy_lhs, format="coo")
    rhs_coo = ms.from_scipy(scipy_rhs, format="coo")
    with ms.runtime.context(spgemm_parallel=False):
        serial_coo = lhs_coo @ rhs_coo
    with ms.runtime.context(spgemm_parallel=True, spgemm_threads=2):
        parallel_coo = lhs_coo @ rhs_coo

    _assert_canonical_coo(serial_coo)
    _assert_canonical_coo(parallel_coo)
    np.testing.assert_allclose(
        to_numpy(parallel_coo.todense()), expected.toarray(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(to_numpy(parallel_coo.row), to_numpy(serial_coo.row))
    np.testing.assert_array_equal(to_numpy(parallel_coo.col), to_numpy(serial_coo.col))
    np.testing.assert_allclose(to_numpy(parallel_coo.data), to_numpy(serial_coo.data))

    lhs_csc = ms.from_scipy(scipy_lhs, format="csc")
    rhs_csc = ms.from_scipy(scipy_rhs, format="csc")
    with ms.runtime.context(spgemm_parallel=False):
        serial_csc = lhs_csc @ rhs_csc
    with ms.runtime.context(spgemm_parallel=True, spgemm_threads=2):
        parallel_csc = lhs_csc @ rhs_csc

    _assert_canonical_csc(serial_csc)
    _assert_canonical_csc(parallel_csc)
    np.testing.assert_allclose(
        to_numpy(parallel_csc.todense()), expected.toarray(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(
        to_numpy(parallel_csc.indptr), to_numpy(serial_csc.indptr)
    )
    np.testing.assert_array_equal(
        to_numpy(parallel_csc.indices), to_numpy(serial_csc.indices)
    )
    np.testing.assert_allclose(to_numpy(parallel_csc.data), to_numpy(serial_csc.data))


def test_coo_spgemm_dense_ordered_extraction_keeps_canonical(mx, scipy_sparse):
    block = 8
    blocks = 16
    n_cols = block * blocks
    lhs_row = np.zeros(blocks, dtype=np.int32)
    lhs_col = np.arange(blocks, dtype=np.int32)
    lhs_data = np.ones(blocks, dtype=np.float32)

    rhs_row = np.repeat(np.arange(blocks, dtype=np.int32), block)
    rhs_col_parts = []
    for rhs_row_index in range(blocks):
        start = (blocks - rhs_row_index - 1) * block
        rhs_col_parts.append(np.arange(start, start + block, dtype=np.int32))
    rhs_col = np.concatenate(rhs_col_parts)
    rhs_data = np.linspace(0.25, 1.5, rhs_col.size, dtype=np.float32)

    lhs = ms.coo_array(
        (mx.array(lhs_data), (mx.array(lhs_row), mx.array(lhs_col))),
        shape=(1, blocks),
    )
    rhs = ms.coo_array(
        (mx.array(rhs_data), (mx.array(rhs_row), mx.array(rhs_col))),
        shape=(blocks, n_cols),
    )
    expected = scipy_sparse.coo_matrix(
        (lhs_data, (lhs_row, lhs_col)), shape=lhs.shape
    ) @ scipy_sparse.coo_matrix((rhs_data, (rhs_row, rhs_col)), shape=rhs.shape)

    out = lhs @ rhs

    _assert_canonical_coo(out)
    np.testing.assert_allclose(to_numpy(out.todense()), expected.toarray())
    np.testing.assert_array_equal(to_numpy(out.col), np.arange(n_cols))


def test_coo_spgemm_empty_product_preserves_output_shape_and_dtype(mx):
    lhs = ms.coo_array(
        (
            mx.array(np.array([2.0, -3.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 2], dtype=np.int32)),
                mx.array(np.array([0, 0], dtype=np.int32)),
            ),
        ),
        shape=(3, 4),
    )
    rhs = ms.coo_array(
        (
            mx.array(np.array([4.0, 5.0], dtype=np.float32)),
            (
                mx.array(np.array([1, 3], dtype=np.int32)),
                mx.array(np.array([0, 1], dtype=np.int32)),
            ),
        ),
        shape=(4, 2),
    )

    out = lhs @ rhs

    assert out.shape == (3, 2)
    assert out.nnz == 0
    assert out.dtype == lhs.dtype
    assert out.index_dtype == lhs.index_dtype
    assert out.has_canonical_format
    np.testing.assert_array_equal(
        to_numpy(out.todense()), np.zeros((3, 2), dtype=np.float32)
    )


def test_native_coo_spgemm_mixed_index_dtypes_promotes_output_indices(mx, scipy_sparse):
    lhs = ms.coo_array(
        (
            mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 1, 1], dtype=np.int32)),
                mx.array(np.array([0, 1, 2], dtype=np.int32)),
            ),
        ),
        shape=(2, 3),
    )
    rhs = ms.coo_array(
        (
            mx.array(np.array([4.0, -1.0, 5.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 1, 2], dtype=np.int64)),
                mx.array(np.array([1, 0, 1], dtype=np.int64)),
            ),
        ),
        shape=(3, 2),
    )

    data, row, col = native.coo_matmat(lhs, rhs)
    expected = (
        scipy_sparse.coo_matrix(
            (
                to_numpy(lhs.data),
                (
                    to_numpy(lhs.row).astype(np.int64),
                    to_numpy(lhs.col).astype(np.int64),
                ),
            ),
            shape=lhs.shape,
        )
        @ scipy_sparse.coo_matrix(
            (to_numpy(rhs.data), (to_numpy(rhs.row), to_numpy(rhs.col))),
            shape=rhs.shape,
        )
    ).toarray()
    out = ms.COOArray(data, row, col, (2, 2), has_canonical_format=True)

    assert row.dtype == mx.int64
    assert col.dtype == mx.int64
    _assert_canonical_coo(out)
    np.testing.assert_allclose(to_numpy(out.todense()), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_experimental_metal_coo_spgemm_matches_scipy(mx, scipy_sparse):
    lhs_row = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
    lhs_col = np.array([0, 2, 1, 3, 0, 2, 2], dtype=np.int32)
    lhs_data = np.array([1.0, -2.0, 3.0, 4.0, -1.0, 2.5, -0.5], dtype=np.float32)
    rhs_row = np.array([0, 1, 2, 2, 3], dtype=np.int32)
    rhs_col = np.array([1, 0, 0, 2, 1], dtype=np.int32)
    rhs_data = np.array([2.0, -1.0, 5.0, -5.0, 0.5], dtype=np.float32)

    lhs = ms.coo_array(
        (mx.array(lhs_data), (mx.array(lhs_row), mx.array(lhs_col))),
        shape=(3, 4),
    )
    rhs = ms.coo_array(
        (mx.array(rhs_data), (mx.array(rhs_row), mx.array(rhs_col))),
        shape=(4, 3),
    )

    expected = (
        scipy_sparse.coo_matrix((lhs_data, (lhs_row, lhs_col)), shape=lhs.shape)
        @ scipy_sparse.coo_matrix((rhs_data, (rhs_row, rhs_col)), shape=rhs.shape)
    ).toarray()
    with ms.config.patch(EXPERIMENTAL_METAL_SPGEMM=True):
        out = lhs @ rhs

    assert isinstance(out, ms.COOArray)
    _assert_canonical_coo(out)
    np.testing.assert_allclose(to_numpy(out.todense()), expected, rtol=1e-5, atol=1e-5)


def test_csc_spgemm_duplicate_cancellation_and_rectangular_output(mx, scipy_sparse):
    lhs_indices = np.array([1, 0, 0, 3, 1, 0, 2, 3], dtype=np.int32)
    lhs_indptr = np.array([0, 1, 4, 5, 6, 8], dtype=np.int32)
    lhs_data = np.array([3.0, 1.0, -1.0, 4.0, -2.0, 2.0, 5.0, -5.0], dtype=np.float32)
    rhs_indices = np.array([1, 1, 2, 0, 4, 4, 3, 2, 3], dtype=np.int32)
    rhs_indptr = np.array([0, 3, 6, 7, 9], dtype=np.int32)
    rhs_data = np.array(
        [7.0, -7.0, -3.0, 2.0, 1.0, -1.0, 2.0, 4.0, 0.0], dtype=np.float32
    )

    lhs = ms.csc_array(
        (mx.array(lhs_data), mx.array(lhs_indices), mx.array(lhs_indptr)),
        shape=(4, 5),
        sorted_indices=False,
    )
    rhs = ms.csc_array(
        (mx.array(rhs_data), mx.array(rhs_indices), mx.array(rhs_indptr)),
        shape=(5, 4),
        sorted_indices=False,
    )

    expected = (
        scipy_sparse.csc_matrix((lhs_data, lhs_indices, lhs_indptr), shape=lhs.shape)
        @ scipy_sparse.csc_matrix((rhs_data, rhs_indices, rhs_indptr), shape=rhs.shape)
    ).toarray()
    out = lhs @ rhs

    assert isinstance(out, ms.CSCArray)
    assert out.shape == (4, 4)
    _assert_canonical_csc(out)
    np.testing.assert_allclose(to_numpy(out.todense()), expected, rtol=1e-5, atol=1e-5)


def test_csc_spgemm_dense_ordered_extraction_keeps_canonical(mx, scipy_sparse):
    block = 8
    blocks = 16
    n_rows = block * blocks
    lhs_indices_parts = []
    for lhs_col_index in range(blocks):
        start = (blocks - lhs_col_index - 1) * block
        lhs_indices_parts.append(np.arange(start, start + block, dtype=np.int32))
    lhs_indices = np.concatenate(lhs_indices_parts)
    lhs_data = np.linspace(0.25, 1.5, lhs_indices.size, dtype=np.float32)
    lhs_indptr = np.arange(blocks + 1, dtype=np.int32) * block

    rhs_indices = np.arange(blocks, dtype=np.int32)
    rhs_indptr = np.array([0, blocks], dtype=np.int32)
    rhs_data = np.ones(blocks, dtype=np.float32)

    lhs = ms.csc_array(
        (mx.array(lhs_data), mx.array(lhs_indices), mx.array(lhs_indptr)),
        shape=(n_rows, blocks),
        sorted_indices=True,
    )
    rhs = ms.csc_array(
        (mx.array(rhs_data), mx.array(rhs_indices), mx.array(rhs_indptr)),
        shape=(blocks, 1),
        sorted_indices=True,
    )
    expected = scipy_sparse.csc_matrix(
        (lhs_data, lhs_indices, lhs_indptr), shape=lhs.shape
    ) @ scipy_sparse.csc_matrix((rhs_data, rhs_indices, rhs_indptr), shape=rhs.shape)

    out = lhs @ rhs

    _assert_canonical_csc(out)
    np.testing.assert_allclose(to_numpy(out.todense()), expected.toarray())
    np.testing.assert_array_equal(to_numpy(out.indices), np.arange(n_rows))


def test_csc_spgemm_empty_product_preserves_output_shape_and_dtype(mx):
    lhs = ms.csc_array(
        (
            mx.array(np.array([2.0, -3.0], dtype=np.float32)),
            mx.array(np.array([0, 2], dtype=np.int32)),
            mx.array(np.array([0, 2, 2, 2, 2], dtype=np.int32)),
        ),
        shape=(3, 4),
        sorted_indices=True,
    )
    rhs = ms.csc_array(
        (
            mx.array(np.array([4.0, 5.0], dtype=np.float32)),
            mx.array(np.array([1, 3], dtype=np.int32)),
            mx.array(np.array([0, 1, 2], dtype=np.int32)),
        ),
        shape=(4, 2),
        sorted_indices=True,
    )

    out = lhs @ rhs

    assert out.shape == (3, 2)
    assert out.nnz == 0
    assert out.dtype == lhs.dtype
    assert out.index_dtype == lhs.index_dtype
    assert out.sorted_indices
    assert out.has_canonical_format
    np.testing.assert_array_equal(
        to_numpy(out.todense()), np.zeros((3, 2), dtype=np.float32)
    )


def test_native_csc_spgemm_mixed_index_dtypes_promotes_output_indices(mx, scipy_sparse):
    lhs = ms.csc_array(
        (
            mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            mx.array(np.array([0, 1, 1], dtype=np.int32)),
            mx.array(np.array([0, 1, 2, 3], dtype=np.int32)),
        ),
        shape=(2, 3),
        sorted_indices=True,
    )
    rhs = ms.csc_array(
        (
            mx.array(np.array([4.0, -1.0, 5.0], dtype=np.float32)),
            mx.array(np.array([1, 0, 2], dtype=np.int64)),
            mx.array(np.array([0, 1, 3], dtype=np.int64)),
        ),
        shape=(3, 2),
        sorted_indices=True,
    )

    data, indices, indptr = native.csc_matmat(lhs, rhs)
    expected = (
        scipy_sparse.csc_matrix(
            (
                to_numpy(lhs.data),
                to_numpy(lhs.indices).astype(np.int64),
                to_numpy(lhs.indptr).astype(np.int64),
            ),
            shape=lhs.shape,
        )
        @ scipy_sparse.csc_matrix(
            (to_numpy(rhs.data), to_numpy(rhs.indices), to_numpy(rhs.indptr)),
            shape=rhs.shape,
        )
    ).toarray()
    out = ms.CSCArray(
        data,
        indices,
        indptr,
        (2, 2),
        sorted_indices=True,
        has_canonical_format=True,
    )

    assert indices.dtype == mx.int64
    assert indptr.dtype == mx.int64
    _assert_canonical_csc(out)
    np.testing.assert_allclose(to_numpy(out.todense()), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_experimental_metal_csc_spgemm_matches_scipy(mx, scipy_sparse):
    lhs_indices = np.array([0, 2, 1, 0, 2, 2, 2], dtype=np.int32)
    lhs_indptr = np.array([0, 2, 3, 6, 7], dtype=np.int32)
    lhs_data = np.array([1.0, -1.0, 3.0, -2.0, 2.5, -0.5, 4.0], dtype=np.float32)
    rhs_indices = np.array([1, 0, 2, 2, 3], dtype=np.int32)
    rhs_indptr = np.array([0, 1, 3, 5], dtype=np.int32)
    rhs_data = np.array([-1.0, 2.0, 5.0, -5.0, 0.5], dtype=np.float32)

    lhs = ms.csc_array(
        (mx.array(lhs_data), mx.array(lhs_indices), mx.array(lhs_indptr)),
        shape=(3, 4),
        sorted_indices=False,
    )
    rhs = ms.csc_array(
        (mx.array(rhs_data), mx.array(rhs_indices), mx.array(rhs_indptr)),
        shape=(4, 3),
        sorted_indices=False,
    )

    expected = (
        scipy_sparse.csc_matrix((lhs_data, lhs_indices, lhs_indptr), shape=lhs.shape)
        @ scipy_sparse.csc_matrix((rhs_data, rhs_indices, rhs_indptr), shape=rhs.shape)
    ).toarray()
    with ms.config.patch(EXPERIMENTAL_METAL_SPGEMM=True):
        out = lhs @ rhs

    assert isinstance(out, ms.CSCArray)
    _assert_canonical_csc(out)
    np.testing.assert_allclose(to_numpy(out.todense()), expected, rtol=1e-5, atol=1e-5)
