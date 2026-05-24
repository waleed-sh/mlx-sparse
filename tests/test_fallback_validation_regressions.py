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
import mlx_sparse._validation as validation
from mlx_sparse._host import to_numpy


def _coo_problem(index_dtype=np.int32):
    row = np.array([0, 0, 1, 2, 3, 3, 3], dtype=index_dtype)
    col = np.array([0, 2, 1, 2, 0, 3, 3], dtype=index_dtype)
    data = np.array([2.0, -1.0, 0.5, 4.0, 3.0, -2.0, 1.0], dtype=np.float32)
    dense = np.zeros((4, 5), dtype=np.float32)
    np.add.at(dense, (row, col), data)
    return data, row, col, dense


def _make_csr(mx):
    data, row, col, dense = _coo_problem()
    coo = ms.coo_array((mx.array(data), (mx.array(row), mx.array(col))), dense.shape)
    return coo.tocsr(canonical=True), dense


def _make_coo(mx):
    data, row, col, dense = _coo_problem()
    return (
        ms.coo_array((mx.array(data), (mx.array(row), mx.array(col))), dense.shape),
        dense,
    )


def _make_csc(mx):
    data, row, col, dense = _coo_problem()
    coo = ms.coo_array((mx.array(data), (mx.array(row), mx.array(col))), dense.shape)
    return coo.tocsc(canonical=True), dense


def test_python_fallback_covers_coo_csc_dense_product_paths(monkeypatch, mx):
    monkeypatch.setattr(native, "extension", lambda: None)
    data, row, col, dense = _coo_problem()

    data_mx = mx.array(data)
    row_mx = mx.array(row)
    col_mx = mx.array(col)
    csc_data, csc_indices, csc_indptr = native.coo_tocsc(
        data_mx, row_mx, col_mx, dense.shape
    )
    coo_dense = fallback.coo_todense(data_mx, row_mx, col_mx, dense.shape)
    csc_dense = native.csc_todense(csc_data, csc_indices, csc_indptr, dense.shape)
    np.testing.assert_allclose(to_numpy(coo_dense), dense)
    np.testing.assert_allclose(to_numpy(csc_dense), dense)

    x = np.linspace(-1.0, 1.0, dense.shape[1], dtype=np.float32)
    tx = np.linspace(0.25, 1.25, dense.shape[0], dtype=np.float32)
    rhs = (np.arange(dense.shape[1] * 3, dtype=np.float32).reshape(-1, 3) - 4.0) / 5.0
    trhs = (np.arange(dense.shape[0] * 2, dtype=np.float32).reshape(-1, 2) + 1.0) / 7.0
    batched_vecs = np.arange(2 * 3 * dense.shape[1], dtype=np.float32).reshape(
        2, 3, dense.shape[1]
    )
    batched_mats = (
        np.arange(2 * dense.shape[1] * 4, dtype=np.float32).reshape(
            2, dense.shape[1], 4
        )
        / 11.0
    )

    np.testing.assert_allclose(
        to_numpy(native.coo_matvec(data_mx, row_mx, col_mx, mx.array(x), dense.shape)),
        dense @ x,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.csc_matvec(
                csc_data, csc_indices, csc_indptr, mx.array(x), dense.shape
            )
        ),
        dense @ x,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.csc_matvec_transpose(
                csc_data, csc_indices, csc_indptr, mx.array(tx), dense.shape
            )
        ),
        dense.T @ tx,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.coo_matmul(data_mx, row_mx, col_mx, mx.array(rhs), dense.shape)
        ),
        dense @ rhs,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.csc_matmul(
                csc_data, csc_indices, csc_indptr, mx.array(rhs), dense.shape
            )
        ),
        dense @ rhs,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.csc_matmul_transpose(
                csc_data, csc_indices, csc_indptr, mx.array(trhs), dense.shape
            )
        ),
        dense.T @ trhs,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.coo_batched_matvec(
                data_mx, row_mx, col_mx, mx.array(batched_vecs), dense.shape
            )
        ),
        batched_vecs @ dense.T,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.csc_batched_matvec(
                csc_data, csc_indices, csc_indptr, mx.array(batched_vecs), dense.shape
            )
        ),
        batched_vecs @ dense.T,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.coo_batched_matmul(
                data_mx, row_mx, col_mx, mx.array(batched_mats), dense.shape
            )
        ),
        dense[None, :, :] @ batched_mats,
    )
    np.testing.assert_allclose(
        to_numpy(
            native.csc_batched_matmul(
                csc_data, csc_indices, csc_indptr, mx.array(batched_mats), dense.shape
            )
        ),
        dense[None, :, :] @ batched_mats,
    )


def test_python_fallback_covers_csc_conversion_sort_and_duplicates(monkeypatch, mx):
    monkeypatch.setattr(native, "extension", lambda: None)
    csr, dense = _make_csr(mx)

    csc_data, csc_indices, csc_indptr = native.csr_tocsc(
        csr.data, csr.indices, csr.indptr, csr.shape
    )
    roundtrip = native.csc_tocsr(csc_data, csc_indices, csc_indptr, csr.shape)
    np.testing.assert_allclose(
        to_numpy(ms.csr_array(roundtrip, shape=csr.shape).todense()),
        dense,
    )

    unsorted_data = mx.array(np.array([3.0, 1.0, 2.0, 7.0], dtype=np.float32))
    unsorted_indices = mx.array(np.array([2, 0, 1, 0], dtype=np.int32))
    unsorted_indptr = mx.array(np.array([0, 3, 4], dtype=np.int32))
    sorted_data, sorted_indices, returned_indptr = native.csc_sort_indices(
        unsorted_data, unsorted_indices, unsorted_indptr
    )
    np.testing.assert_allclose(to_numpy(sorted_data), [1.0, 2.0, 3.0, 7.0])
    np.testing.assert_array_equal(to_numpy(sorted_indices), [0, 1, 2, 0])
    assert returned_indptr is unsorted_indptr

    dup_data = mx.array(np.array([1.0, 2.5, -0.5, 4.0], dtype=np.float32))
    dup_indices = mx.array(np.array([2, 2, 0, 0], dtype=np.int32))
    dup_indptr = mx.array(np.array([0, 3, 4], dtype=np.int32))
    summed_data, summed_indices, summed_indptr = native.csc_sum_duplicates(
        dup_data, dup_indices, dup_indptr
    )
    np.testing.assert_allclose(to_numpy(summed_data), [3.5, -0.5, 4.0])
    np.testing.assert_array_equal(to_numpy(summed_indices), [2, 0, 0])
    np.testing.assert_array_equal(to_numpy(summed_indptr), [0, 2, 3])

    empty_data = mx.array(np.array([], dtype=np.float32))
    empty_indices = mx.array(np.array([], dtype=np.int32))
    empty_indptr = mx.array(np.array([0, 0, 0], dtype=np.int32))
    empty = native.csc_sum_duplicates(empty_data, empty_indices, empty_indptr)
    assert empty[0].size == 0
    assert empty[1].size == 0
    np.testing.assert_array_equal(to_numpy(empty[2]), [0, 0, 0])


def test_python_fallback_sparse_sparse_matmul_regressions(monkeypatch, mx):
    monkeypatch.setattr(native, "extension", lambda: None)

    lhs_coo = ms.coo_array(
        (
            mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 0, 1], dtype=np.int32)),
                mx.array(np.array([0, 1, 2], dtype=np.int32)),
            ),
        ),
        shape=(2, 3),
    )
    rhs_coo = ms.coo_array(
        (
            mx.array(np.array([4.0, -2.0, 5.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 1, 2], dtype=np.int64)),
                mx.array(np.array([1, 1, 0], dtype=np.int64)),
            ),
        ),
        shape=(3, 2),
    )
    coo_out = ms.coo_matmat(lhs_coo, rhs_coo)
    assert coo_out.index_dtype == mx.int64
    np.testing.assert_allclose(to_numpy(coo_out.todense()), [[0.0, 0.0], [15.0, 0.0]])

    lhs_csc = ms.csc_array(
        (
            mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            mx.array(np.array([0, 0, 1], dtype=np.int32)),
            mx.array(np.array([0, 1, 2, 3], dtype=np.int32)),
        ),
        shape=(2, 3),
        sorted_indices=True,
        canonical=True,
    )
    rhs_csc = ms.csc_array(
        (
            mx.array(np.array([5.0, 4.0, -2.0], dtype=np.float32)),
            mx.array(np.array([2, 0, 1], dtype=np.int64)),
            mx.array(np.array([0, 1, 3], dtype=np.int64)),
        ),
        shape=(3, 2),
        sorted_indices=True,
        canonical=True,
    )
    csc_out = ms.csc_matmat(lhs_csc, rhs_csc)
    assert csc_out.index_dtype == mx.int64
    np.testing.assert_allclose(to_numpy(csc_out.todense()), [[0.0, 0.0], [15.0, 0.0]])

    cancel_lhs = ms.csc_array(
        (
            mx.array(np.array([1.0, -1.0], dtype=np.float32)),
            mx.array(np.array([0, 0], dtype=np.int32)),
            mx.array(np.array([0, 1, 2], dtype=np.int32)),
        ),
        shape=(1, 2),
        sorted_indices=True,
        canonical=True,
    )
    cancel_rhs = ms.csc_array(
        (
            mx.array(np.array([2.0, 2.0], dtype=np.float32)),
            mx.array(np.array([0, 1], dtype=np.int64)),
            mx.array(np.array([0, 2], dtype=np.int64)),
        ),
        shape=(2, 1),
        sorted_indices=True,
        canonical=True,
    )
    empty_data, empty_indices, empty_indptr = native.csc_matmat(cancel_lhs, cancel_rhs)
    assert empty_data.size == 0
    assert empty_indices.dtype == mx.int64
    np.testing.assert_array_equal(to_numpy(empty_indptr), [0, 0])

    with pytest.raises(ValueError, match="dimension mismatch"):
        native.coo_matmat(lhs_coo, lhs_coo)
    with pytest.raises(TypeError, match="matching value dtypes"):
        native.csc_matmat(
            lhs_csc,
            ms.csc_array(
                (
                    mx.array(np.array([1.0], dtype=np.float16)),
                    mx.array(np.array([0], dtype=np.int32)),
                    mx.array(np.array([0, 1], dtype=np.int32)),
                ),
                shape=(3, 1),
            ),
        )


def test_public_ops_validation_and_alias_paths(mx):
    csr, dense = _make_csr(mx)
    coo, _ = _make_coo(mx)
    csc, _ = _make_csc(mx)

    np.testing.assert_allclose(
        to_numpy(ms.csr_column_sums(csr)),
        dense.sum(axis=0),
    )
    unsorted = ms.csr_array(
        (
            mx.array(np.array([3.0, 4.0, 0.0], dtype=np.float32)),
            mx.array(np.array([1, 1, 0], dtype=np.int32)),
            mx.array(np.array([0, 3], dtype=np.int32)),
        ),
        shape=(1, 2),
    )
    np.testing.assert_allclose(to_numpy(ms.csr_row_norms(unsorted)), [7.0])

    with pytest.raises(TypeError, match="todense expects"):
        ms.todense(object())
    with pytest.raises(TypeError, match="CSRArray"):
        ms.csr_row_sums(coo)
    with pytest.raises(TypeError, match="CSCArray"):
        ms.csc_matvec(csr, mx.zeros((csr.shape[1],), dtype=mx.float32))
    with pytest.raises(TypeError, match="COOArray"):
        ms.coo_matvec(csr, mx.zeros((csr.shape[1],), dtype=mx.float32))

    with pytest.raises(ValueError, match="rank-2 or higher"):
        ms.csr_batched_matvec(csr, mx.zeros((csr.shape[1],), dtype=mx.float32))
    with pytest.raises(ValueError, match="vector dimension"):
        ms.coo_batched_matvec(coo, mx.zeros((2, coo.shape[1] + 1), dtype=mx.float32))
    with pytest.raises(TypeError, match="same dtype"):
        ms.csc_batched_matvec(csc, mx.zeros((2, csc.shape[1]), dtype=mx.float16))

    with pytest.raises(ValueError, match="rank-3 or higher"):
        ms.coo_batched_matmul(coo, mx.zeros((coo.shape[1], 2), dtype=mx.float32))
    with pytest.raises(ValueError, match="sparse dimension"):
        ms.csc_batched_matmul(csc, mx.zeros((2, csc.shape[1] + 1, 2), dtype=mx.float32))
    with pytest.raises(TypeError, match="same dtype"):
        ms.csr_batched_matmul(csr, mx.zeros((2, csr.shape[1], 2), dtype=mx.float16))

    with pytest.raises(ValueError, match="rank-2 or higher"):
        ms.coo_matmul(coo, mx.array(1.0, dtype=mx.float32))
    with pytest.raises(ValueError, match="sparse dimension"):
        ms.csc_matmul(csc, mx.zeros((2, csc.shape[1] + 1, 2), dtype=mx.float32))
    with pytest.raises(TypeError, match="same dtype"):
        ms.csc_matmul(csc, mx.zeros((2, csc.shape[1], 2), dtype=mx.float16))

    with pytest.raises(TypeError, match="COOArray rhs"):
        ms.coo_matmat(coo, csr)
    with pytest.raises(ValueError, match="dimension mismatch"):
        ms.csc_matmat(csc, csc)
    with pytest.raises(TypeError, match="matching value dtypes"):
        ms.coo_matmat(
            coo,
            ms.coo_array(
                (
                    mx.array(np.array([1.0], dtype=np.float16)),
                    (
                        mx.array(np.array([0], dtype=np.int32)),
                        mx.array(np.array([0], dtype=np.int32)),
                    ),
                ),
                shape=(coo.shape[1], 1),
            ),
        )


def test_sparse_container_edge_methods(mx):
    csc, dense = _make_csc(mx)
    csr_raw = csc.tocsr(canonical=False)
    assert not csr_raw.has_canonical_format
    np.testing.assert_allclose(to_numpy(csr_raw.todense()), dense)

    np.testing.assert_allclose(to_numpy(csc.T.todense()), dense.T)
    complex_csc = ms.csc_array(
        (
            mx.array(np.array([1.0 + 2.0j], dtype=np.complex64)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
        ),
        shape=(1, 1),
        sorted_indices=True,
        canonical=True,
    )
    np.testing.assert_allclose(to_numpy(complex_csc.conjugate().data), [1.0 - 2.0j])
    np.testing.assert_allclose(to_numpy(complex_csc.H.todense()), [[1.0 - 2.0j]])

    with pytest.raises(NotImplementedError, match="Mixed-format CSC"):
        _ = csc @ csc.tocsr(canonical=True)
    with pytest.raises(ValueError, match="rank-1 or higher"):
        _ = csc @ mx.array(1.0, dtype=mx.float32)
    with pytest.raises(TypeError, match="csc_array expects"):
        ms.csc_array(object(), shape=(1, 1))
    with pytest.raises(ValueError, match="shape mismatch"):
        ms.csc_array(csc, shape=(csc.shape[0] + 1, csc.shape[1]))


def test_constructor_conversion_contracts(mx, scipy_sparse):
    matrix = scipy_sparse.coo_matrix(
        (
            np.array([1.0, 2.0, -3.0], dtype=np.float32),
            (np.array([0, 0, 1]), np.array([1, 1, 0])),
        ),
        shape=(2, 3),
    )

    csc = ms.from_scipy(matrix, format="csc", canonical=False, index_dtype=mx.int64)
    assert isinstance(csc, ms.CSCArray)
    assert csc.index_dtype == mx.int64
    assert not csc.has_canonical_format
    np.testing.assert_allclose(to_numpy(csc.todense()), matrix.toarray())

    coo = ms.from_scipy(matrix, format="coo", canonical=False, dtype=mx.float16)
    assert isinstance(coo, ms.COOArray)
    assert coo.dtype == mx.float16
    assert not coo.has_canonical_format
    np.testing.assert_allclose(to_numpy(coo.todense()), matrix.toarray(), atol=1e-3)

    same = ms.asarray(csc)
    assert same is csc
    cast = ms.asarray(csc, dtype=mx.float16)
    assert isinstance(cast, ms.CSCArray)
    assert cast.dtype == mx.float16
    assert cast.indices is csc.indices
    np.testing.assert_allclose(to_numpy(cast.todense()), matrix.toarray(), atol=1e-3)

    from_numpy = ms.from_numpy(np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32))
    assert isinstance(from_numpy, ms.CSRArray)
    np.testing.assert_allclose(to_numpy(from_numpy.todense()), [[0.0, 2.0], [3.0, 0.0]])


def test_low_level_validation_rejects_corrupt_sparse_metadata(mx):
    data = mx.array(np.array([1.0, 2.0], dtype=np.float32))
    indices = mx.array(np.array([0, 1], dtype=np.int32))
    indptr = mx.array(np.array([0, 1, 2], dtype=np.int32))

    validation.validate_csc_values(
        mx.array(np.array([], dtype=np.int32)),
        mx.array(np.array([0, 0, 0], dtype=np.int32)),
        (2, 2),
        0,
    )
    validation.validate_coo_values(
        mx.array(np.array([], dtype=np.int32)),
        mx.array(np.array([], dtype=np.int32)),
        (2, 2),
    )

    with pytest.raises(TypeError, match="same dtype"):
        validation.validate_csc_metadata(
            data,
            indices,
            mx.array(np.array([0, 1, 2], dtype=np.int64)),
            (2, 2),
        )
    with pytest.raises(ValueError, match="same length"):
        validation.validate_csc_metadata(data[:1], indices, indptr, (2, 2))
    with pytest.raises(ValueError, match="n_cols"):
        validation.validate_csc_metadata(data, indices, indptr, (2, 3))

    with pytest.raises(ValueError, match=r"indptr\[0\]"):
        validation.validate_csc_values(
            indices, mx.array(np.array([1, 1, 2])), (2, 2), 2
        )
    with pytest.raises(ValueError, match=r"indptr\[-1\]"):
        validation.validate_csc_values(
            indices, mx.array(np.array([0, 1, 1])), (2, 2), 2
        )
    with pytest.raises(ValueError, match="monotonically"):
        validation.validate_csc_values(
            indices, mx.array(np.array([0, 2, 1, 2])), (2, 3), 2
        )
    with pytest.raises(ValueError, match="in bounds"):
        validation.validate_csc_values(
            mx.array(np.array([0, 2], dtype=np.int32)),
            indptr,
            (2, 2),
            2,
        )
    with pytest.raises(ValueError, match="row coordinates"):
        validation.validate_coo_values(
            mx.array(np.array([-1], dtype=np.int32)),
            mx.array(np.array([0], dtype=np.int32)),
            (2, 2),
        )
    with pytest.raises(ValueError, match="col coordinates"):
        validation.validate_coo_values(
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([2], dtype=np.int32)),
            (2, 2),
        )


@pytest.mark.parametrize(
    ("validator", "rhs", "message"),
    [
        (
            validation.validate_csr_matvec_inputs,
            lambda mx: mx.zeros((1, 2), dtype=mx.float32),
            "rank-1",
        ),
        (
            validation.validate_csc_matvec_inputs,
            lambda mx: mx.zeros((3,), dtype=mx.float32),
            "length",
        ),
        (
            validation.validate_csc_matvec_transpose_inputs,
            lambda mx: mx.zeros((3,), dtype=mx.float32),
            "n_rows",
        ),
        (
            validation.validate_coo_matvec_inputs,
            lambda mx: mx.zeros((2,), dtype=mx.float16),
            "same dtype",
        ),
    ],
)
def test_low_level_matvec_validation_errors(mx, validator, rhs, message):
    data = mx.array(np.array([1.0], dtype=np.float32))
    indices = mx.array(np.array([0], dtype=np.int32))
    indptr = mx.array(np.array([0, 1, 1], dtype=np.int32))

    if validator is validation.validate_coo_matvec_inputs:
        args = (data, indices, indices, rhs(mx), (1, 2))
    elif validator is validation.validate_csr_matvec_inputs:
        args = (data, indices, indptr, rhs(mx), (2, 2))
    else:
        args = (data, indices, indptr, rhs(mx), (1, 2))

    with pytest.raises((TypeError, ValueError), match=message):
        validator(*args)


@pytest.mark.parametrize(
    ("validator", "rhs", "message"),
    [
        (
            validation.validate_csr_matmul_inputs,
            lambda mx: mx.zeros((2,), dtype=mx.float32),
            "rank-2",
        ),
        (
            validation.validate_csc_matmul_inputs,
            lambda mx: mx.zeros((3, 1), dtype=mx.float32),
            "leading dimension",
        ),
        (
            validation.validate_coo_matmul_inputs,
            lambda mx: mx.zeros((2, 1), dtype=mx.float16),
            "same dtype",
        ),
    ],
)
def test_low_level_matmul_validation_errors(mx, validator, rhs, message):
    data = mx.array(np.array([1.0], dtype=np.float32))
    indices = mx.array(np.array([0], dtype=np.int32))
    indptr = mx.array(np.array([0, 1, 1], dtype=np.int32))

    if validator is validation.validate_coo_matmul_inputs:
        args = (data, indices, indices, rhs(mx), (1, 2))
    elif validator is validation.validate_csr_matmul_inputs:
        args = (data, indices, indptr, rhs(mx), (2, 2))
    else:
        args = (data, indices, indptr, rhs(mx), (1, 2))

    with pytest.raises((TypeError, ValueError), match=message):
        validator(*args)
