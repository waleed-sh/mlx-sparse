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


def _complex_energy(mx, y):
    return mx.sum(mx.real(y * mx.conjugate(y)))


def test_csr_matvec_dense_rhs_gradient_matches_dense_mlx(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    dense = csr.todense()

    def sparse_loss(x):
        y = csr @ x
        return mx.sum(y * y)

    def dense_loss(x):
        y = dense @ x
        return mx.sum(y * y)

    x = mx.array(np.array([3.0, 10.0, 7.0, -2.0], dtype=np.float32))

    np.testing.assert_allclose(
        to_numpy(mx.grad(sparse_loss)(x)),
        to_numpy(mx.grad(dense_loss)(x)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_matmul_dense_rhs_gradient_matches_dense_mlx(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    dense = csr.todense()

    def sparse_loss(rhs):
        y = csr @ rhs
        return mx.sum(y * y)

    def dense_loss(rhs):
        y = dense @ rhs
        return mx.sum(y * y)

    rhs = mx.array(
        np.array(
            [
                [3.0, 1.0],
                [10.0, -2.0],
                [7.0, 4.0],
                [-2.0, 6.0],
            ],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(
        to_numpy(mx.grad(sparse_loss)(rhs)),
        to_numpy(mx.grad(dense_loss)(rhs)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_batched_matvec_data_and_rhs_gradients_match_dense_mlx(mx):
    row_np = np.array([0, 0, 2, 2], dtype=np.int32)
    col_np = np.array([0, 2, 1, 3], dtype=np.int32)
    indptr_np = np.array([0, 2, 2, 4], dtype=np.int32)
    data_np = np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32)
    rhs_np = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 7.0
    dense_np = np.zeros((3, 4), dtype=np.float32)
    dense_np[row_np, col_np] = data_np

    indices = mx.array(col_np)
    indptr = mx.array(indptr_np)
    data = mx.array(data_np)
    rhs = mx.array(rhs_np)
    dense = mx.array(dense_np)

    def sparse_loss(values, vectors):
        csr = ms.csr_array(
            (values, indices, indptr),
            shape=(3, 4),
            sorted_indices=True,
            canonical=True,
        )
        y = ms.csr_batched_matvec(csr, vectors)
        return mx.sum(y * y)

    def dense_loss(matrix, vectors):
        y = vectors @ mx.transpose(matrix)
        return mx.sum(y * y)

    grad_data_sparse, grad_rhs_sparse = mx.grad(sparse_loss, argnums=(0, 1))(data, rhs)
    grad_dense, grad_rhs_dense = mx.grad(dense_loss, argnums=(0, 1))(dense, rhs)

    np.testing.assert_allclose(
        to_numpy(grad_data_sparse),
        to_numpy(grad_dense)[row_np, col_np],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(grad_rhs_sparse),
        to_numpy(grad_rhs_dense),
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_batched_matmul_data_and_rhs_gradients_match_dense_mlx(mx):
    row_np = np.array([0, 0, 2, 2], dtype=np.int32)
    col_np = np.array([0, 2, 1, 3], dtype=np.int32)
    indptr_np = np.array([0, 2, 2, 4], dtype=np.int32)
    data_np = np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32)
    rhs_np = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) / 9.0
    dense_np = np.zeros((3, 4), dtype=np.float32)
    dense_np[row_np, col_np] = data_np

    indices = mx.array(col_np)
    indptr = mx.array(indptr_np)
    data = mx.array(data_np)
    rhs = mx.array(rhs_np)
    dense = mx.array(dense_np)

    def sparse_loss(values, matrices):
        csr = ms.csr_array(
            (values, indices, indptr),
            shape=(3, 4),
            sorted_indices=True,
            canonical=True,
        )
        y = ms.csr_batched_matmul(csr, matrices)
        return mx.sum(y * y)

    def dense_loss(matrix, matrices):
        y = matrix[None, :, :] @ matrices
        return mx.sum(y * y)

    grad_data_sparse, grad_rhs_sparse = mx.grad(sparse_loss, argnums=(0, 1))(data, rhs)
    grad_dense, grad_rhs_dense = mx.grad(dense_loss, argnums=(0, 1))(dense, rhs)

    np.testing.assert_allclose(
        to_numpy(grad_data_sparse),
        to_numpy(grad_dense)[row_np, col_np],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(grad_rhs_sparse),
        to_numpy(grad_rhs_dense),
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_matvec_data_and_rhs_gradients_match_dense_mlx(mx):
    indices_np = np.array([0, 2, 1, 3], dtype=np.int32)
    indptr_np = np.array([0, 2, 2, 4], dtype=np.int32)
    row_np = np.array([0, 0, 2, 2], dtype=np.int32)
    col_np = indices_np
    data_np = np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32)
    dense_np = np.zeros((3, 4), dtype=np.float32)
    dense_np[row_np, col_np] = data_np

    indices = mx.array(indices_np)
    indptr = mx.array(indptr_np)
    x = mx.array(np.array([3.0, 10.0, 7.0, -2.0], dtype=np.float32))
    data = mx.array(data_np)
    dense = mx.array(dense_np)

    def sparse_loss(values, rhs):
        csr = ms.csr_array(
            (values, indices, indptr),
            shape=(3, 4),
            sorted_indices=True,
            canonical=True,
        )
        y = csr @ rhs
        return mx.sum(y * y)

    def dense_loss(matrix, rhs):
        y = matrix @ rhs
        return mx.sum(y * y)

    grad_data_sparse, grad_x_sparse = mx.grad(sparse_loss, argnums=(0, 1))(data, x)
    grad_dense, grad_x_dense = mx.grad(dense_loss, argnums=(0, 1))(dense, x)

    np.testing.assert_allclose(
        to_numpy(grad_data_sparse),
        to_numpy(grad_dense)[row_np, col_np],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(grad_x_sparse),
        to_numpy(grad_x_dense),
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_matmul_complex_data_and_rhs_gradients_match_dense_mlx(mx):
    row_np = np.array([0, 0, 1, 2], dtype=np.int32)
    col_np = np.array([0, 2, 1, 3], dtype=np.int32)
    indptr_np = np.array([0, 2, 3, 4], dtype=np.int32)
    data_np = np.array(
        [1.0 + 0.5j, -2.0 + 1.0j, 0.25 - 1.5j, 3.0 + 0.25j],
        dtype=np.complex64,
    )
    rhs_np = np.array(
        [
            [1.0 - 0.5j, 2.0 + 0.25j],
            [-1.0 + 1.5j, 0.5 - 0.75j],
            [2.0 + 0.0j, -3.0 + 0.5j],
            [0.25 - 2.0j, 1.0 + 1.0j],
        ],
        dtype=np.complex64,
    )
    dense_np = np.zeros((3, 4), dtype=np.complex64)
    dense_np[row_np, col_np] = data_np

    indices = mx.array(col_np)
    indptr = mx.array(indptr_np)
    data = mx.array(data_np)
    rhs = mx.array(rhs_np)
    dense = mx.array(dense_np)

    def sparse_loss(values, matrix_rhs):
        csr = ms.csr_array(
            (values, indices, indptr),
            shape=(3, 4),
            sorted_indices=True,
            canonical=True,
        )
        return _complex_energy(mx, csr @ matrix_rhs)

    def dense_loss(matrix, matrix_rhs):
        return _complex_energy(mx, matrix @ matrix_rhs)

    grad_data_sparse, grad_rhs_sparse = mx.grad(sparse_loss, argnums=(0, 1))(data, rhs)
    grad_dense, grad_rhs_dense = mx.grad(dense_loss, argnums=(0, 1))(dense, rhs)

    np.testing.assert_allclose(
        to_numpy(grad_data_sparse),
        to_numpy(grad_dense)[row_np, col_np],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(grad_rhs_sparse),
        to_numpy(grad_rhs_dense),
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_matvec_complex_jvp_and_vjp_match_dense_mlx(mx):
    data = mx.array(
        np.array([1.0 + 0.5j, -2.0 + 1.0j, 3.0 - 0.25j], dtype=np.complex64)
    )
    indices = mx.array(np.array([0, 2, 1], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 3], dtype=np.int32))
    csr = ms.csr_array(
        (data, indices, indptr),
        shape=(2, 3),
        sorted_indices=True,
        canonical=True,
    )
    dense = csr.todense()
    x = mx.array(np.array([1.0 - 1.0j, 2.0 + 0.5j, -1.0j], dtype=np.complex64))
    tangent = mx.array(
        np.array([0.25 + 0.5j, -1.0 + 0.75j, 2.0 - 0.5j], dtype=np.complex64)
    )
    cotangent = mx.array(np.array([2.0 - 1.0j, -0.5 + 3.0j], dtype=np.complex64))

    _, sparse_jvp = mx.jvp(lambda rhs: csr @ rhs, (x,), (tangent,))
    _, dense_jvp = mx.jvp(lambda rhs: dense @ rhs, (x,), (tangent,))
    _, sparse_vjp = mx.vjp(lambda rhs: csr @ rhs, (x,), (cotangent,))
    _, dense_vjp = mx.vjp(lambda rhs: dense @ rhs, (x,), (cotangent,))

    np.testing.assert_allclose(
        to_numpy(sparse_jvp[0]),
        to_numpy(dense_jvp[0]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(sparse_vjp[0]),
        to_numpy(dense_vjp[0]),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("dtype_name", "rtol", "atol"),
    [
        ("float16", 5e-3, 5e-3),
        ("bfloat16", 3e-2, 3e-2),
    ],
)
def test_csr_matvec_low_precision_vjp_matches_dense_mlx(mx, dtype_name, rtol, atol):
    indices_np = np.array([0, 3, 1, 3, 2, 4], dtype=np.int64)
    indptr_np = np.array([0, 2, 3, 5, 6], dtype=np.int64)
    data_np = np.array([2.0, -1.0, 0.5, 3.0, -2.5, 1.25], dtype=np.float32)
    x_np = np.array([1.0, -0.5, 2.0, 0.75, -1.25], dtype=np.float32)
    cotangent_np = np.array([1.0, -0.5, 2.0, 0.75], dtype=np.float32)

    dtype = getattr(mx, dtype_name)
    data = mx.array(data_np).astype(dtype)
    indices = mx.array(indices_np)
    indptr = mx.array(indptr_np)
    x = mx.array(x_np).astype(dtype)
    cotangent = mx.array(cotangent_np).astype(dtype)
    csr = ms.csr_array(
        (data, indices, indptr),
        shape=(4, 5),
        sorted_indices=True,
        canonical=True,
    )
    dense = csr.todense()

    _, sparse_vjp = mx.vjp(lambda rhs: csr @ rhs, (x,), (cotangent,))
    _, dense_vjp = mx.vjp(lambda rhs: dense @ rhs, (x,), (cotangent,))

    np.testing.assert_allclose(
        to_numpy(sparse_vjp[0]).astype(np.float32),
        to_numpy(dense_vjp[0]).astype(np.float32),
        rtol=rtol,
        atol=atol,
    )


def test_coo_matmul_data_rhs_gradients_and_jvp_vjp_match_dense_mlx(mx):
    row_np = np.array([0, 0, 1, 2, 2], dtype=np.int32)
    col_np = np.array([0, 2, 1, 0, 3], dtype=np.int32)
    data_np = np.array([2.0, -1.0, 0.5, 3.0, -2.0], dtype=np.float32)
    rhs_np = np.arange(4 * 3, dtype=np.float32).reshape(4, 3) / 7.0
    dense_np = np.zeros((3, 4), dtype=np.float32)
    dense_np[row_np, col_np] = data_np

    row = mx.array(row_np)
    col = mx.array(col_np)
    data = mx.array(data_np)
    rhs = mx.array(rhs_np)
    dense = mx.array(dense_np)

    def sparse_loss(values, matrix_rhs):
        coo = ms.coo_array((values, (row, col)), shape=(3, 4))
        y = coo @ matrix_rhs
        return mx.sum(y * y)

    def dense_loss(matrix, matrix_rhs):
        y = matrix @ matrix_rhs
        return mx.sum(y * y)

    grad_data_sparse, grad_rhs_sparse = mx.grad(sparse_loss, argnums=(0, 1))(data, rhs)
    grad_dense, grad_rhs_dense = mx.grad(dense_loss, argnums=(0, 1))(dense, rhs)

    np.testing.assert_allclose(
        to_numpy(grad_data_sparse),
        to_numpy(grad_dense)[row_np, col_np],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(grad_rhs_sparse), to_numpy(grad_rhs_dense), rtol=1e-5, atol=1e-5
    )

    tangent = mx.array(np.full_like(rhs_np, 0.25))
    cotangent = mx.array(np.ones((3, 3), dtype=np.float32))
    coo = ms.coo_array((data, (row, col)), shape=(3, 4))
    _, sparse_jvp = mx.jvp(lambda x: coo @ x, (rhs,), (tangent,))
    _, dense_jvp = mx.jvp(lambda x: dense @ x, (rhs,), (tangent,))
    _, sparse_vjp = mx.vjp(lambda x: coo @ x, (rhs,), (cotangent,))
    _, dense_vjp = mx.vjp(lambda x: dense @ x, (rhs,), (cotangent,))

    np.testing.assert_allclose(to_numpy(sparse_jvp[0]), to_numpy(dense_jvp[0]))
    np.testing.assert_allclose(to_numpy(sparse_vjp[0]), to_numpy(dense_vjp[0]))


def test_csc_batched_products_data_rhs_gradients_match_dense_mlx(mx):
    row_np = np.array([0, 0, 1, 2, 2], dtype=np.int32)
    col_np = np.array([0, 2, 1, 0, 3], dtype=np.int32)
    data_np = np.array([2.0, -1.0, 0.5, 3.0, -2.0], dtype=np.float32)
    dense_np = np.zeros((3, 4), dtype=np.float32)
    dense_np[row_np, col_np] = data_np
    rhs_np = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) / 9.0

    coo = ms.coo_array(
        (mx.array(data_np), (mx.array(row_np), mx.array(col_np))), shape=(3, 4)
    )
    csc_template = coo.tocsc(canonical=False)
    csc_rows_np = to_numpy(csc_template.indices)
    csc_cols_np = np.empty(csc_template.nnz, dtype=np.int64)
    indptr_np = to_numpy(csc_template.indptr)
    for col in range(csc_template.shape[1]):
        csc_cols_np[indptr_np[col] : indptr_np[col + 1]] = col

    data = csc_template.data
    indices = csc_template.indices
    indptr = csc_template.indptr
    rhs = mx.array(rhs_np)
    dense = mx.array(dense_np)

    def sparse_loss(values, matrices):
        csc = ms.csc_array((values, indices, indptr), shape=(3, 4))
        y = ms.csc_batched_matmul(csc, matrices)
        return mx.sum(y * y)

    def dense_loss(matrix, matrices):
        y = matrix[None, :, :] @ matrices
        return mx.sum(y * y)

    grad_data_sparse, grad_rhs_sparse = mx.grad(sparse_loss, argnums=(0, 1))(data, rhs)
    grad_dense, grad_rhs_dense = mx.grad(dense_loss, argnums=(0, 1))(dense, rhs)

    np.testing.assert_allclose(
        to_numpy(grad_data_sparse),
        to_numpy(grad_dense)[csc_rows_np, csc_cols_np],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(grad_rhs_sparse), to_numpy(grad_rhs_dense), rtol=1e-5, atol=1e-5
    )


def test_csc_matvec_transpose_vjp_and_jvp_match_dense_mlx(mx):
    row_np = np.array([0, 0, 1, 2, 2], dtype=np.int32)
    col_np = np.array([0, 2, 1, 0, 3], dtype=np.int32)
    data_np = np.array([2.0, -1.0, 0.5, 3.0, -2.0], dtype=np.float32)
    coo = ms.coo_array(
        (mx.array(data_np), (mx.array(row_np), mx.array(col_np))), shape=(3, 4)
    )
    csc = coo.tocsc(canonical=False)
    dense = csc.todense()
    x = mx.array(np.array([1.0, -0.5, 2.0], dtype=np.float32))
    tangent = mx.array(np.array([0.25, -1.0, 0.5], dtype=np.float32))
    cotangent = mx.array(np.array([1.0, -2.0, 0.5, 0.25], dtype=np.float32))

    _, sparse_jvp = mx.jvp(
        lambda rhs: ms.csc_matvec_transpose(csc, rhs), (x,), (tangent,)
    )
    _, dense_jvp = mx.jvp(lambda rhs: mx.transpose(dense) @ rhs, (x,), (tangent,))
    _, sparse_vjp = mx.vjp(
        lambda rhs: ms.csc_matvec_transpose(csc, rhs), (x,), (cotangent,)
    )
    _, dense_vjp = mx.vjp(lambda rhs: mx.transpose(dense) @ rhs, (x,), (cotangent,))

    np.testing.assert_allclose(to_numpy(sparse_jvp[0]), to_numpy(dense_jvp[0]))
    np.testing.assert_allclose(to_numpy(sparse_vjp[0]), to_numpy(dense_vjp[0]))


def _stored_sparse(mx, format_name, values):
    shape = (3, 3)
    index_dtype = mx.int64
    if format_name == "coo":
        row_np = np.array([0, 0, 1, 2, 2], dtype=np.int64)
        col_np = np.array([0, 2, 1, 0, 2], dtype=np.int64)
        array = ms.coo_array(
            (
                values,
                (
                    mx.array(row_np, dtype=index_dtype),
                    mx.array(col_np, dtype=index_dtype),
                ),
            ),
            shape=shape,
            canonical=True,
        )
        return array, row_np, col_np
    if format_name == "csr":
        row_np = np.array([0, 0, 1, 2, 2], dtype=np.int64)
        col_np = np.array([0, 2, 1, 0, 2], dtype=np.int64)
        array = ms.csr_array(
            (
                values,
                mx.array(col_np, dtype=index_dtype),
                mx.array(np.array([0, 2, 3, 5], dtype=np.int64)),
            ),
            shape=shape,
            sorted_indices=True,
            canonical=True,
        )
        return array, row_np, col_np
    if format_name == "csc":
        row_np = np.array([0, 2, 1, 0, 2], dtype=np.int64)
        col_np = np.array([0, 0, 1, 2, 2], dtype=np.int64)
        array = ms.csc_array(
            (
                values,
                mx.array(row_np, dtype=index_dtype),
                mx.array(np.array([0, 2, 3, 5], dtype=np.int64)),
            ),
            shape=shape,
            sorted_indices=True,
            canonical=True,
        )
        return array, row_np, col_np
    raise AssertionError(format_name)


def _scatter_dense(values, row, col, shape):
    out = np.zeros(shape, dtype=values.dtype)
    np.add.at(out, (row, col), values)
    return out


def _reduction_expected(values, row, col, shape, op_name):
    if op_name == "row_sums":
        out = np.zeros(shape[0], dtype=values.dtype)
        np.add.at(out, row, values)
        return out
    if op_name == "col_sums":
        out = np.zeros(shape[1], dtype=values.dtype)
        np.add.at(out, col, values)
        return out
    diag_size = min(shape)
    on_diag = (row == col) & (row < diag_size)
    if op_name == "diagonal":
        out = np.zeros(diag_size, dtype=values.dtype)
        np.add.at(out, row[on_diag], values[on_diag])
        return out
    if op_name == "trace":
        return np.asarray(values[on_diag].sum(), dtype=values.dtype)
    raise AssertionError(op_name)


def _reduction_vjp_expected(cotangent, row, col, shape, op_name):
    if op_name == "row_sums":
        return cotangent[row]
    if op_name == "col_sums":
        return cotangent[col]
    diag_size = min(shape)
    on_diag = (row == col) & (row < diag_size)
    if op_name == "diagonal":
        out = np.zeros(row.size, dtype=cotangent.dtype)
        out[on_diag] = cotangent[row[on_diag]]
        return out
    if op_name == "trace":
        return np.where(on_diag, cotangent, np.zeros((), dtype=cotangent.dtype))
    raise AssertionError(op_name)


def _reduction_output(array, op_name):
    if op_name == "row_sums":
        return array.row_sums()
    if op_name == "col_sums":
        return array.col_sums()
    if op_name == "diagonal":
        return array.diagonal()
    if op_name == "trace":
        return array.trace()
    raise AssertionError(op_name)


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_todense_sparse_value_grad_jvp_vjp_sample_stored_coordinates(
    mx, format_name, complex_values
):
    data_np = np.array([1.0, -2.0, 0.5, 3.0, -4.0], dtype=np.float32)
    tangent_np = np.array([0.25, -1.0, 2.0, -0.5, 1.5], dtype=np.float32)
    cotangent_np = np.arange(9, dtype=np.float32).reshape(3, 3) / 7.0
    if complex_values:
        data_np = data_np.astype(np.complex64) * (1.0 + 0.25j)
        tangent_np = tangent_np.astype(np.complex64) * (0.5 - 1.0j)
        cotangent_np = cotangent_np.astype(np.complex64) * (1.0 + 2.0j)

    data = mx.array(data_np)
    tangent = mx.array(tangent_np)
    cotangent = mx.array(cotangent_np)

    def materialize(values):
        array, _, _ = _stored_sparse(mx, format_name, values)
        return array.todense()

    array, row_np, col_np = _stored_sparse(mx, format_name, data)
    _, jvp = mx.jvp(materialize, (data,), (tangent,))
    _, vjp = mx.vjp(materialize, (data,), (cotangent,))

    np.testing.assert_allclose(
        to_numpy(jvp[0]),
        _scatter_dense(tangent_np, row_np, col_np, array.shape),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(vjp[0]),
        cotangent_np[row_np, col_np],
        rtol=1e-5,
        atol=1e-5,
    )

    if complex_values:
        grad = mx.grad(lambda values: _complex_energy(mx, materialize(values)))(data)
        np.testing.assert_allclose(to_numpy(grad), 2 * data_np, rtol=1e-5, atol=1e-5)
    else:
        weights = mx.array(cotangent_np)
        grad = mx.grad(lambda values: mx.sum(materialize(values) * weights))(data)
        np.testing.assert_allclose(
            to_numpy(grad),
            cotangent_np[row_np, col_np],
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
@pytest.mark.parametrize("op_name", ["row_sums", "col_sums", "diagonal", "trace"])
def test_sparse_reduction_value_grad_jvp_vjp_match_coordinate_rules(
    mx, format_name, op_name
):
    data_np = np.array([1.0, -2.0, 0.5, 3.0, -4.0], dtype=np.float32)
    tangent_np = np.array([0.25, -1.0, 2.0, -0.5, 1.5], dtype=np.float32)
    data = mx.array(data_np)
    tangent = mx.array(tangent_np)
    array, row_np, col_np = _stored_sparse(mx, format_name, data)
    expected_jvp = _reduction_expected(tangent_np, row_np, col_np, array.shape, op_name)
    if op_name == "trace":
        cotangent_np = np.array(2.5, dtype=np.float32)
    else:
        cotangent_np = np.arange(expected_jvp.size, dtype=np.float32) / 3.0 + 0.5
    cotangent = mx.array(cotangent_np)

    def reduced(values):
        sparse, _, _ = _stored_sparse(mx, format_name, values)
        return _reduction_output(sparse, op_name)

    def weighted(values):
        out = reduced(values)
        return out * cotangent if op_name == "trace" else mx.sum(out * cotangent)

    _, jvp = mx.jvp(reduced, (data,), (tangent,))
    _, vjp = mx.vjp(reduced, (data,), (cotangent,))
    grad = mx.grad(weighted)(data)
    expected_vjp = _reduction_vjp_expected(
        cotangent_np, row_np, col_np, array.shape, op_name
    )

    np.testing.assert_allclose(to_numpy(jvp[0]), expected_jvp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(vjp[0]), expected_vjp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(grad), expected_vjp, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
@pytest.mark.parametrize("op_name", ["row_sums", "col_sums", "trace"])
def test_sparse_reduction_complex_value_grad_jvp_vjp(mx, format_name, op_name):
    data_np = np.array(
        [1.0 + 0.5j, -2.0 + 1.0j, 0.5 - 0.25j, 3.0j, -4.0 + 0.75j],
        dtype=np.complex64,
    )
    tangent_np = np.array(
        [0.25 - 0.5j, -1.0 + 0.25j, 2.0j, -0.5 + 0.5j, 1.5 - 1.0j],
        dtype=np.complex64,
    )
    data = mx.array(data_np)
    tangent = mx.array(tangent_np)
    array, row_np, col_np = _stored_sparse(mx, format_name, data)
    expected_jvp = _reduction_expected(tangent_np, row_np, col_np, array.shape, op_name)
    cotangent_np = (
        np.array(1.25 - 0.5j, dtype=np.complex64)
        if op_name == "trace"
        else (np.arange(array.shape[0], dtype=np.float32) + 1).astype(np.complex64)
        * (1.0 - 0.5j)
    )
    cotangent = mx.array(cotangent_np)

    def reduced(values):
        sparse, _, _ = _stored_sparse(mx, format_name, values)
        return _reduction_output(sparse, op_name)

    _, jvp = mx.jvp(reduced, (data,), (tangent,))
    _, vjp = mx.vjp(reduced, (data,), (cotangent,))
    grad = mx.grad(lambda values: _complex_energy(mx, reduced(values)))(data)
    output_np = _reduction_expected(data_np, row_np, col_np, array.shape, op_name)
    expected_grad = _reduction_vjp_expected(
        2 * output_np, row_np, col_np, array.shape, op_name
    )

    np.testing.assert_allclose(to_numpy(jvp[0]), expected_jvp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        to_numpy(vjp[0]),
        _reduction_vjp_expected(cotangent_np, row_np, col_np, array.shape, op_name),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(to_numpy(grad), expected_grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("format_name", ["coo", "csc"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_tocsr_fixed_topology_conversion_data_grad_jvp_vjp(
    mx, format_name, complex_values
):
    if format_name == "coo":
        row_np = np.array([2, 0, 1, 0, 0], dtype=np.int64)
        col_np = np.array([1, 2, 0, 0, 2], dtype=np.int64)
        order = np.array([3, 1, 4, 2, 0], dtype=np.int64)

        def sparse_with_values(values):
            return ms.coo_array(
                (
                    values,
                    (
                        mx.array(row_np, dtype=mx.int64),
                        mx.array(col_np, dtype=mx.int64),
                    ),
                ),
                shape=(3, 3),
            )

    else:
        row_np = np.array([0, 2, 1], dtype=np.int64)
        order = np.array([0, 2, 1], dtype=np.int64)

        def sparse_with_values(values):
            return ms.csc_array(
                (
                    values,
                    mx.array(row_np, dtype=mx.int64),
                    mx.array(np.array([0, 1, 2, 3], dtype=np.int64)),
                ),
                shape=(3, 3),
                sorted_indices=True,
                canonical=True,
            )

    data_np = np.array([1.0, -2.0, 0.5, 3.0, -4.0], dtype=np.float32)[: order.size]
    tangent_np = np.array([0.25, -1.0, 2.0, -0.5, 1.5], dtype=np.float32)[: order.size]
    cotangent_np = np.arange(order.size, dtype=np.float32) / 3.0 + 0.75
    if complex_values:
        data_np = data_np.astype(np.complex64) * (1.0 + 0.5j)
        tangent_np = tangent_np.astype(np.complex64) * (0.25 - 0.75j)
        cotangent_np = cotangent_np.astype(np.complex64) * (1.0 - 0.25j)

    data = mx.array(data_np)
    tangent = mx.array(tangent_np)
    cotangent = mx.array(cotangent_np)

    def converted_data(values):
        return sparse_with_values(values).tocsr(canonical=False).data

    _, jvp = mx.jvp(converted_data, (data,), (tangent,))
    _, vjp = mx.vjp(converted_data, (data,), (cotangent,))
    grad = mx.grad(lambda values: _complex_energy(mx, converted_data(values)))(data)

    expected_vjp = np.empty_like(cotangent_np)
    expected_vjp[order] = cotangent_np

    np.testing.assert_allclose(
        to_numpy(jvp[0]), tangent_np[order], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(to_numpy(vjp[0]), expected_vjp, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(grad), 2 * data_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_diagonal_vjp_rectangular_matrix_ignores_out_of_range_rows(mx, format_name):
    data = mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    tangent = mx.array(np.array([0.5, -1.0, 2.0], dtype=np.float32))
    cotangent = mx.array(np.array([5.0, 7.0], dtype=np.float32))

    def array_with_values(values):
        if format_name == "coo":
            return ms.coo_array(
                (
                    values,
                    (
                        mx.array(np.array([0, 2, 3], dtype=np.int32)),
                        mx.array(np.array([0, 1, 1], dtype=np.int32)),
                    ),
                ),
                shape=(4, 2),
                canonical=True,
            )
        if format_name == "csr":
            return ms.csr_array(
                (
                    values,
                    mx.array(np.array([0, 1, 1], dtype=np.int32)),
                    mx.array(np.array([0, 1, 1, 2, 3], dtype=np.int32)),
                ),
                shape=(4, 2),
                sorted_indices=True,
                canonical=True,
            )
        return ms.csc_array(
            (
                values,
                mx.array(np.array([0, 2, 3], dtype=np.int32)),
                mx.array(np.array([0, 1, 3], dtype=np.int32)),
            ),
            shape=(4, 2),
            sorted_indices=True,
            canonical=True,
        )

    def diagonal(values):
        return array_with_values(values).diagonal()

    _, jvp = mx.jvp(diagonal, (data,), (tangent,))
    _, vjp = mx.vjp(diagonal, (data,), (cotangent,))

    np.testing.assert_allclose(to_numpy(jvp[0]), np.array([0.5, 0.0]))
    np.testing.assert_allclose(to_numpy(vjp[0]), np.array([5.0, 0.0, 0.0]))


def test_diags_fixed_topology_value_grad_jvp_vjp(mx):
    main_np = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    upper_np = np.array([4.0, -5.0], dtype=np.float32)
    main_tangent_np = np.array([0.5, 1.5, -2.0], dtype=np.float32)
    upper_tangent_np = np.array([-1.0, 2.0], dtype=np.float32)
    cotangent_np = np.arange(9, dtype=np.float32).reshape(3, 3) / 5.0
    main = mx.array(main_np)
    upper = mx.array(upper_np)
    main_tangent = mx.array(main_tangent_np)
    upper_tangent = mx.array(upper_tangent_np)
    cotangent = mx.array(cotangent_np)

    def dense_from_diags(main_values, upper_values):
        return ms.diags(
            [main_values, upper_values],
            offsets=[0, 1],
            shape=(3, 3),
        ).todense()

    _, jvp = mx.jvp(
        dense_from_diags,
        (main, upper),
        (main_tangent, upper_tangent),
    )
    _, vjp = mx.vjp(dense_from_diags, (main, upper), (cotangent,))
    grad = mx.grad(
        lambda main_values, upper_values: mx.sum(
            dense_from_diags(main_values, upper_values) * cotangent
        ),
        argnums=(0, 1),
    )(main, upper)

    expected_jvp = np.zeros((3, 3), dtype=np.float32)
    expected_jvp[np.arange(3), np.arange(3)] = main_tangent_np
    expected_jvp[np.arange(2), np.arange(1, 3)] = upper_tangent_np
    expected_main = np.diag(cotangent_np)
    expected_upper = cotangent_np[np.arange(2), np.arange(1, 3)]

    np.testing.assert_allclose(to_numpy(jvp[0]), expected_jvp)
    np.testing.assert_allclose(to_numpy(vjp[0]), expected_main)
    np.testing.assert_allclose(to_numpy(vjp[1]), expected_upper)
    np.testing.assert_allclose(to_numpy(grad[0]), expected_main)
    np.testing.assert_allclose(to_numpy(grad[1]), expected_upper)


@pytest.mark.parametrize(
    "constructor_name", ["block_array", "block_diag", "vstack", "hstack"]
)
def test_block_stack_fixed_topology_value_grad_jvp_vjp(mx, constructor_name):
    row = mx.array(np.array([0, 1], dtype=np.int32))
    col_a = mx.array(np.array([0, 1], dtype=np.int32))
    col_b = mx.array(np.array([1, 0], dtype=np.int32))
    left_np = np.array([1.0, -2.0], dtype=np.float32)
    right_np = np.array([3.0, -4.0], dtype=np.float32)
    left_tangent_np = np.array([0.25, 0.5], dtype=np.float32)
    right_tangent_np = np.array([-1.5, 2.0], dtype=np.float32)
    cotangent_np = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
    left = mx.array(left_np)
    right = mx.array(right_np)
    left_tangent = mx.array(left_tangent_np)
    right_tangent = mx.array(right_tangent_np)
    cotangent = mx.array(cotangent_np)

    def values(left_values, right_values):
        A = ms.coo_array((left_values, (row, col_a)), shape=(2, 2), canonical=True)
        B = ms.coo_array((right_values, (row, col_b)), shape=(2, 2), canonical=True)
        if constructor_name == "block_array":
            return ms.block_array([[A, None], [None, B]], format="coo").data
        if constructor_name == "block_diag":
            return ms.block_diag([A, B], format="coo").data
        if constructor_name == "vstack":
            return ms.vstack([A, B], format="coo").data
        if constructor_name == "hstack":
            return ms.hstack([A, B], format="coo").data
        raise AssertionError(constructor_name)

    _, jvp = mx.jvp(values, (left, right), (left_tangent, right_tangent))
    _, vjp = mx.vjp(values, (left, right), (cotangent,))
    grad = mx.grad(
        lambda left_values, right_values: mx.sum(
            values(left_values, right_values) * cotangent
        ),
        argnums=(0, 1),
    )(left, right)

    np.testing.assert_allclose(
        to_numpy(jvp[0]), np.concatenate([left_tangent_np, right_tangent_np])
    )
    np.testing.assert_allclose(to_numpy(vjp[0]), cotangent_np[:2])
    np.testing.assert_allclose(to_numpy(vjp[1]), cotangent_np[2:])
    np.testing.assert_allclose(to_numpy(grad[0]), cotangent_np[:2])
    np.testing.assert_allclose(to_numpy(grad[1]), cotangent_np[2:])


def test_kron_fixed_topology_value_grad_jvp_vjp(mx):
    row_a = mx.array(np.array([0, 1], dtype=np.int32))
    col_a = mx.array(np.array([0, 1], dtype=np.int32))
    row_b = mx.array(np.array([0, 1], dtype=np.int32))
    col_b = mx.array(np.array([1, 0], dtype=np.int32))
    lhs_np = np.array([2.0, -3.0], dtype=np.float32)
    rhs_np = np.array([5.0, 7.0], dtype=np.float32)
    lhs_tangent_np = np.array([0.25, 1.5], dtype=np.float32)
    rhs_tangent_np = np.array([-2.0, 3.0], dtype=np.float32)
    cotangent_np = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float32)
    lhs = mx.array(lhs_np)
    rhs = mx.array(rhs_np)
    lhs_tangent = mx.array(lhs_tangent_np)
    rhs_tangent = mx.array(rhs_tangent_np)
    cotangent = mx.array(cotangent_np)

    def values(left_values, right_values):
        A = ms.coo_array((left_values, (row_a, col_a)), shape=(2, 2), canonical=True)
        B = ms.coo_array((right_values, (row_b, col_b)), shape=(2, 2), canonical=True)
        return ms.kron(A, B, format="coo").data

    _, jvp = mx.jvp(values, (lhs, rhs), (lhs_tangent, rhs_tangent))
    _, vjp = mx.vjp(values, (lhs, rhs), (cotangent,))
    grad = mx.grad(
        lambda left_values, right_values: mx.sum(
            values(left_values, right_values) * cotangent
        ),
        argnums=(0, 1),
    )(lhs, rhs)

    cotangent_matrix = cotangent_np.reshape(lhs_np.size, rhs_np.size)
    expected_jvp = (
        lhs_tangent_np[:, None] * rhs_np[None, :]
        + lhs_np[:, None] * rhs_tangent_np[None, :]
    ).reshape(-1)
    expected_lhs = np.sum(cotangent_matrix * rhs_np[None, :], axis=1)
    expected_rhs = np.sum(cotangent_matrix * lhs_np[:, None], axis=0)

    np.testing.assert_allclose(to_numpy(jvp[0]), expected_jvp)
    np.testing.assert_allclose(to_numpy(vjp[0]), expected_lhs)
    np.testing.assert_allclose(to_numpy(vjp[1]), expected_rhs)
    np.testing.assert_allclose(to_numpy(grad[0]), expected_lhs)
    np.testing.assert_allclose(to_numpy(grad[1]), expected_rhs)


def test_dynamic_topology_fromdense_autodiff_errors(mx):
    dense = mx.array(np.eye(3, dtype=np.float32))
    with pytest.raises(
        RuntimeError, match="fromdense has value-dependent sparse topology"
    ):
        mx.grad(lambda values: mx.sum(ms.fromdense(values).data))(dense)
    with pytest.raises(
        RuntimeError, match="fromdense has value-dependent sparse topology"
    ):
        mx.jvp(
            lambda values: ms.fromdense(values).data,
            (dense,),
            (mx.ones_like(dense),),
        )


@pytest.mark.parametrize("format_name", ["csr", "csc"])
def test_duplicate_canonicalization_autodiff_errors(mx, format_name):
    data = mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    indices = mx.array(np.array([0, 0, 1], dtype=np.int32))
    indptr = mx.array(np.array([0, 3], dtype=np.int32))

    def canonical_data(values):
        if format_name == "csr":
            array = ms.csr_array(
                (values, indices, indptr),
                shape=(1, 2),
                sorted_indices=True,
                canonical=False,
            )
        else:
            array = ms.csc_array(
                (values, indices, indptr),
                shape=(2, 1),
                sorted_indices=True,
                canonical=False,
            )
        return array.sum_duplicates().data

    with pytest.raises(RuntimeError, match="canonicalization sums duplicates"):
        mx.grad(lambda values: mx.sum(canonical_data(values)))(data)


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_triangular_structural_filter_autodiff_errors(mx, format_name):
    data = mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def triangular_data(values):
        if format_name == "coo":
            array = ms.coo_array(
                (
                    values,
                    (
                        mx.array(np.array([0, 1, 1], dtype=np.int32)),
                        mx.array(np.array([0, 0, 1], dtype=np.int32)),
                    ),
                ),
                shape=(2, 2),
                canonical=True,
            )
        elif format_name == "csr":
            array = ms.csr_array(
                (
                    values,
                    mx.array(np.array([0, 0, 1], dtype=np.int32)),
                    mx.array(np.array([0, 1, 3], dtype=np.int32)),
                ),
                shape=(2, 2),
                sorted_indices=True,
                canonical=True,
            )
        else:
            array = ms.csc_array(
                (
                    values,
                    mx.array(np.array([0, 1, 1], dtype=np.int32)),
                    mx.array(np.array([0, 2, 3], dtype=np.int32)),
                ),
                shape=(2, 2),
                sorted_indices=True,
                canonical=True,
            )
        return ms.tril(array, format=format_name).data

    with pytest.raises(
        RuntimeError, match="structural filters that compact sparse topology"
    ):
        mx.grad(lambda values: mx.sum(triangular_data(values)))(data)
