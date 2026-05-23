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
