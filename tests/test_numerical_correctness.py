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
from mlx_sparse._host import to_numpy

import mlx_sparse as ms


def _csr_from_scipy(mx, scipy_matrix, *, dtype):
    csr = scipy_matrix.tocsr()
    return ms.csr_array(
        (
            mx.array(csr.data.astype(dtype)),
            mx.array(csr.indices.astype(np.int32)),
            mx.array(csr.indptr.astype(np.int32)),
        ),
        shape=csr.shape,
        sorted_indices=True,
        canonical=True,
    )


def test_random_csr_matvec_and_matmul_match_mlx_numpy_and_scipy(mx, scipy_sparse):
    rng = np.random.default_rng(20260521)
    scipy_csr = scipy_sparse.random(
        48,
        64,
        density=0.08,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    scipy_csr.sum_duplicates()
    scipy_csr.sort_indices()
    csr = _csr_from_scipy(mx, scipy_csr, dtype=np.float32)
    dense_mx = csr.todense()
    dense_np = scipy_csr.toarray().astype(np.float32)

    x_np = rng.standard_normal(64).astype(np.float32)
    rhs_np = rng.standard_normal((64, 5)).astype(np.float32)
    batched_np = rng.standard_normal((3, 64, 4)).astype(np.float32)

    sparse_vec = to_numpy(csr @ mx.array(x_np))
    mlx_vec = to_numpy(dense_mx @ mx.array(x_np))
    numpy_vec = dense_np @ x_np
    scipy_vec = scipy_csr @ x_np
    np.testing.assert_allclose(sparse_vec, mlx_vec, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sparse_vec, numpy_vec, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sparse_vec, scipy_vec, rtol=1e-5, atol=1e-5)

    sparse_mat = to_numpy(csr @ mx.array(rhs_np))
    mlx_mat = to_numpy(dense_mx @ mx.array(rhs_np))
    numpy_mat = dense_np @ rhs_np
    scipy_mat = scipy_csr @ rhs_np
    np.testing.assert_allclose(sparse_mat, mlx_mat, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sparse_mat, numpy_mat, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sparse_mat, scipy_mat, rtol=1e-5, atol=1e-5)

    sparse_batched = to_numpy(csr @ mx.array(batched_np))
    numpy_batched = dense_np @ batched_np
    np.testing.assert_allclose(sparse_batched, numpy_batched, rtol=1e-5, atol=1e-5)


def test_complex_csr_matches_dense_mlx_numpy_and_scipy(mx, scipy_sparse):
    rng = np.random.default_rng(314159)
    row = np.array([0, 0, 1, 2, 2, 2, 3], dtype=np.int32)
    col = np.array([1, 4, 2, 0, 3, 5, 4], dtype=np.int32)
    data = (rng.standard_normal(row.size) + 1j * rng.standard_normal(row.size)).astype(
        np.complex64
    )
    scipy_csr = scipy_sparse.coo_matrix((data, (row, col)), shape=(4, 6)).tocsr()
    csr = _csr_from_scipy(mx, scipy_csr, dtype=np.complex64)
    dense_np = scipy_csr.toarray().astype(np.complex64)
    dense_mx = csr.todense()

    x_np = (rng.standard_normal(6) + 1j * rng.standard_normal(6)).astype(np.complex64)
    rhs_np = (rng.standard_normal((6, 3)) + 1j * rng.standard_normal((6, 3))).astype(
        np.complex64
    )

    sparse_vec = to_numpy(csr @ mx.array(x_np))
    np.testing.assert_allclose(
        sparse_vec,
        to_numpy(dense_mx @ mx.array(x_np)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(sparse_vec, dense_np @ x_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sparse_vec, scipy_csr @ x_np, rtol=1e-5, atol=1e-5)

    sparse_mat = to_numpy(csr @ mx.array(rhs_np))
    np.testing.assert_allclose(
        sparse_mat,
        to_numpy(dense_mx @ mx.array(rhs_np)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(sparse_mat, dense_np @ rhs_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sparse_mat, scipy_csr @ rhs_np, rtol=1e-5, atol=1e-5)


def test_duplicate_coo_semantics_match_numpy_and_scipy(mx, scipy_sparse):
    row = np.array([0, 0, 0, 1, 2, 2], dtype=np.int32)
    col = np.array([1, 1, 2, 0, 2, 2], dtype=np.int32)
    data = np.array([1.0, 3.0, -2.0, 5.0, 7.0, -4.0], dtype=np.float32)
    scipy_coo = scipy_sparse.coo_matrix((data, (row, col)), shape=(3, 4))
    scipy_csr = scipy_coo.tocsr()
    scipy_csr.sum_duplicates()

    coo = ms.coo_array(
        (mx.array(data), (mx.array(row), mx.array(col))),
        shape=scipy_coo.shape,
    )
    csr = coo.tocsr(canonical=True)

    expected_dense = np.zeros(scipy_coo.shape, dtype=np.float32)
    np.add.at(expected_dense, (row, col), data)
    np.testing.assert_allclose(to_numpy(csr.todense()), expected_dense)
    np.testing.assert_allclose(to_numpy(csr.todense()), scipy_csr.toarray())

    x_np = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    np.testing.assert_allclose(to_numpy(csr @ mx.array(x_np)), expected_dense @ x_np)
    np.testing.assert_allclose(to_numpy(csr @ mx.array(x_np)), scipy_csr @ x_np)


def test_sparse_sparse_matmul_matches_scipy_and_numpy(mx, scipy_sparse):
    lhs = scipy_sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0, 0.0],
                [0.0, -3.0, 0.0, 4.0],
                [5.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    rhs = scipy_sparse.csr_matrix(
        np.array(
            [
                [0.0, 6.0],
                [7.0, 0.0],
                [0.0, -1.0],
                [2.0, 3.0],
            ],
            dtype=np.float32,
        )
    )

    sparse_out = ms.from_scipy(lhs) @ ms.from_scipy(rhs)
    scipy_out = lhs @ rhs
    numpy_out = lhs.toarray() @ rhs.toarray()

    assert isinstance(sparse_out, ms.CSRArray)
    assert sparse_out.has_canonical_format
    np.testing.assert_allclose(to_numpy(sparse_out.todense()), scipy_out.toarray())
    np.testing.assert_allclose(to_numpy(sparse_out.todense()), numpy_out)
