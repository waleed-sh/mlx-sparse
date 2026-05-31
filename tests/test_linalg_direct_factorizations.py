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
from mlx_sparse._ext_loader import extension_available

pytestmark = [pytest.mark.native, pytest.mark.cpu_only]


def _require_native():
    if not extension_available():
        pytest.skip("native extension unavailable")


def _index_mx_dtype(mx, index_dtype):
    return mx.int64 if index_dtype == np.int64 else mx.int32


def _csr_from_dense(mx, dense, *, index_dtype=np.int32, storage="full"):
    rows = []
    cols = []
    data = []
    n_rows, n_cols = dense.shape
    for row in range(n_rows):
        for col in range(n_cols):
            if storage == "lower" and col > row:
                continue
            if storage == "upper" and col < row:
                continue
            value = dense[row, col]
            if value != 0.0:
                rows.append(row)
                cols.append(col)
                data.append(value)

    indptr = np.zeros(n_rows + 1, dtype=index_dtype)
    for row in rows:
        indptr[row + 1] += 1
    np.cumsum(indptr, out=indptr)
    return ms.csr_array(
        (
            mx.array(np.array(data, dtype=np.float32)),
            mx.array(np.array(cols, dtype=index_dtype)),
            mx.array(indptr),
        ),
        shape=(n_rows, n_cols),
        sorted_indices=True,
        canonical=True,
        validate="full",
    )


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_sparse_cholesky_banded_spd_factor_and_solve(mx, to_numpy, index_dtype):
    _require_native()
    n = 10
    dense = np.diag(np.full(n, 4.0, dtype=np.float32))
    dense += np.diag(np.full(n - 1, -1.0, dtype=np.float32), k=-1)
    dense += np.diag(np.full(n - 1, -1.0, dtype=np.float32), k=1)
    csr = _csr_from_dense(mx, dense, index_dtype=index_dtype)
    rhs_np = np.linspace(-1.0, 1.0, n, dtype=np.float32)

    factor = linalg.sparse_cholesky(csr)
    dense_l = to_numpy(factor.L.todense())
    got = factor.solve(mx.array(rhs_np))

    assert factor.L.indices.dtype == _index_mx_dtype(mx, index_dtype)
    np.testing.assert_allclose(dense_l @ dense_l.T, dense, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(
        to_numpy(got), np.linalg.solve(dense, rhs_np), rtol=5e-4, atol=5e-4
    )


@pytest.mark.parametrize("storage", ["lower", "upper"])
def test_sparse_cholesky_accepts_single_triangle_symmetric_storage(
    mx, to_numpy, storage
):
    _require_native()
    dense = np.array(
        [
            [5.0, -1.0, 0.25, 0.0],
            [-1.0, 4.5, -0.5, 0.25],
            [0.25, -0.5, 3.5, -0.75],
            [0.0, 0.25, -0.75, 3.0],
        ],
        dtype=np.float32,
    )
    csr = _csr_from_dense(mx, dense, storage=storage)
    rhs_np = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)

    factor = linalg.sparse_cholesky(csr)
    dense_l = to_numpy(factor.L.todense())
    got = factor.solve(mx.array(rhs_np))

    np.testing.assert_allclose(dense_l @ dense_l.T, dense, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(
        to_numpy(got), np.linalg.solve(dense, rhs_np), rtol=5e-4, atol=5e-4
    )


def test_sparse_lu_preserves_partial_pivoting_semantics(mx, to_numpy):
    _require_native()
    dense = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 4.0],
        ],
        dtype=np.float32,
    )
    csr = _csr_from_dense(mx, dense)
    rhs_np = np.array([2.0, -1.0, 0.5], dtype=np.float32)

    factor = linalg.sparse_lu(csr)
    perm = to_numpy(factor.perm)
    dense_l = to_numpy(factor.L.todense())
    dense_u = to_numpy(factor.U.todense())
    got = factor.solve(mx.array(rhs_np))

    assert not np.array_equal(perm, np.arange(dense.shape[0], dtype=perm.dtype))
    np.testing.assert_allclose(dense_l @ dense_u, dense[perm], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        to_numpy(got), np.linalg.solve(dense, rhs_np), rtol=1e-4, atol=1e-4
    )


def test_direct_factorizations_reject_near_zero_pivots(mx):
    _require_native()
    dense = np.diag(np.array([1.0e-9, 1.0], dtype=np.float32))
    csr = _csr_from_dense(mx, dense)

    with pytest.raises(RuntimeError, match="positive-definite|pivot"):
        linalg.sparse_cholesky(csr)
    with pytest.raises(RuntimeError, match="singular pivot"):
        linalg.sparse_lu(csr)


def test_direct_factorizations_canonicalize_duplicate_unsorted_inputs(mx, to_numpy):
    _require_native()
    data = np.array(
        [0.75, 4.0, 0.25, 0.5, 1.0, 3.0, 2.0, 0.5],
        dtype=np.float32,
    )
    indices = np.array([1, 0, 1, 2, 0, 1, 2, 1], dtype=np.int32)
    indptr = np.array([0, 3, 6, 8], dtype=np.int32)
    csr = ms.csr_array(
        (
            mx.array(data),
            mx.array(indices),
            mx.array(indptr),
        ),
        shape=(3, 3),
        sorted_indices=False,
        canonical=False,
    )
    dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 0.5],
            [0.0, 0.5, 2.0],
        ],
        dtype=np.float32,
    )
    rhs_np = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    expected = np.linalg.solve(dense, rhs_np)

    chol = linalg.sparse_cholesky(csr)
    lu = linalg.sparse_lu(csr)

    np.testing.assert_allclose(
        to_numpy(chol.solve(mx.array(rhs_np))), expected, rtol=5e-4, atol=5e-4
    )
    np.testing.assert_allclose(
        to_numpy(lu.solve(mx.array(rhs_np))), expected, rtol=5e-4, atol=5e-4
    )
