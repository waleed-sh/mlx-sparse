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
import mlx_sparse._construct as construct
from mlx_sparse._host import to_numpy


def test_sparse_suite_mixed_constructors_match_dense_and_scipy(mx, scipy_sparse):
    dense_np = np.array(
        [
            [1.0, 0.0, -2.0, 0.0],
            [0.0, 0.25, 0.0, 3.0],
            [4.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    csr = ms.fromdense(mx.array(dense_np), threshold=0.2, index_dtype=mx.int64)
    scipy_csr = scipy_sparse.csr_matrix(dense_np * (np.abs(dense_np) > 0.2))

    assert csr.shape == scipy_csr.shape
    assert csr.index_dtype == mx.int64
    assert csr.has_canonical_format
    np.testing.assert_allclose(to_numpy(csr.todense()), scipy_csr.toarray())

    x_np = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    rhs_np = np.arange(8, dtype=np.float32).reshape(4, 2) / 3.0
    np.testing.assert_allclose(to_numpy(csr @ mx.array(x_np)), scipy_csr @ x_np)
    np.testing.assert_allclose(to_numpy(csr @ mx.array(rhs_np)), scipy_csr @ rhs_np)

    row = mx.array(np.array([0, 0, 2, 1, 1], dtype=np.int32))
    col = mx.array(np.array([0, 2, 0, 1, 3], dtype=np.int32))
    data = mx.array(np.array([1.0, -2.0, 4.0, 0.25, 3.0], dtype=np.float32))
    coo = ms.coo_array((data, (row, col)), shape=dense_np.shape)
    coo_csr = ms.asarray(coo, dtype=mx.float16)

    assert coo_csr.dtype == mx.float16
    np.testing.assert_allclose(
        to_numpy(coo_csr.todense()).astype(np.float32),
        scipy_csr.toarray(),
        rtol=5e-3,
        atol=5e-3,
    )


def test_sparse_suite_diagonal_constructors_cover_scalar_array_and_empty(mx):
    scalar_diag = ms.diags(5.0, offsets=1, shape=(2, 3))
    np.testing.assert_allclose(
        to_numpy(scalar_diag.todense()),
        np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
    )

    zero_dim_diag = ms.diags(mx.array(7.0, dtype=mx.float32), offsets=-1, shape=(3, 3))
    np.testing.assert_allclose(
        to_numpy(zero_dim_diag.todense()),
        np.array(
            [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )

    stacked = ms.diags(
        mx.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        offsets=[0, 1],
        shape=(3, 3),
    )
    np.testing.assert_allclose(
        to_numpy(stacked.todense()),
        np.array(
            [[1.0, 3.0, 0.0], [0.0, 2.0, 4.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )

    empty = ms.diags([], offsets=[], shape=(2, 2))
    assert empty.nnz == 0
    np.testing.assert_allclose(to_numpy(empty.todense()), np.zeros((2, 2)))


def test_sparse_suite_constructor_validation_edges(mx, scipy_sparse):
    with pytest.raises(TypeError, match="dtype must be one"):
        ms.eye(2, dtype=mx.int32)

    with pytest.raises(TypeError, match="index_dtype"):
        ms.eye(2, index_dtype=mx.float32)

    with pytest.raises(ValueError, match="repeated offsets"):
        ms.diags([[1.0], [2.0]], offsets=[0, 0], shape=(2, 2))

    with pytest.raises(ValueError, match="can hold at most"):
        ms.diags([1.0, 2.0, 3.0], offsets=2, shape=(2, 2))

    assert (
        construct._infer_value_dtype_from_numpy(np.array([1.0], dtype=np.float16))
        == mx.float16
    )

    scipy_float64 = scipy_sparse.eye(2, format="coo", dtype=np.float64)
    bfloat = ms.from_scipy(scipy_float64, format="coo", dtype=mx.bfloat16)
    assert bfloat.dtype == mx.bfloat16


def test_sparse_suite_public_operation_errors(mx):
    csr = ms.eye(2)

    with pytest.raises(TypeError, match="todense expects"):
        ms.todense(object())

    with pytest.raises(TypeError, match="csr_matvec expects CSRArray"):
        ms.csr_matvec(object(), mx.array([1.0, 2.0], dtype=mx.float32))

    with pytest.raises(TypeError, match="csr_matmul expects CSRArray"):
        ms.csr_matmul(object(), mx.eye(2, dtype=mx.float32))

    with pytest.raises(TypeError, match="csr_matmat expects CSRArray lhs"):
        ms.csr_matmat(object(), csr)

    with pytest.raises(TypeError, match="csr_matmat expects CSRArray rhs"):
        ms.csr_matmat(csr, object())

    with pytest.raises(ValueError, match="rank-2 or higher"):
        ms.csr_batched_matvec(csr, mx.array([1.0, 2.0], dtype=mx.float32))

    with pytest.raises(ValueError, match="vector dimension"):
        ms.csr_batched_matvec(csr, mx.ones((2, 3), dtype=mx.float32))

    with pytest.raises(TypeError, match="same dtype"):
        ms.csr_batched_matvec(csr, mx.ones((2, 2), dtype=mx.float16))

    with pytest.raises(ValueError, match="rank-3 or higher"):
        ms.csr_batched_matmul(csr, mx.eye(2, dtype=mx.float32))

    with pytest.raises(ValueError, match="sparse dimension"):
        ms.csr_batched_matmul(csr, mx.ones((1, 3, 2), dtype=mx.float32))

    with pytest.raises(TypeError, match="same dtype"):
        ms.csr_batched_matmul(csr, mx.ones((1, 2, 2), dtype=mx.float16))

    with pytest.raises(ValueError, match="rank-2 or higher"):
        ms.csr_matmul(csr, mx.array([1.0, 2.0], dtype=mx.float32))

    with pytest.raises(ValueError, match="sparse dimension"):
        ms.csr_matmul(csr, mx.ones((1, 3, 2), dtype=mx.float32))

    with pytest.raises(TypeError, match="same dtype"):
        ms.csr_matmul(csr, mx.ones((1, 2, 2), dtype=mx.float16))
