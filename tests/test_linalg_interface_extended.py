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

"""Tests for mlx_sparse.linalg._interface (LinearOperator) and _utils.ensure_array."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse.linalg._interface import LinearOperator, aslinearoperator
from mlx_sparse.linalg._utils import ensure_array


class TestEnsureArray:
    def test_passthrough_mx_array_no_dtype(self):
        x = mx.array([1.0, 2.0], dtype=mx.float32)
        result = ensure_array(x)
        assert result is x

    def test_passthrough_mx_array_same_dtype(self):
        x = mx.array([1.0, 2.0], dtype=mx.float32)
        result = ensure_array(x, dtype=mx.float32)
        assert result is x

    def test_cast_mx_array_different_dtype(self):
        x = mx.array([1.0, 2.0], dtype=mx.float32)
        result = ensure_array(x, dtype=mx.float16)
        assert result.dtype == mx.float16

    def test_convert_list_no_dtype(self):
        result = ensure_array([1.0, 2.0, 3.0])
        assert isinstance(result, mx.array)
        assert result.shape == (3,)

    def test_convert_list_with_dtype(self):
        result = ensure_array([1.0, 2.0], dtype=mx.float16)
        assert isinstance(result, mx.array)
        assert result.dtype == mx.float16

    def test_convert_numpy_array(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = ensure_array(arr)
        assert isinstance(result, mx.array)


def _make_identity_op(n: int) -> LinearOperator:
    """Simple n×n identity LinearOperator."""
    return LinearOperator(
        (n, n),
        matvec=lambda x: x,
        matmat=lambda X: X,
        rmatvec=lambda x: x,
        dtype=mx.float32,
    )


def _make_2x2_csr():
    return ms.csr_array(
        (
            mx.array([2.0, 1.0, 1.0, 3.0], dtype=mx.float32),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            mx.array([0, 2, 4], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )


class TestLinearOperatorConstruction:
    def test_matvec_positional(self):
        op = LinearOperator((3, 3), lambda x: x)
        assert op.shape == (3, 3)

    def test_matvec_fn_keyword(self):
        op = LinearOperator((3, 3), matvec_fn=lambda x: x)
        assert op.shape == (3, 3)

    def test_matvec_fn_takes_priority(self):
        called = []
        fn_a = lambda x: (called.append("a"), x)[1]
        fn_b = lambda x: (called.append("b"), x)[1]
        op = LinearOperator((2, 2), matvec=fn_a, matvec_fn=fn_b)
        v = mx.array([1.0, 2.0], dtype=mx.float32)
        op.matvec(v)
        assert called == ["b"]

    def test_no_matvec_raises(self):
        with pytest.raises(TypeError, match="requires a matvec callable"):
            LinearOperator((3, 3))

    def test_ndim_always_2(self):
        op = _make_identity_op(4)
        assert op.ndim == 2

    def test_dtype_stored(self):
        op = _make_identity_op(4)
        assert op.dtype == mx.float32


class TestLinearOperatorMatvec:
    def test_valid_matvec(self):
        op = _make_identity_op(3)
        x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        result = op.matvec(x)
        mx.eval(result)
        np.testing.assert_allclose(np.array(result), [1.0, 2.0, 3.0])

    def test_matmul_rank1_dispatches_to_matvec(self):
        op = _make_identity_op(3)
        x = mx.array([4.0, 5.0, 6.0], dtype=mx.float32)
        result = op @ x
        mx.eval(result)
        np.testing.assert_allclose(np.array(result), [4.0, 5.0, 6.0])

    def test_matvec_rank2_raises(self):
        op = _make_identity_op(3)
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=mx.float32)
        with pytest.raises(ValueError, match="rank-1"):
            op.matvec(X)

    def test_matvec_wrong_length_raises(self):
        op = _make_identity_op(3)
        x = mx.array([1.0, 2.0], dtype=mx.float32)  # length 2, expected 3
        with pytest.raises(ValueError, match="length"):
            op.matvec(x)


class TestLinearOperatorMatmat:
    def test_valid_matmat(self):
        op = _make_identity_op(3)
        X = mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=mx.float32)
        result = op.matmat(X)
        mx.eval(result)
        np.testing.assert_allclose(np.array(result), np.array(X))

    def test_matmul_rank2_dispatches_to_matmat(self):
        op = _make_identity_op(2)
        X = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        result = op @ X
        mx.eval(result)
        np.testing.assert_allclose(np.array(result), np.array(X))

    def test_matmat_rank1_raises(self):
        op = _make_identity_op(3)
        x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        with pytest.raises(ValueError, match="rank-2"):
            op.matmat(x)

    def test_matmat_wrong_leading_dim_raises(self):
        op = _make_identity_op(3)
        # shape[1]=3 but X.shape[0]=2
        X = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
        with pytest.raises(ValueError, match="leading dimension"):
            op.matmat(X)

    def test_matmat_none_raises(self):
        op = LinearOperator((3, 3), matvec=lambda x: x)  # no matmat
        X = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="matmat"):
            op.matmat(X)

    def test_matmat_fn_takes_priority(self):
        called = []
        fn_a = lambda X: (called.append("a"), X)[1]
        fn_b = lambda X: (called.append("b"), X)[1]
        op = LinearOperator((2, 2), matvec=lambda x: x, matmat=fn_a, matmat_fn=fn_b)
        X = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
        op.matmat(X)
        assert called == ["b"]


class TestLinearOperatorRmatvec:
    def test_valid_rmatvec(self):
        op = _make_identity_op(3)
        x = mx.array([7.0, 8.0, 9.0], dtype=mx.float32)
        result = op.rmatvec(x)
        mx.eval(result)
        np.testing.assert_allclose(np.array(result), [7.0, 8.0, 9.0])

    def test_rmatvec_rank2_raises(self):
        op = _make_identity_op(3)
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=mx.float32)
        with pytest.raises(ValueError, match="rank-1"):
            op.rmatvec(X)

    def test_rmatvec_wrong_length_raises(self):
        op = _make_identity_op(3)
        x = mx.array([1.0, 2.0], dtype=mx.float32)  # shape[0]=3 but len 2
        with pytest.raises(ValueError, match="length"):
            op.rmatvec(x)

    def test_rmatvec_none_raises(self):
        op = LinearOperator((3, 3), matvec=lambda x: x)  # no rmatvec
        x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="rmatvec"):
            op.rmatvec(x)

    def test_rmatvec_fn_takes_priority(self):
        called = []
        fn_a = lambda x: (called.append("a"), x)[1]
        fn_b = lambda x: (called.append("b"), x)[1]
        op = LinearOperator((2, 2), matvec=lambda x: x, rmatvec=fn_a, rmatvec_fn=fn_b)
        x = mx.array([1.0, 2.0], dtype=mx.float32)
        op.rmatvec(x)
        assert called == ["b"]


class TestLinearOperatorMatmul:
    def test_rank3_raises(self):
        op = _make_identity_op(3)
        x = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
        with pytest.raises(ValueError, match="rank-1 or rank-2"):
            op @ x


class TestAsLinearOperator:
    def test_passthrough_linear_operator(self):
        op = _make_identity_op(3)
        result = aslinearoperator(op)
        assert result is op

    def test_from_csr_array(self):
        csr = _make_2x2_csr()
        op = aslinearoperator(csr)
        assert isinstance(op, LinearOperator)
        assert op.shape == (2, 2)
        x = mx.array([1.0, 0.0], dtype=mx.float32)
        y = op @ x
        mx.eval(y)
        np.testing.assert_allclose(np.array(y), [2.0, 1.0], rtol=1e-5)

    def test_from_csr_rmatvec(self):
        # rmatvec uses adjoint (A.H @ x)
        csr = _make_2x2_csr()
        op = aslinearoperator(csr)
        x = mx.array([1.0, 0.0], dtype=mx.float32)
        y = op.rmatvec(x)
        mx.eval(y)
        # For real matrices, A.H == A.T
        A_dense = np.array(csr.todense())
        np.testing.assert_allclose(
            np.array(y), A_dense.T @ np.array([1.0, 0.0]), rtol=1e-5
        )

    def test_from_csr_matmat(self):
        csr = _make_2x2_csr()
        op = aslinearoperator(csr)
        X = mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32)
        Y = op.matmat(X)
        mx.eval(Y)
        A = np.array(csr.todense())
        np.testing.assert_allclose(np.array(Y), A @ np.eye(2), rtol=1e-5)

    def test_from_coo_array(self):
        coo = ms.coo_array(
            (
                mx.array([2.0, 3.0], dtype=mx.float32),
                (mx.array([0, 1], dtype=mx.int32), mx.array([0, 1], dtype=mx.int32)),
            ),
            shape=(2, 2),
        )
        op = aslinearoperator(coo)
        assert isinstance(op, LinearOperator)
        x = mx.array([1.0, 1.0], dtype=mx.float32)
        y = op @ x
        mx.eval(y)
        np.testing.assert_allclose(np.array(y), [2.0, 3.0], rtol=1e-5)

    def test_from_tuple_shape_matvec(self):
        fn = lambda x: 2.0 * x
        op = aslinearoperator(((3, 3), fn))
        assert op.shape == (3, 3)
        x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        y = op @ x
        mx.eval(y)
        np.testing.assert_allclose(np.array(y), [2.0, 4.0, 6.0])

    def test_from_tuple_with_matmat(self):
        fn = lambda x: 2.0 * x
        op = aslinearoperator(((3, 3), fn, fn))
        assert op.shape == (3, 3)
        X = mx.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=mx.float32)
        Y = op.matmat(X)
        mx.eval(Y)
        np.testing.assert_allclose(np.array(Y), 2.0 * np.array(X))

    def test_from_scipy_sparse(self):
        scipy_sparse = pytest.importorskip("scipy.sparse")
        A = scipy_sparse.eye(3, format="csr", dtype=np.float32)
        op = aslinearoperator(A)
        assert isinstance(op, LinearOperator)
        assert op.shape == (3, 3)

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="aslinearoperator"):
            aslinearoperator("not-a-matrix")

    def test_unknown_type_raises_on_dense(self):
        x = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        with pytest.raises(TypeError, match="aslinearoperator"):
            aslinearoperator(x)
