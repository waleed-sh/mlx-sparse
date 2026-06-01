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
from mlx_sparse.linalg import preconditioners

pytestmark = pytest.mark.native


def _require_native():
    if not extension_available():
        pytest.skip("native extension unavailable")


def _csr_from_dense(mx, dense):
    scipy_sparse = pytest.importorskip("scipy.sparse")
    csr = scipy_sparse.csr_array(np.asarray(dense, dtype=np.float32))
    return ms.csr_array(
        (
            mx.array(csr.data, dtype=mx.float32),
            mx.array(csr.indices.astype(np.int32), dtype=mx.int32),
            mx.array(csr.indptr.astype(np.int32), dtype=mx.int32),
        ),
        shape=csr.shape,
        canonical=True,
    )


def _scipy_minres(scipy_linalg, A, b, *, M=None, rtol=1e-6, shift=0.0, maxiter=64):
    try:
        return scipy_linalg.minres(A, b, M=M, rtol=rtol, shift=shift, maxiter=maxiter)
    except TypeError:
        return scipy_linalg.minres(A, b, M=M, tol=rtol, shift=shift, maxiter=maxiter)


def _relative_residual(A_dense, x, b, *, shift=0.0):
    shifted = A_dense - shift * np.eye(A_dense.shape[0], dtype=A_dense.dtype)
    return np.linalg.norm(shifted @ x - b) / max(np.linalg.norm(b), 1.0)


def test_minres_symmetric_indefinite_matches_scipy_and_dense(mx, to_numpy):
    _require_native()
    scipy_sparse = pytest.importorskip("scipy.sparse")
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    dense = np.array(
        [
            [2.0, 3.0, 0.0, 0.0],
            [3.0, 2.0, 0.5, 0.0],
            [0.0, 0.5, -1.5, 1.0],
            [0.0, 0.0, 1.0, 2.5],
        ],
        dtype=np.float32,
    )
    A = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)

    x, info = linalg.minres(A, b, rtol=1e-6, atol=1e-7, maxiter=64)
    scipy_x, scipy_info = _scipy_minres(
        scipy_linalg, scipy_sparse.csr_array(dense), b_np, rtol=1e-6, maxiter=64
    )

    assert info == 0
    assert scipy_info == 0
    got = to_numpy(x)
    np.testing.assert_allclose(got, np.linalg.solve(dense, b_np), rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(got, scipy_x, rtol=5e-5, atol=5e-5)
    assert _relative_residual(dense, got, b_np) < 2e-6


def test_minres_singular_compatible_system_converges(mx, to_numpy):
    _require_native()
    dense = np.diag(np.array([0.0, 2.0, 3.0, -4.0], dtype=np.float32))
    A = _csr_from_dense(mx, dense)
    b_np = np.array([0.0, 4.0, -3.0, 8.0], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)

    x, info = linalg.minres(A, b, rtol=1e-7, atol=1e-7, maxiter=32)

    got = to_numpy(x)
    assert info == 0
    assert np.all(np.isfinite(got))
    assert _relative_residual(dense, got, b_np) < 1e-7
    np.testing.assert_allclose(dense @ got, b_np, rtol=1e-6, atol=1e-6)


def test_minres_shift_matches_scipy_convention(mx, to_numpy):
    _require_native()
    scipy_sparse = pytest.importorskip("scipy.sparse")
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    dense = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 2.5, 0.5],
            [0.0, 0.5, 3.0],
        ],
        dtype=np.float32,
    )
    shift = 0.75
    A = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)

    x, info = linalg.minres(A, b, shift=shift, rtol=1e-6, atol=1e-7, maxiter=64)
    scipy_x, scipy_info = _scipy_minres(
        scipy_linalg,
        scipy_sparse.csr_array(dense),
        b_np,
        rtol=1e-6,
        shift=shift,
        maxiter=64,
    )

    assert info == 0
    assert scipy_info == 0
    got = to_numpy(x)
    expected = np.linalg.solve(dense - shift * np.eye(3, dtype=np.float32), b_np)
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(got, scipy_x, rtol=2e-5, atol=2e-5)
    assert _relative_residual(dense, got, b_np, shift=shift) < 2e-6


def test_minres_diagonal_preconditioner_matches_scipy_linear_operator(mx, to_numpy):
    _require_native()
    scipy_sparse = pytest.importorskip("scipy.sparse")
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    dense = np.array(
        [
            [2.0, 3.0, 0.0],
            [3.0, 2.0, 0.5],
            [0.0, 0.5, 4.0],
        ],
        dtype=np.float32,
    )
    A = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)
    M = preconditioners.jacobi(A, check=True)
    inv_diag_np = to_numpy(M.inverse_diagonal)
    scipy_M = scipy_linalg.LinearOperator(
        dense.shape,
        matvec=lambda x: inv_diag_np * x,
        dtype=np.float32,
    )

    x, info = linalg.minres(A, b, M=M, rtol=1e-6, atol=1e-7, maxiter=64)
    scipy_x, scipy_info = _scipy_minres(
        scipy_linalg,
        scipy_sparse.csr_array(dense),
        b_np,
        M=scipy_M,
        rtol=1e-6,
        maxiter=64,
    )

    assert info == 0
    assert scipy_info == 0
    got = to_numpy(x)
    np.testing.assert_allclose(got, np.linalg.solve(dense, b_np), rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(got, scipy_x, rtol=8e-5, atol=8e-5)
    assert _relative_residual(dense, got, b_np) < 2e-6


def test_minres_jacobi_preconditioner_fixes_near_singular_diagonal(mx, to_numpy):
    _require_native()
    dense = np.diag(np.array([1.0e-8, 1.0, 2.0, 3.0], dtype=np.float32))
    A = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)

    x, info = linalg.minres(
        A,
        b,
        M=preconditioners.jacobi(A, check=True),
        rtol=1e-6,
        atol=1e-7,
        maxiter=8,
    )

    got = to_numpy(x)
    assert info == 0
    assert np.all(np.isfinite(got))
    assert _relative_residual(dense, got, b_np) == pytest.approx(0.0, abs=1e-7)


def test_minres_rejects_non_spd_diagonal_preconditioner(mx):
    _require_native()
    dense = np.diag(np.array([1.0, 2.0], dtype=np.float32))
    A = _csr_from_dense(mx, dense)
    b = mx.array([1.0, 1.0], dtype=mx.float32)
    M = preconditioners.diagonal(
        mx.array([1.0, -1.0], dtype=mx.float32),
        inverse=True,
    )

    with pytest.raises(ValueError, match="symmetric positive-definite"):
        linalg.minres(A, b, M=M)

    _, info = linalg.minres(A, b, M=M, check_preconditioner=False, maxiter=8)
    assert info < 0


def test_minres_rejects_callable_and_exact_preconditioners_until_native_spd_kernels(mx):
    _require_native()
    dense = np.array([[2.0, 0.25], [0.25, 3.0]], dtype=np.float32)
    A = _csr_from_dense(mx, dense)
    b = mx.array([1.0, -2.0], dtype=mx.float32)

    with pytest.raises(TypeError, match="identity, diagonal, and Jacobi"):
        linalg.minres(A, b, M=lambda x: x)

    with pytest.raises(TypeError, match="identity, diagonal, and Jacobi"):
        linalg.minres(A, b, M=preconditioners.exact(A, method="cholesky"))
