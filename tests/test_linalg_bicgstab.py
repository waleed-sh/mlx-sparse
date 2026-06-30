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
from mlx_sparse.linalg import LinearOperator, preconditioners

pytestmark = pytest.mark.native


def _require_native():
    if not extension_available():
        pytest.skip("native extension unavailable")


def _csr_from_dense(mx, dense, *, index_dtype=None):
    dense = np.asarray(dense, dtype=np.float32)
    data = []
    indices = []
    indptr = [0]
    for row in range(dense.shape[0]):
        cols = np.flatnonzero(dense[row])
        data.extend(dense[row, cols].tolist())
        indices.extend(cols.tolist())
        indptr.append(len(data))
    index_dtype = mx.int32 if index_dtype is None else index_dtype
    return ms.csr_array(
        (
            mx.array(data, dtype=mx.float32),
            mx.array(indices, dtype=index_dtype),
            mx.array(indptr, dtype=index_dtype),
        ),
        shape=dense.shape,
        canonical=True,
    )


def _as_format(A, fmt):
    if fmt == "csr":
        return A
    if fmt == "coo":
        return A.tocoo(canonical=True)
    if fmt == "csc":
        return A.tocsc(canonical=True)
    raise AssertionError(f"unexpected sparse format {fmt!r}")


def _nonsymmetric_dense():
    return np.array(
        [
            [5.0, 0.3, 0.0, 0.0],
            [-0.8, 4.0, 0.2, 0.0],
            [0.0, -0.5, 4.5, 0.1],
            [0.2, 0.0, -0.4, 3.5],
        ],
        dtype=np.float32,
    )


def _convection_diffusion_dense(n=6):
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        A[i, i] = 3.0 + 0.1 * i
        if i > 0:
            A[i, i - 1] = -1.15
        if i + 1 < n:
            A[i, i + 1] = -0.35
        if i + 2 < n:
            A[i, i + 2] = 0.08
    return A


def _relative_residual(A_dense, x, b):
    return np.linalg.norm(A_dense @ x - b) / max(np.linalg.norm(b), 1.0)


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
@pytest.mark.parametrize("index_dtype_name", ["int32", "int64"])
def test_bicgstab_nonsymmetric_formats_match_dense(mx, to_numpy, fmt, index_dtype_name):
    _require_native()
    index_dtype = getattr(mx, index_dtype_name)
    dense = _nonsymmetric_dense()
    A = _as_format(_csr_from_dense(mx, dense, index_dtype=index_dtype), fmt)
    b_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)

    x, info = linalg.bicgstab(A, b, rtol=1e-6, atol=1e-7, maxiter=64)

    got = to_numpy(x)
    assert info == 0
    np.testing.assert_allclose(got, np.linalg.solve(dense, b_np), rtol=3e-5, atol=3e-5)
    assert _relative_residual(dense, got, b_np) < 2e-6


def test_bicgstab_solver_info_and_tol_alias(mx, to_numpy):
    _require_native()
    dense = _convection_diffusion_dense()
    A = _csr_from_dense(mx, dense)
    b_np = np.linspace(0.25, 1.5, dense.shape[0], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)

    x, info = linalg.bicgstab(
        A,
        b,
        tol=1e-5,
        atol=1e-7,
        maxiter=80,
        return_info=True,
    )

    assert info.solver == "bicgstab"
    assert info.status == 0
    assert info.convergence_reason == "converged"
    assert info.breakdown_reason is None
    assert info.rtol == 1e-5
    assert info.atol == 1e-7
    assert info.maxiter == 80
    assert _relative_residual(dense, to_numpy(x), b_np) < 2e-5
    with pytest.raises(ValueError, match="tol is a compatibility alias"):
        linalg.bicgstab(A, b, rtol=1e-5, tol=1e-6)


def test_bicgstab_zero_rhs_and_already_converged_x0(mx, to_numpy):
    _require_native()
    dense = _nonsymmetric_dense()
    A = _csr_from_dense(mx, dense)
    zero = mx.zeros((dense.shape[0],), dtype=mx.float32)

    x_zero, info_zero = linalg.bicgstab(A, zero, return_info=True)

    assert info_zero.status == 0
    assert info_zero.iterations == 0
    np.testing.assert_array_equal(to_numpy(x_zero), np.zeros(dense.shape[0]))

    exact = np.linalg.solve(dense, np.arange(1, dense.shape[0] + 1, dtype=np.float32))
    b = mx.array(dense @ exact, dtype=mx.float32)
    x0 = mx.array(exact, dtype=mx.float32)
    x, info = linalg.bicgstab(A, b, x0=x0, rtol=1e-7, atol=1e-7, return_info=True)

    assert info.status == 0
    assert info.iterations == 0
    np.testing.assert_allclose(to_numpy(x), exact, rtol=0.0, atol=0.0)


def test_bicgstab_iteration_budget_reports_positive_info(mx):
    _require_native()
    dense = _convection_diffusion_dense()
    A = _csr_from_dense(mx, dense)
    b = mx.array(np.linspace(-1.0, 1.0, dense.shape[0], dtype=np.float32))

    _, info = linalg.bicgstab(A, b, maxiter=0, return_info=True)

    assert info.status > 0
    assert info.convergence_reason == "iteration_limit"
    assert info.iterations == 0


@pytest.mark.parametrize("kind", ["jacobi", "ilu0", "exact_lu"])
def test_bicgstab_native_preconditioners_converge(mx, to_numpy, kind):
    _require_native()
    dense = _nonsymmetric_dense()
    A = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)
    if kind == "jacobi":
        M = preconditioners.jacobi(A)
    elif kind == "ilu0":
        M = preconditioners.ilu0(A)
    else:
        M = preconditioners.exact(A, method="lu")

    x, info = linalg.bicgstab(
        A,
        b,
        M=M,
        rtol=1e-4,
        atol=1e-7,
        maxiter=80,
        return_info=True,
    )

    got = to_numpy(x)
    assert info.status == 0
    assert info.preconditioner in {"jacobi", "ilu0", "exact"}
    assert _relative_residual(dense, got, b_np) < 2e-5
    np.testing.assert_allclose(got, np.linalg.solve(dense, b_np), rtol=8e-5, atol=8e-5)


def test_bicgstab_callable_preconditioner_failure_is_diagnostic(mx):
    _require_native()
    dense = _nonsymmetric_dense()
    A = _csr_from_dense(mx, dense)
    b = mx.ones((dense.shape[0],), dtype=mx.float32)

    def bad_preconditioner(rhs):
        return mx.full(rhs.shape, mx.nan, dtype=mx.float32)

    _, info = linalg.bicgstab(
        A,
        b,
        M=bad_preconditioner,
        maxiter=8,
        return_info=True,
    )

    assert info.status == -3
    assert info.breakdown_reason == "non_finite"
    assert info.preconditioner == "callable"


def test_bicgstab_singular_compatible_and_incompatible_systems(mx, to_numpy):
    _require_native()
    compatible_dense = np.diag(np.array([0.0, 2.0, 3.0, 4.0], dtype=np.float32))
    compatible = _csr_from_dense(mx, compatible_dense)
    b_compatible_np = np.array([0.0, 4.0, -3.0, 8.0], dtype=np.float32)

    x, info = linalg.bicgstab(
        compatible,
        mx.array(b_compatible_np, dtype=mx.float32),
        rtol=1e-6,
        atol=1e-7,
        maxiter=16,
        return_info=True,
    )

    assert info.status == 0
    assert _relative_residual(compatible_dense, to_numpy(x), b_compatible_np) < 1e-7

    incompatible_dense = np.zeros((3, 3), dtype=np.float32)
    incompatible = _csr_from_dense(mx, incompatible_dense)
    _, bad_info = linalg.bicgstab(
        incompatible,
        mx.array([1.0, 0.0, 0.0], dtype=mx.float32),
        maxiter=8,
        return_info=True,
    )

    assert bad_info.status < 0
    assert bad_info.breakdown_reason in {"breakdown", "non_finite"}


def test_bicgstab_matrix_free_host_fallback(mx, to_numpy):
    _require_native()
    dense = _nonsymmetric_dense()
    b_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)
    A = LinearOperator(
        dense.shape,
        matvec=lambda x: mx.array(dense @ to_numpy(x), dtype=mx.float32),
    )

    x, info = linalg.bicgstab(A, b, rtol=1e-6, atol=1e-7, maxiter=64)

    got = to_numpy(x)
    assert info == 0
    assert _relative_residual(dense, got, b_np) < 2e-6


def test_bicgstab_matches_scipy_when_available(mx, to_numpy):
    _require_native()
    scipy_sparse = pytest.importorskip("scipy.sparse")
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    dense = _nonsymmetric_dense()
    A = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    x, info = linalg.bicgstab(
        A,
        mx.array(b_np, dtype=mx.float32),
        rtol=1e-6,
        atol=1e-7,
        maxiter=64,
    )
    try:
        scipy_x, scipy_info = scipy_linalg.bicgstab(
            scipy_sparse.csr_array(dense),
            b_np,
            rtol=1e-6,
            atol=1e-7,
            maxiter=64,
        )
    except TypeError:
        scipy_x, scipy_info = scipy_linalg.bicgstab(
            scipy_sparse.csr_matrix(dense),
            b_np,
            tol=1e-6,
            maxiter=64,
        )

    assert info == 0
    assert scipy_info == 0
    np.testing.assert_allclose(to_numpy(x), scipy_x, rtol=8e-5, atol=8e-5)


@pytest.mark.gpu
def test_bicgstab_gpu_native_identity_and_jacobi(mx, to_numpy):
    _require_native()
    dense = _nonsymmetric_dense()
    A = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)

    x, info = linalg.bicgstab(A, b, rtol=1e-6, atol=1e-7, maxiter=80)
    x_jacobi, info_jacobi = linalg.bicgstab(
        A,
        b,
        M=preconditioners.jacobi(A),
        rtol=1e-4,
        atol=1e-7,
        maxiter=80,
    )

    assert info == 0
    assert info_jacobi == 0
    assert _relative_residual(dense, to_numpy(x), b_np) < 2e-6
    assert _relative_residual(dense, to_numpy(x_jacobi), b_np) < 2e-5
