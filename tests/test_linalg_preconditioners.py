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
import mlx_sparse._native as _native
from mlx_sparse import linalg
from mlx_sparse._ext_loader import extension_available
from mlx_sparse.linalg import preconditioners


def _csr_from_scipy(mx, scipy_csr):
    scipy_csr = scipy_csr.astype(np.float32).tocsr()
    return ms.csr_array(
        (
            mx.array(scipy_csr.data, dtype=mx.float32),
            mx.array(scipy_csr.indices, dtype=mx.int32),
            mx.array(scipy_csr.indptr, dtype=mx.int32),
        ),
        shape=scipy_csr.shape,
        canonical=True,
    )


def _scipy_cg(scipy_linalg, A, b, *, M=None, rtol=1e-6, atol=0.0, maxiter=64):
    try:
        return scipy_linalg.cg(A, b, M=M, rtol=rtol, atol=atol, maxiter=maxiter)
    except TypeError:
        return scipy_linalg.cg(A, b, M=M, tol=rtol, maxiter=maxiter)


def _scipy_gmres(
    scipy_linalg,
    A,
    b,
    *,
    M=None,
    rtol=1e-6,
    atol=0.0,
    restart=None,
    maxiter=64,
):
    try:
        return scipy_linalg.gmres(
            A, b, M=M, rtol=rtol, atol=atol, restart=restart, maxiter=maxiter
        )
    except TypeError:
        return scipy_linalg.gmres(A, b, M=M, tol=rtol, restart=restart, maxiter=maxiter)


def _relative_residual(A_dense, x, b):
    return np.linalg.norm(A_dense @ x - b) / max(np.linalg.norm(b), 1.0)


def _spd_2x2(mx):
    return ms.csr_array(
        (
            mx.array([4.0, 1.0, 1.0, 3.0], dtype=mx.float32),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            mx.array([0, 2, 4], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )


def _diagonal_system(mx):
    return ms.csr_array(
        (
            mx.array([1.0e-6, 1.0, 10.0], dtype=mx.float32),
            mx.array([0, 1, 2], dtype=mx.int32),
            mx.array([0, 1, 2, 3], dtype=mx.int32),
        ),
        shape=(3, 3),
        canonical=True,
    )


def _general_3x3(mx):
    return ms.csr_array(
        (
            mx.array([5.0, -1.0, 0.5, 4.0, -1.5, 1.0, 3.5], dtype=mx.float32),
            mx.array([0, 1, 0, 1, 2, 1, 2], dtype=mx.int32),
            mx.array([0, 2, 5, 7], dtype=mx.int32),
        ),
        shape=(3, 3),
        canonical=True,
    )


def test_preconditioner_namespace_is_public():
    assert "preconditioners" in linalg.__all__
    assert linalg.preconditioners is preconditioners
    assert callable(preconditioners.jacobi)
    assert callable(preconditioners.diagonal)
    assert callable(preconditioners.identity)
    assert callable(preconditioners.from_factorized)
    assert callable(preconditioners.exact)


def test_preconditioner_metadata_is_explicit_and_conservative(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _spd_2x2(mx)

    identity = preconditioners.identity(A)
    diagonal = preconditioners.diagonal(mx.array([4.0, 3.0], dtype=mx.float32))
    jacobi_unchecked = preconditioners.jacobi(A)
    jacobi_checked = preconditioners.jacobi(A, check=True)

    assert identity.shape == A.shape
    assert identity.nnz == A.shape[0]
    assert identity.setup_device == "none"
    assert identity.apply_device == "none"
    assert identity.setup_info["is_positive_definite"] is True

    assert diagonal.shape == A.shape
    assert diagonal.nnz == A.shape[0]
    assert diagonal.setup_device == "host_validation"
    assert diagonal.apply_device == "native_cpu_or_metal"
    assert diagonal.setup_info["kind"] == "diagonal"
    assert diagonal.setup_info["is_positive_definite"] is False

    assert jacobi_unchecked.is_symmetric is True
    assert jacobi_unchecked.is_positive_definite is False
    assert jacobi_unchecked.setup_device == "native_sparse_diagonal"
    assert jacobi_unchecked.apply_device == "native_cpu_or_metal"
    assert jacobi_unchecked.setup_info["checked"] is False
    assert jacobi_unchecked.setup_info["positive_diagonal"] is None

    assert jacobi_checked.is_positive_definite is True
    assert jacobi_checked.setup_info["checked"] is True
    assert jacobi_checked.setup_info["positive_diagonal"] is True
    assert jacobi_checked.setup_info["omega"] == pytest.approx(1.0)
    assert jacobi_checked.setup_info["shift"] == pytest.approx(0.0)


def test_aspreconditioner_wraps_callable_inverse_apply(mx, to_numpy):
    scale = mx.array([0.5, 0.25], dtype=mx.float32)
    M = preconditioners.aspreconditioner(lambda x: scale * x, (2, 2))

    assert isinstance(M, preconditioners.CallablePreconditioner)
    assert M.shape == (2, 2)
    assert M.dtype == mx.float32
    assert M.nnz == -1
    assert M.setup_device == "python_host"
    assert M.apply_device == "python_host"
    assert M.setup_info["assume_inverse"] is True
    got = M(mx.array([2.0, 8.0], dtype=mx.float32))
    np.testing.assert_allclose(to_numpy(got), [1.0, 2.0], rtol=1e-6)


def test_identity_preconditioner_validates_rank_and_finiteness(mx, to_numpy):
    M = preconditioners.identity((2, 2))
    rhs = mx.array([[1.0, -2.0], [3.0, 4.0]], dtype=mx.float32)

    got = M(rhs)

    np.testing.assert_allclose(to_numpy(got), to_numpy(rhs), rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="finite"):
        M(mx.array([1.0, np.nan], dtype=mx.float32))


def test_callable_preconditioner_rejects_noncallable_and_dtype(mx):
    with pytest.raises(TypeError, match="callable"):
        preconditioners.CallablePreconditioner(object(), (2, 2))

    with pytest.raises(TypeError, match="float32"):
        preconditioners.CallablePreconditioner(lambda x: x, (2, 2), dtype=mx.float16)


def test_callable_preconditioner_rejects_wrong_output_shape(mx):
    M = preconditioners.aspreconditioner(
        lambda x: mx.ones((3,), dtype=mx.float32), (2, 2)
    )

    with pytest.raises(ValueError, match="output shape"):
        M(mx.ones((2,), dtype=mx.float32))


def test_callable_preconditioner_rejects_shape_preserving_contract_mismatch(mx):
    M = preconditioners.aspreconditioner(
        lambda x: mx.ones((2, 1), dtype=mx.float32), (2, 2)
    )

    with pytest.raises(ValueError, match="does not match input shape"):
        M(mx.ones((2,), dtype=mx.float32))


def test_from_factorized_wraps_native_lu_metadata_and_rank2_apply(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _general_3x3(mx)
    factor = linalg.sparse_lu(A)
    M = preconditioners.from_factorized(factor)
    rhs_np = np.array([[1.0, 0.25], [-2.0, 1.5], [0.5, -0.75]], dtype=np.float32)

    got = M(mx.array(rhs_np, dtype=mx.float32))
    expected = np.linalg.solve(to_numpy(A.todense()), rhs_np)

    assert isinstance(M, preconditioners.ExactFactorPreconditioner)
    assert M.shape == A.shape
    assert M.method == "lu"
    assert M.backend == "native"
    assert M.setup_device == "native_cpu"
    assert M.apply_device == "native_cpu_or_metal"
    assert M.dtype == mx.float32
    assert M.nnz == factor.L.nnz + factor.U.nnz
    assert M.native_apply_kind == "lu"
    assert M.native_factorization is factor
    assert M.setup_info["solver_type"] == "SparseLU"
    assert M.setup_info["has_native_solver_apply"] is True
    np.testing.assert_allclose(
        to_numpy(M.matvec(mx.array(rhs_np[:, 0], dtype=mx.float32))),
        expected[:, 0],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(to_numpy(got), expected, rtol=1e-4, atol=1e-4)


def test_from_factorized_wraps_native_cholesky_metadata_and_rank1_apply(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _spd_2x2(mx)
    factor = linalg.sparse_cholesky(A)
    M = preconditioners.from_factorized(factor)
    rhs_np = np.array([1.0, 2.0], dtype=np.float32)

    got = M(mx.array(rhs_np, dtype=mx.float32))
    expected = np.linalg.solve(to_numpy(A.todense()), rhs_np)

    assert M.method == "cholesky"
    assert M.backend == "native"
    assert M.is_symmetric is True
    assert M.is_positive_definite is True
    assert M.nnz == factor.L.nnz
    assert M.native_apply_kind == "cholesky"
    assert M.native_factorization is factor
    assert M.setup_info["has_native_solver_apply"] is True
    np.testing.assert_allclose(to_numpy(got), expected, rtol=2e-4, atol=2e-4)


def test_aspreconditioner_preserves_factorized_solve_backend_metadata(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _general_3x3(mx)
    solver = linalg.factorized(A, method="lu")

    M = preconditioners.aspreconditioner(solver, A)
    got = M(mx.array([1.0, -2.0, 0.5], dtype=mx.float32))
    expected = np.linalg.solve(
        to_numpy(A.todense()), np.array([1.0, -2.0, 0.5], dtype=np.float32)
    )

    assert isinstance(M, preconditioners.ExactFactorPreconditioner)
    assert M.method == solver.method
    assert M.backend == solver.backend
    assert M.setup_info["backend"] == solver.backend
    assert M.setup_info["has_native_solver_apply"] is True
    if solver.backend == "accelerate":
        assert M.native_apply_kind == "accelerate"
        assert M.apply_device == "accelerate_cpu"
    else:
        assert M.native_apply_kind == "lu"
        assert M.apply_device == "native_cpu_or_metal"
    np.testing.assert_allclose(to_numpy(got), expected, rtol=1e-3, atol=1e-3)


def test_exact_convenience_preconditioner_matches_direct_solve(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _general_3x3(mx)
    rhs_np = np.array([1.0, -2.0, 0.5], dtype=np.float32)

    M = preconditioners.exact(A, method="lu")
    got = M(mx.array(rhs_np, dtype=mx.float32))
    expected = np.linalg.solve(to_numpy(A.todense()), rhs_np)

    assert M.kind == "exact"
    assert M.shape == A.shape
    np.testing.assert_allclose(to_numpy(got), expected, rtol=1e-3, atol=1e-3)


def test_native_exact_lu_apply_matches_numpy_rank1_and_rank2(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _general_3x3(mx)
    factor = linalg.sparse_lu(A)
    rhs_np = np.array([[1.0, 0.25], [-2.0, 1.5], [0.5, -0.75]], dtype=np.float32)

    got_matrix = _native.csr_exact_lu_preconditioner_apply(
        factor.perm,
        factor.L.data,
        factor.L.indices,
        factor.L.indptr,
        factor.U.data,
        factor.U.indices,
        factor.U.indptr,
        mx.array(rhs_np, dtype=mx.float32),
        A.shape,
    )
    got_vector = _native.csr_exact_lu_preconditioner_apply(
        factor.perm,
        factor.L.data,
        factor.L.indices,
        factor.L.indptr,
        factor.U.data,
        factor.U.indices,
        factor.U.indptr,
        mx.array(rhs_np[:, 0], dtype=mx.float32),
        A.shape,
    )
    expected = np.linalg.solve(to_numpy(A.todense()), rhs_np)

    np.testing.assert_allclose(to_numpy(got_matrix), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        to_numpy(got_vector), expected[:, 0], rtol=1e-4, atol=1e-4
    )


def test_native_exact_cholesky_apply_matches_numpy_rank1_and_rank2(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _spd_2x2(mx)
    factor = linalg.sparse_cholesky(A)
    upper = factor._upper()
    rhs_np = np.array([[1.0, 0.25], [2.0, -0.75]], dtype=np.float32)

    got_matrix = _native.csr_exact_cholesky_preconditioner_apply(
        factor.L.data,
        factor.L.indices,
        factor.L.indptr,
        upper.data,
        upper.indices,
        upper.indptr,
        mx.array(rhs_np, dtype=mx.float32),
        A.shape,
    )
    got_vector = _native.csr_exact_cholesky_preconditioner_apply(
        factor.L.data,
        factor.L.indices,
        factor.L.indptr,
        upper.data,
        upper.indices,
        upper.indptr,
        mx.array(rhs_np[:, 0], dtype=mx.float32),
        A.shape,
    )
    expected = np.linalg.solve(to_numpy(A.todense()), rhs_np)

    np.testing.assert_allclose(to_numpy(got_matrix), expected, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        to_numpy(got_vector), expected[:, 0], rtol=2e-4, atol=2e-4
    )


def test_from_factorized_preserves_accelerate_metadata_without_dependency(mx, to_numpy):
    class _IdentitySolver:
        def solve(self, rhs):
            return rhs

    solver = linalg.FactorizedSolve(
        _solver=_IdentitySolver(),
        shape=(2, 2),
        method="lu",
        backend="accelerate",
        rhs_size=2,
        solution_size=2,
    )
    M = preconditioners.from_factorized(solver)
    rhs = mx.array([[1.0, -2.0], [0.5, 3.0]], dtype=mx.float32)

    got = M(rhs)

    assert M.backend == "accelerate"
    assert M.setup_device == "accelerate_cpu"
    assert M.apply_device == "accelerate_cpu"
    assert M.native_apply_kind is None
    assert M.native_factorization is None
    assert M.setup_info["backend"] == "accelerate"
    assert M.setup_info["apply_device"] == "accelerate_cpu"
    assert M.setup_info["has_native_solver_apply"] is False
    np.testing.assert_allclose(to_numpy(got), to_numpy(rhs), rtol=0.0, atol=0.0)


def test_exact_factor_preconditioner_rejects_rectangular_factorized_solve(mx):
    class _RectangularSolver:
        def solve(self, rhs):
            return rhs[:2]

    solver = linalg.FactorizedSolve(
        _solver=_RectangularSolver(),
        shape=(3, 2),
        method="qr",
        backend="test",
        rhs_size=3,
        solution_size=2,
    )

    with pytest.raises(ValueError, match="square"):
        preconditioners.from_factorized(solver)


def test_exact_factor_preconditioner_rejects_inconsistent_factorized_sizes(mx):
    class _IdentitySolver:
        def solve(self, rhs):
            return rhs

    solver = linalg.FactorizedSolve(
        _solver=_IdentitySolver(),
        shape=(2, 2),
        method="lu",
        backend="test",
        rhs_size=3,
        solution_size=2,
    )

    with pytest.raises(ValueError, match="matching RHS"):
        preconditioners.from_factorized(solver)


def test_exact_factor_preconditioner_rejects_bad_wrapper_metadata(mx):
    class _NoSolve:
        pass

    with pytest.raises(TypeError, match="solve"):
        preconditioners.ExactFactorPreconditioner(
            solver=_NoSolve(), shape=(2, 2), method="lu", backend="native"
        )

    with pytest.raises(ValueError, match="factor_nnz"):
        preconditioners.ExactFactorPreconditioner(
            solver=preconditioners.identity((2, 2)),
            shape=(2, 2),
            method="lu",
            backend="native",
            factor_nnz=-2,
        )


def test_exact_factor_preconditioner_rejects_nonfinite_output(mx):
    class _BadSolver:
        def solve(self, rhs):
            return mx.array([np.nan, 0.0], dtype=mx.float32)

    solver = linalg.FactorizedSolve(
        _solver=_BadSolver(),
        shape=(2, 2),
        method="lu",
        backend="test",
        rhs_size=2,
        solution_size=2,
    )
    M = preconditioners.from_factorized(solver)

    with pytest.raises(ValueError, match="finite"):
        M(mx.ones((2,), dtype=mx.float32))


def test_exact_factor_preconditioner_rejects_output_shape_mismatch(mx):
    class _BadShapeSolver:
        def solve(self, rhs):
            return mx.ones((2, 1), dtype=mx.float32)

    solver = linalg.FactorizedSolve(
        _solver=_BadShapeSolver(),
        shape=(2, 2),
        method="lu",
        backend="test",
        rhs_size=2,
        solution_size=2,
    )
    M = preconditioners.from_factorized(solver)

    with pytest.raises(ValueError, match="does not match input shape"):
        M(mx.ones((2,), dtype=mx.float32))


def test_from_factorized_rejects_unsupported_solver_object():
    with pytest.raises(TypeError, match="FactorizedSolve"):
        preconditioners.from_factorized(object())


def test_diagonal_preconditioner_apply_rank1_and_rank2(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    M = preconditioners.diagonal(mx.array([2.0, 4.0], dtype=mx.float32))

    got_vector = M(mx.array([2.0, 8.0], dtype=mx.float32))
    got_matrix = M(mx.array(np.array([[2.0, 4.0], [8.0, 12.0]], dtype=np.float32)))

    np.testing.assert_allclose(to_numpy(got_vector), [1.0, 2.0], rtol=1e-6)
    np.testing.assert_allclose(
        to_numpy(got_matrix),
        np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        rtol=1e-6,
    )


def test_diagonal_preconditioner_promotes_low_precision_rhs(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    M = preconditioners.diagonal(mx.array([2.0, 4.0], dtype=mx.float32))

    got = M(mx.array([2.0, 8.0], dtype=mx.float16))

    assert got.dtype == mx.float32
    np.testing.assert_allclose(to_numpy(got), [1.0, 2.0], rtol=1e-6)


def test_diagonal_preconditioner_rejects_nonfinite_setup_values(mx):
    with pytest.raises(ValueError, match="finite"):
        preconditioners.diagonal(
            mx.array([1.0, np.inf], dtype=mx.float32), inverse=True
        )


def test_diagonal_preconditioner_rejects_shape_and_policy_errors(mx):
    with pytest.raises(ValueError, match="expected 3"):
        preconditioners.diagonal(mx.array([1.0, 2.0], dtype=mx.float32), shape=(3, 3))

    with pytest.raises(ValueError, match="zero_atol"):
        preconditioners.diagonal(mx.array([1.0, 2.0], dtype=mx.float32), zero_atol=-1.0)

    with pytest.raises(ValueError, match="zero or near-zero"):
        preconditioners.diagonal(mx.array([0.0, 2.0], dtype=mx.float32))

    with pytest.raises(TypeError, match="float32"):
        preconditioners.diagonal(
            mx.array([1.0, 2.0], dtype=mx.float32), dtype=mx.float16
        )


def test_diagonal_preconditioner_rejects_direct_length_mismatch(mx):
    with pytest.raises(ValueError, match="inverse_diagonal has length"):
        preconditioners.DiagonalPreconditioner(
            mx.array([1.0, 2.0], dtype=mx.float32), (3, 3)
        )


def test_diagonal_preconditioner_rejects_nonfinite_rhs(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    M = preconditioners.diagonal(mx.array([2.0, 4.0], dtype=mx.float32))

    with pytest.raises(ValueError, match="right-hand side.*finite"):
        M(mx.array([1.0, np.nan], dtype=mx.float32))


def test_identity_preconditioner_rejects_unsupported_dtype(mx):
    with pytest.raises(TypeError, match="float32"):
        preconditioners.identity((2, 2), dtype=mx.float16)


def test_jacobi_accepts_csr_coo_and_csc(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    coo = ms.coo_array(
        (
            mx.array([2.0, 2.0, -1.0, 5.0], dtype=mx.float32),
            (
                mx.array([0, 0, 0, 1], dtype=mx.int32),
                mx.array([0, 0, 1, 1], dtype=mx.int32),
            ),
        ),
        shape=(2, 2),
    )
    rhs = mx.array([4.0, 5.0], dtype=mx.float32)

    for sparse in (coo.tocsr(), coo, coo.tocsc()):
        M = preconditioners.jacobi(sparse)
        np.testing.assert_allclose(to_numpy(M(rhs)), [1.0, 1.0], rtol=1e-6)


@pytest.mark.cpu_only
def test_jacobi_csr_coo_csc_cpu_correctness(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    _assert_jacobi_csr_coo_csc_correctness(mx, to_numpy)


@pytest.mark.gpu
def test_jacobi_csr_coo_csc_gpu_correctness(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    _assert_jacobi_csr_coo_csc_correctness(mx, to_numpy)


def test_jacobi_rejects_zero_diagonal_by_default(mx):
    A = ms.csr_array(
        (
            mx.array([0.0, 2.0], dtype=mx.float32),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )

    with pytest.raises(ValueError, match="zero or near-zero"):
        preconditioners.jacobi(A)


def test_jacobi_rejects_invalid_policy_rectangular_and_nonfinite_shift(mx):
    A = _spd_2x2(mx)

    with pytest.raises(ValueError, match="zero_policy"):
        preconditioners.jacobi(A, zero_policy="ignore")

    with pytest.raises(ValueError, match="finite"):
        preconditioners.jacobi(A, shift=np.inf)

    with pytest.raises(ValueError, match="zero_atol"):
        preconditioners.jacobi(A, zero_atol=-1.0)

    rectangular = ms.csr_array(
        (
            mx.array([1.0, 2.0], dtype=mx.float32),
            mx.array([0, 2], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 3),
        canonical=True,
    )
    with pytest.raises(ValueError, match="square matrix"):
        preconditioners.jacobi(rectangular)


def test_jacobi_unit_policy_explicitly_replaces_zero_diagonal(mx, to_numpy):
    A = ms.csr_array(
        (
            mx.array([0.0, 2.0], dtype=mx.float32),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )

    M = preconditioners.jacobi(A, zero_policy="unit")

    got = M(mx.array([3.0, 4.0], dtype=mx.float32))
    np.testing.assert_allclose(to_numpy(got), [3.0, 2.0], rtol=1e-6)


def test_jacobi_check_rejects_nonpositive_shifted_diagonal(mx):
    A = ms.csr_array(
        (
            mx.array([-1.0, 2.0], dtype=mx.float32),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )

    with pytest.raises(ValueError, match="strictly positive"):
        preconditioners.jacobi(A, check=True)


def test_jacobi_check_rejects_zero_diagonal_even_with_unit_policy(mx):
    A = ms.csr_array(
        (
            mx.array([0.0, 2.0], dtype=mx.float32),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )

    with pytest.raises(ValueError, match="strictly positive"):
        preconditioners.jacobi(A, zero_policy="unit", check=True)


def test_jacobi_check_rejects_nonpositive_omega(mx):
    A = _spd_2x2(mx)

    with pytest.raises(ValueError, match="omega must be positive"):
        preconditioners.jacobi(A, omega=0.0, check=True)


def test_jacobi_setup_does_not_mutate_input_csr_metadata(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    data = mx.array([2.0, 1.0, -1.0, 5.0], dtype=mx.float32)
    indices = mx.array([0, 0, 1, 1], dtype=mx.int32)
    indptr = mx.array([0, 3, 4], dtype=mx.int32)
    csr = ms.csr_array(
        (data, indices, indptr),
        shape=(2, 2),
        sorted_indices=True,
        canonical=False,
        validate="full",
    )
    before = (
        to_numpy(csr.data).copy(),
        to_numpy(csr.indices).copy(),
        to_numpy(csr.indptr).copy(),
        csr.sorted_indices,
        csr.has_canonical_format,
    )

    M = preconditioners.jacobi(csr)

    assert M.shape == csr.shape
    assert csr.sorted_indices == before[3]
    assert csr.has_canonical_format == before[4]
    np.testing.assert_array_equal(to_numpy(csr.data), before[0])
    np.testing.assert_array_equal(to_numpy(csr.indices), before[1])
    np.testing.assert_array_equal(to_numpy(csr.indptr), before[2])


def test_cg_identity_preconditioner_uses_native_cg(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _spd_2x2(mx)
    b = mx.array([1.0, 2.0], dtype=mx.float32)

    x_base, info_base = linalg.cg(A, b, rtol=1e-6, maxiter=32)
    x_identity, info_identity = linalg.cg(
        A, b, M=preconditioners.identity(A), rtol=1e-6, maxiter=32
    )

    assert info_base == 0
    assert info_identity == 0
    np.testing.assert_allclose(to_numpy(x_identity), to_numpy(x_base), rtol=1e-6)


def test_aspreconditioner_rejects_ambiguous_or_mismatched_inputs(mx):
    with pytest.raises(ValueError, match="A is required"):
        preconditioners.aspreconditioner(None)

    with pytest.raises(ValueError, match="does not match"):
        preconditioners.aspreconditioner(preconditioners.identity((2, 2)), (3, 3))

    with pytest.raises(TypeError, match="sparse matrices"):
        preconditioners.aspreconditioner(_spd_2x2(mx), _spd_2x2(mx))

    with pytest.raises(TypeError, match="must apply the inverse"):
        preconditioners.aspreconditioner(lambda x: x, (2, 2), assume_inverse=False)

    with pytest.raises(ValueError, match="A is required"):
        preconditioners.aspreconditioner(lambda x: x)

    with pytest.raises(TypeError, match="M must be"):
        preconditioners.aspreconditioner(object(), (2, 2))


def test_aspreconditioner_rejects_custom_object_shape_mismatch(mx):
    class _SolveObject:
        shape = (3, 3)

        def solve(self, rhs):
            return rhs

    with pytest.raises(ValueError, match="does not match"):
        preconditioners.aspreconditioner(_SolveObject(), (2, 2))

    with pytest.raises(TypeError, match="must apply the inverse"):
        preconditioners.aspreconditioner(_SolveObject(), (3, 3), assume_inverse=False)


def test_gmres_identity_preconditioner_matches_native_unpreconditioned(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _spd_2x2(mx)
    b = mx.array([1.0, 2.0], dtype=mx.float32)

    x_base, info_base = linalg.gmres(A, b, rtol=1e-6, maxiter=32)
    x_identity, info_identity = linalg.gmres(
        A, b, M=preconditioners.identity(A), rtol=1e-6, maxiter=32
    )

    assert info_base == 0
    assert info_identity == 0
    np.testing.assert_allclose(to_numpy(x_identity), to_numpy(x_base), rtol=1e-6)


def test_gmres_diagonal_left_preconditioner_matches_scipy(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    A_dense = np.array(
        [
            [7.0, -2.0, 0.5, 0.0],
            [1.0, 5.5, -1.0, 0.25],
            [0.0, -0.5, 4.5, 1.0],
            [0.25, 0.0, -0.75, 3.75],
        ],
        dtype=np.float32,
    )
    scipy_A = scipy_sparse.csr_array(A_dense)
    A = _csr_from_scipy(mx, scipy_A)
    b_np = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)

    M = preconditioners.jacobi(A)
    x_mlx, info_mlx = linalg.gmres(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=M,
        rtol=1e-6,
        atol=1e-7,
        restart=4,
        maxiter=32,
    )

    inv_diag = 1.0 / scipy_A.diagonal()
    M_scipy = scipy_linalg.LinearOperator(
        scipy_A.shape,
        matvec=lambda x: inv_diag * x,
        dtype=np.float32,
    )
    x_scipy, info_scipy = _scipy_gmres(
        scipy_linalg,
        scipy_A,
        b_np,
        M=M_scipy,
        rtol=1e-6,
        atol=1e-7,
        restart=4,
        maxiter=32,
    )

    x_np = to_numpy(x_mlx)
    assert info_mlx == 0
    assert info_scipy == 0
    np.testing.assert_allclose(x_np, x_scipy, rtol=2e-4, atol=2e-4)
    assert _relative_residual(A_dense, x_np, b_np) <= 2e-6


def test_native_gmres_jacobi_reports_true_residual_and_iterations(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    A_dense = np.array(
        [
            [5.0, -1.0, 0.25, 0.0],
            [1.25, 4.0, -0.5, 0.25],
            [0.0, 0.75, 3.5, -1.0],
            [0.5, 0.0, 1.0, 4.5],
        ],
        dtype=np.float32,
    )
    A = _csr_from_scipy(mx, scipy_sparse.csr_array(A_dense))
    b_np = np.array([2.0, -1.0, 0.25, 3.0], dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)
    x0 = mx.zeros((A.shape[0],), dtype=mx.float32)
    M = preconditioners.jacobi(A)

    x, info, residual, iterations = _native.csr_gmres_jacobi(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        M.inverse_diagonal,
        A.shape,
        rtol=1e-6,
        atol=1e-7,
        restart=4,
        maxiter=32,
    )

    x_np = to_numpy(x)
    residual_np = float(np.asarray(to_numpy(residual)).item())
    true_residual = np.linalg.norm(A_dense @ x_np - b_np)
    assert int(np.asarray(to_numpy(info)).item()) == 0
    assert int(np.asarray(to_numpy(iterations)).item()) <= 8
    assert residual_np == pytest.approx(true_residual, abs=1e-5)
    assert true_residual <= 2e-6 * max(np.linalg.norm(b_np), 1.0)


def test_native_gmres_jacobi_rejects_nonfinite_inverse_diagonal(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _spd_2x2(mx)
    b = mx.array([1.0, 2.0], dtype=mx.float32)
    x0 = mx.zeros((2,), dtype=mx.float32)
    inv_diag = mx.array([np.inf, 1.0], dtype=mx.float32)

    x, info, residual, iterations = _native.csr_gmres_jacobi(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        inv_diag,
        A.shape,
        rtol=1e-6,
        atol=1e-8,
        restart=2,
        maxiter=8,
    )

    assert int(np.asarray(to_numpy(info)).item()) == -3
    assert int(np.asarray(to_numpy(iterations)).item()) == 0
    assert np.isinf(float(np.asarray(to_numpy(residual)).item()))
    np.testing.assert_allclose(to_numpy(x), [0.0, 0.0], rtol=0.0, atol=0.0)


def test_gmres_callable_left_preconditioner_converges(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    A_dense = np.array(
        [
            [6.0, -1.0, 0.0],
            [0.5, 4.0, -1.5],
            [0.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    A = _csr_from_scipy(mx, scipy_sparse.csr_array(A_dense))
    b_np = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    inv_diag = mx.array(1.0 / np.diag(A_dense), dtype=mx.float32)

    x, info = linalg.gmres(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=lambda rhs: inv_diag * rhs,
        rtol=1e-6,
        atol=1e-7,
        restart=3,
        maxiter=24,
    )

    x_np = to_numpy(x)
    expected = np.linalg.solve(A_dense.astype(np.float64), b_np.astype(np.float64))
    assert info == 0
    np.testing.assert_allclose(x_np, expected, rtol=2e-5, atol=2e-5)
    assert _relative_residual(A_dense, x_np, b_np) <= 2e-6


def test_gmres_exact_factor_preconditioner_converges_to_direct_solution(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _general_3x3(mx)
    b_np = np.array([2.0, -1.0, 0.5], dtype=np.float32)

    x, info = linalg.gmres(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=preconditioners.exact(A, method="lu"),
        rtol=1e-6,
        atol=1e-7,
        restart=2,
        maxiter=4,
    )

    x_np = to_numpy(x)
    dense = to_numpy(A.todense())
    expected = np.linalg.solve(dense, b_np)
    assert info == 0
    np.testing.assert_allclose(x_np, expected, rtol=1e-3, atol=1e-3)
    assert _relative_residual(dense, x_np, b_np) <= 1e-5


def test_gmres_exact_lu_preconditioner_uses_native_path(mx, to_numpy, monkeypatch):
    if not extension_available():
        pytest.skip("native extension unavailable")
    import mlx_sparse.linalg._iterative as iterative_module

    A = _general_3x3(mx)
    b_np = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    M = preconditioners.from_factorized(linalg.sparse_lu(A))

    def forbidden_host_fallback(*args, **kwargs):
        raise AssertionError("exact LU GMRES used the Python host fallback")

    monkeypatch.setattr(iterative_module, "_left_pgmres_host", forbidden_host_fallback)
    x, info = linalg.gmres(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=M,
        rtol=1e-6,
        atol=1e-7,
        restart=2,
        maxiter=4,
    )

    x_np = to_numpy(x)
    dense = to_numpy(A.todense())
    assert info == 0
    np.testing.assert_allclose(x_np, np.linalg.solve(dense, b_np), rtol=1e-3, atol=1e-3)
    assert _relative_residual(dense, x_np, b_np) <= 1e-5


def test_gmres_exact_cholesky_preconditioner_uses_native_path(
    mx, to_numpy, monkeypatch
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    import mlx_sparse.linalg._iterative as iterative_module

    A = _spd_2x2(mx)
    b_np = np.array([1.0, 2.0], dtype=np.float32)
    M = preconditioners.from_factorized(linalg.sparse_cholesky(A))

    def forbidden_host_fallback(*args, **kwargs):
        raise AssertionError("exact Cholesky GMRES used the Python host fallback")

    monkeypatch.setattr(iterative_module, "_left_pgmres_host", forbidden_host_fallback)
    x, info = linalg.gmres(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=M,
        rtol=1e-6,
        atol=1e-7,
        restart=2,
        maxiter=4,
    )

    x_np = to_numpy(x)
    dense = to_numpy(A.todense())
    assert info == 0
    np.testing.assert_allclose(x_np, np.linalg.solve(dense, b_np), rtol=1e-4, atol=1e-4)
    assert _relative_residual(dense, x_np, b_np) <= 1e-6


def test_cg_jacobi_preconditioner_converges(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _spd_2x2(mx)
    b = mx.array([1.0, 2.0], dtype=mx.float32)

    x, info = linalg.cg(A, b, M=preconditioners.jacobi(A), rtol=1e-6, maxiter=32)

    assert info == 0
    np.testing.assert_allclose(to_numpy(x), [1.0 / 11.0, 7.0 / 11.0], rtol=1e-5)


def test_cg_jacobi_matches_dense_numpy_solve_on_small_spd(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    A_dense = np.array(
        [
            [6.0, 1.0, 0.0, 0.0],
            [1.0, 5.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    b_np = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float32)
    A = _csr_from_scipy(mx, scipy_sparse.csr_matrix(A_dense))

    x, info = linalg.cg(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=preconditioners.jacobi(A),
        rtol=1e-7,
        atol=1e-8,
        maxiter=32,
    )

    x_np = to_numpy(x)
    expected = np.linalg.solve(A_dense.astype(np.float64), b_np.astype(np.float64))
    assert info == 0
    np.testing.assert_allclose(x_np, expected, rtol=2e-5, atol=2e-5)
    assert np.linalg.norm(A_dense @ x_np - b_np) <= 2e-5


def test_cg_jacobi_matches_scipy_pcg_on_scaled_spd(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")

    n = 8
    poisson = scipy_sparse.diags(
        [-np.ones(n - 1), 2.5 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
        dtype=np.float32,
    )
    scaling = np.geomspace(1.0e-1, 1.0e1, n).astype(np.float32)
    D = scipy_sparse.diags(scaling, format="csr", dtype=np.float32)
    scipy_A = (D @ poisson @ D).astype(np.float32).tocsr()
    x_true = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    b_np = scipy_A @ x_true
    A = _csr_from_scipy(mx, scipy_A)

    x_mlx, info_mlx = linalg.cg(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=preconditioners.jacobi(A),
        rtol=1e-5,
        atol=1e-7,
        maxiter=64,
    )

    inv_diag = 1.0 / scipy_A.diagonal()
    M_scipy = scipy_linalg.LinearOperator(
        scipy_A.shape,
        matvec=lambda x: inv_diag * x,
        dtype=np.float32,
    )
    x_scipy, info_scipy = _scipy_cg(
        scipy_linalg,
        scipy_A,
        b_np,
        M=M_scipy,
        rtol=1e-5,
        atol=1e-7,
        maxiter=64,
    )

    x_np = to_numpy(x_mlx)
    assert info_mlx == 0
    assert info_scipy == 0
    np.testing.assert_allclose(x_np, x_scipy, rtol=2e-3, atol=2e-3)
    residual_bound = 5e-5 * np.linalg.norm(b_np) + 1e-5
    assert np.linalg.norm(scipy_A @ x_np - b_np) <= residual_bound


def test_cg_jacobi_handles_scaled_diagonal_system(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _diagonal_system(mx)
    x_true = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    b = A @ x_true

    x, info = linalg.cg(
        A,
        b,
        M=preconditioners.jacobi(A),
        rtol=1e-6,
        atol=1e-8,
        maxiter=16,
    )

    assert info == 0
    np.testing.assert_allclose(to_numpy(x), to_numpy(x_true), rtol=1e-4, atol=1e-4)


def test_native_pcg_reports_true_residual_and_iteration_count(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _diagonal_system(mx)
    b = mx.array([1.0, 2.0, 30.0], dtype=mx.float32)
    x0 = mx.zeros((3,), dtype=mx.float32)
    M = preconditioners.jacobi(A, check=True)

    x, info, residual, iterations = _native.csr_pcg_jacobi(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        M.inverse_diagonal,
        A.shape,
        rtol=1e-6,
        atol=1e-8,
        maxiter=16,
    )

    x_np = to_numpy(x)
    residual_np = float(np.asarray(to_numpy(residual)).item())
    assert int(np.asarray(to_numpy(info)).item()) == 0
    assert int(np.asarray(to_numpy(iterations)).item()) <= 2
    np.testing.assert_allclose(x_np, [1.0e6, 2.0, 3.0], rtol=1e-4, atol=1e-4)
    assert residual_np == pytest.approx(
        np.linalg.norm(to_numpy(A.todense()) @ x_np - to_numpy(b)), abs=1e-4
    )


def test_diagonal_preconditioner_rejects_wrong_rhs_shape(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    M = preconditioners.diagonal(mx.array([2.0, 3.0], dtype=mx.float32))

    with pytest.raises(ValueError, match="leading dimension"):
        M(mx.ones((3,), dtype=mx.float32))


def test_cg_reports_breakdown_for_indefinite_diagonal_preconditioner(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = ms.csr_array(
        (
            mx.array([1.0, 1.0], dtype=mx.float32),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )
    b = mx.array([1.0, 0.0], dtype=mx.float32)
    M = preconditioners.diagonal(mx.array([-1.0, 1.0], dtype=mx.float32), inverse=True)

    x, info = linalg.cg(A, b, M=M, rtol=1e-6, maxiter=8)

    assert info < 0
    assert np.all(np.isfinite(to_numpy(x)))


def _assert_jacobi_csr_coo_csc_correctness(mx, to_numpy):
    dense = np.array([[4.0, -1.0, 0.0], [0.5, 8.0, 1.0], [0.0, 2.0, 16.0]])
    rows, cols = np.nonzero(dense)
    data = dense[rows, cols].astype(np.float32)
    coo = ms.coo_array(
        (
            mx.array(data, dtype=mx.float32),
            (
                mx.array(rows.astype(np.int32), dtype=mx.int32),
                mx.array(cols.astype(np.int32), dtype=mx.int32),
            ),
        ),
        shape=dense.shape,
    )
    rhs_np = np.array([[4.0, 8.0], [16.0, 24.0], [32.0, 48.0]], dtype=np.float32)
    expected = rhs_np / np.diag(dense).astype(np.float32)[:, None]

    for sparse in (coo.tocsr(), coo, coo.tocsc()):
        M = preconditioners.jacobi(sparse, check=True)
        assert M.is_positive_definite is True
        got = M(mx.array(rhs_np, dtype=mx.float32))
        np.testing.assert_allclose(to_numpy(got), expected, rtol=1e-6, atol=1e-6)
