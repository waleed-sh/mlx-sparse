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


def _reference_ilu0_from_csr(scipy_csr, *, shift=0.0, check=True):
    A = scipy_csr.astype(np.float64).tocsr()
    n, m = A.shape
    if n != m:
        raise ValueError("reference ILU0 requires a square matrix")
    rows = []
    for row in range(n):
        entries = {
            int(col): float(value)
            for col, value in zip(
                A.indices[A.indptr[row] : A.indptr[row + 1]],
                A.data[A.indptr[row] : A.indptr[row + 1]],
            )
        }
        if row not in entries:
            raise RuntimeError("missing diagonal")
        entries[row] += float(shift)
        rows.append(entries)

    L = np.zeros((n, n), dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)
    eps = np.finfo(np.float32).eps
    for row in range(n):
        work = dict(rows[row])
        for col in sorted(k for k in work if k < row):
            pivot = U[col, col]
            scale = max(1.0, np.sum(np.abs(U[col, :]))) if check else 1.0
            threshold = eps * scale if check else 0.0
            if abs(pivot) <= threshold:
                raise RuntimeError("zero pivot")
            factor = work[col] / pivot
            work[col] = factor
            for upper_col in np.flatnonzero(U[col, :]):
                if upper_col > col and upper_col in work:
                    work[upper_col] -= factor * U[col, upper_col]
        scale = max(1.0, sum(abs(value) for value in work.values()))
        threshold = eps * scale if check else 0.0
        if abs(work[row]) <= threshold:
            raise RuntimeError("zero pivot")
        for col, value in work.items():
            if col < row:
                L[row, col] = value
            else:
                U[row, col] = value
        L[row, row] = 1.0
    return L.astype(np.float32), U.astype(np.float32)


def _solve_lu_dense(L, U, rhs):
    rhs = np.asarray(rhs, dtype=np.float64)
    vector_input = rhs.ndim == 1
    if vector_input:
        rhs = rhs[:, None]
    n = L.shape[0]
    y = np.zeros_like(rhs, dtype=np.float64)
    for row in range(n):
        y[row] = rhs[row] - L[row, :row] @ y[:row]
    x = np.zeros_like(rhs, dtype=np.float64)
    for row in range(n - 1, -1, -1):
        x[row] = (y[row] - U[row, row + 1 :] @ x[row + 1 :]) / U[row, row]
    return x[:, 0].astype(np.float32) if vector_input else x.astype(np.float32)


def _reference_ic0_lower_from_csr(scipy_csr, *, shift=0.0, check=True):
    A = scipy_csr.astype(np.float64).tocsr()
    n, m = A.shape
    if n != m:
        raise ValueError("reference IC0 requires a square matrix")

    rows = []
    for row in range(n):
        entries = {
            int(col): float(value)
            for col, value in zip(
                A.indices[A.indptr[row] : A.indptr[row + 1]],
                A.data[A.indptr[row] : A.indptr[row + 1]],
            )
        }
        rows.append(entries)

    lower = [dict() for _ in range(n)]
    for row, entries in enumerate(rows):
        for col, value in entries.items():
            if col <= row:
                lower[row][col] = value
    for row, entries in enumerate(rows):
        for col, value in entries.items():
            if col <= row:
                continue
            if row in lower[col]:
                if check and not np.isclose(
                    lower[col][row], value, rtol=1e-5, atol=1e-7
                ):
                    raise RuntimeError("nonsymmetric values")
            else:
                lower[col][row] = value
    for row in range(n):
        if row not in lower[row]:
            raise RuntimeError("missing diagonal")
        lower[row][row] += float(shift)

    L = np.zeros((n, n), dtype=np.float64)
    eps = np.finfo(np.float32).eps
    for row in range(n):
        entries = lower[row]
        for col in sorted(k for k in entries if k < row):
            acc = entries[col]
            for k in sorted(k for k in entries if k < col):
                acc -= L[row, k] * L[col, k]
            threshold = (
                eps * max(1.0, np.sum(np.abs(L[col, : col + 1]))) if check else 0.0
            )
            if L[col, col] <= threshold:
                raise RuntimeError("nonpositive pivot")
            L[row, col] = acc / L[col, col]
        diag = entries[row] - np.dot(L[row, :row], L[row, :row])
        threshold = (
            eps * max(1.0, sum(abs(v) for v in entries.values())) if check else 0.0
        )
        if diag <= threshold:
            raise RuntimeError("nonpositive pivot")
        L[row, row] = np.sqrt(diag)
    return L.astype(np.float32)


def _solve_cholesky_dense(L, rhs):
    rhs = np.asarray(rhs, dtype=np.float64)
    vector_input = rhs.ndim == 1
    if vector_input:
        rhs = rhs[:, None]
    n = L.shape[0]
    y = np.zeros_like(rhs, dtype=np.float64)
    for row in range(n):
        y[row] = (rhs[row] - L[row, :row] @ y[:row]) / L[row, row]
    x = np.zeros_like(rhs, dtype=np.float64)
    for row in range(n - 1, -1, -1):
        x[row] = (y[row] - L[row + 1 :, row] @ x[row + 1 :]) / L[row, row]
    return x[:, 0].astype(np.float32) if vector_input else x.astype(np.float32)


def _reference_chebyshev_apply(A_dense, rhs, *, degree, lambda_min, lambda_max):
    A = np.asarray(A_dense, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    vector_input = rhs_arr.ndim == 1
    if vector_input:
        rhs_arr = rhs_arr[:, None]
    scale = 2.0 / (float(lambda_max) + float(lambda_min))
    alpha = 1.0 - scale * float(lambda_min)
    mu = 1.0 / alpha
    omega_prod = 2.0 / alpha
    c_prev = 1.0
    c_cur = mu
    x_prev = np.zeros_like(rhs_arr, dtype=np.float64)
    x_cur = scale * rhs_arr
    for _ in range(1, int(degree)):
        residual = rhs_arr - A @ x_cur
        c_next = 2.0 * mu * c_cur - c_prev
        omega = omega_prod * c_cur / c_next
        x_next = (1.0 - omega) * x_prev + omega * x_cur + omega * scale * residual
        x_prev, x_cur = x_cur, x_next
        c_prev, c_cur = c_cur, c_next
    return x_cur[:, 0].astype(np.float32) if vector_input else x_cur.astype(np.float32)


def _poisson_2d_scipy(scipy_sparse, grid):
    main = 4.0 * np.ones(grid, dtype=np.float32)
    off = -1.0 * np.ones(grid - 1, dtype=np.float32)
    T = scipy_sparse.diags([off, main, off], [-1, 0, 1], format="csr")
    I = scipy_sparse.eye(grid, format="csr", dtype=np.float32)
    S = scipy_sparse.diags([off, off], [-1, 1], shape=(grid, grid), format="csr")
    return (
        scipy_sparse.kron(I, T, format="csr") + scipy_sparse.kron(S, I, format="csr")
    ).astype(np.float32)


def _anisotropic_diffusion_2d_scipy(scipy_sparse, grid, *, ax=0.15, ay=1.5):
    diag = (2.0 * ax + 2.0 * ay + 0.25) * np.ones(grid, dtype=np.float32)
    off_x = -ax * np.ones(grid - 1, dtype=np.float32)
    off_y = -ay * np.ones(grid - 1, dtype=np.float32)
    T = scipy_sparse.diags([off_x, diag, off_x], [-1, 0, 1], format="csr")
    I = scipy_sparse.eye(grid, format="csr", dtype=np.float32)
    Y = scipy_sparse.diags([off_y, off_y], [-1, 1], shape=(grid, grid), format="csr")
    return (
        scipy_sparse.kron(I, T, format="csr") + scipy_sparse.kron(Y, I, format="csr")
    ).astype(np.float32)


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


def _ilu0_4x4_scipy(scipy_sparse):
    dense = np.array(
        [
            [4.0, -1.0, 0.5, 0.0],
            [1.5, 5.0, -1.0, 0.25],
            [0.0, 1.25, 4.5, -0.75],
            [0.5, 0.0, 1.0, 3.75],
        ],
        dtype=np.float32,
    )
    return scipy_sparse.csr_array(dense)


def test_preconditioner_namespace_is_public():
    assert "preconditioners" in linalg.__all__
    assert linalg.preconditioners is preconditioners
    assert callable(preconditioners.jacobi)
    assert callable(preconditioners.diagonal)
    assert callable(preconditioners.identity)
    assert callable(preconditioners.ilu0)
    assert callable(preconditioners.ichol0)
    assert callable(preconditioners.chebyshev)
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
    ilu0 = preconditioners.ilu0(A)
    ichol0 = preconditioners.ichol0(A)
    chebyshev = preconditioners.chebyshev(A, degree=2, estimate=True)

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

    assert ilu0.shape == A.shape
    assert ilu0.kind == "ilu0"
    assert ilu0.setup_device == "native_cpu"
    assert ilu0.apply_device == "native_cpu_or_metal"
    assert ilu0.is_symmetric is False
    assert ilu0.is_positive_definite is False
    assert ilu0.nnz == ilu0.nnz_L + ilu0.nnz_U
    assert ilu0.setup_info["ordering"] == "natural"
    assert ilu0.setup_info["fill"] == 0
    assert ilu0.setup_info["unit_diagonal_L"] is True

    assert ichol0.shape == A.shape
    assert ichol0.kind == "ichol0"
    assert ichol0.setup_device == "native_cpu"
    assert ichol0.apply_device == "native_cpu_or_metal"
    assert ichol0.is_symmetric is True
    assert ichol0.is_positive_definite is True
    assert ichol0.nnz == ichol0.nnz_L
    assert ichol0.setup_info["ordering"] == "natural"
    assert ichol0.setup_info["fill"] == 0
    assert ichol0.setup_info["factor"] == "lower"

    assert chebyshev.shape == A.shape
    assert chebyshev.kind == "chebyshev"
    assert chebyshev.degree == 2
    assert chebyshev.setup_device == "native_cpu"
    assert chebyshev.apply_device == "native_cpu_or_metal"
    assert chebyshev.is_symmetric is True
    assert chebyshev.is_positive_definite is True
    assert chebyshev.nnz == A.nnz
    assert chebyshev.setup_info["lambda_min"] > 0.0
    assert chebyshev.setup_info["lambda_max"] > chebyshev.setup_info["lambda_min"]
    assert "spectral_info" in chebyshev.setup_info


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


def test_aspreconditioner_accepts_documented_inverse_apply_contracts(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    A = _general_3x3(mx)
    rhs_np = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    expected = np.linalg.solve(to_numpy(A.todense()), rhs_np)

    class _SolveObject:
        shape = A.shape

        def solve(self, rhs):
            return rhs / mx.array([5.0, 4.0, 3.5], dtype=mx.float32)

    normalized_none = preconditioners.aspreconditioner(None, A)
    normalized_object = preconditioners.aspreconditioner(_SolveObject(), A)
    normalized_callable = preconditioners.aspreconditioner(
        lambda rhs: rhs / mx.array([5.0, 4.0, 3.5], dtype=mx.float32), A
    )
    normalized_preconditioner = preconditioners.aspreconditioner(
        preconditioners.jacobi(A), A
    )
    normalized_lu = preconditioners.aspreconditioner(linalg.sparse_lu(A), A)
    normalized_factorized = preconditioners.aspreconditioner(
        linalg.factorized(A, method="lu"), A
    )
    spd = _spd_2x2(mx)
    normalized_cholesky = preconditioners.aspreconditioner(
        linalg.sparse_cholesky(spd), spd
    )

    assert isinstance(normalized_none, preconditioners.IdentityPreconditioner)
    assert isinstance(normalized_object, preconditioners.CallablePreconditioner)
    assert isinstance(normalized_callable, preconditioners.CallablePreconditioner)
    assert normalized_preconditioner.kind == "jacobi"
    for exact_pc in (normalized_lu, normalized_factorized):
        assert isinstance(exact_pc, preconditioners.ExactFactorPreconditioner)
        assert exact_pc.shape == A.shape
        np.testing.assert_allclose(
            to_numpy(exact_pc(mx.array(rhs_np, dtype=mx.float32))),
            expected,
            rtol=1e-3,
            atol=1e-3,
        )
    assert normalized_cholesky.method == "cholesky"


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


def test_ilu0_setup_matches_internal_reference_and_preserves_pattern(
    mx, to_numpy, scipy_sparse
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _ilu0_4x4_scipy(scipy_sparse)
    A = _csr_from_scipy(mx, scipy_A)

    M = preconditioners.ilu0(A)
    L_ref, U_ref = _reference_ilu0_from_csr(scipy_A)

    np.testing.assert_allclose(to_numpy(M.L.todense()), L_ref, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(to_numpy(M.U.todense()), U_ref, rtol=2e-6, atol=2e-6)
    assert M.nnz_L == int(np.count_nonzero(np.tril(scipy_A.toarray())))
    assert M.nnz_U == int(np.count_nonzero(np.triu(scipy_A.toarray())))


def _assert_ilu0_apply_rank1_rank2_correctness(mx, to_numpy, scipy_sparse):
    scipy_A = _ilu0_4x4_scipy(scipy_sparse)
    A = _csr_from_scipy(mx, scipy_A)
    M = preconditioners.ilu0(A)
    L_ref, U_ref = _reference_ilu0_from_csr(scipy_A)
    rhs_matrix = np.array(
        [[1.0, 0.25], [-2.0, 1.5], [0.5, -0.75], [3.0, 0.5]],
        dtype=np.float32,
    )

    got_matrix = M(mx.array(rhs_matrix, dtype=mx.float32))
    got_vector = M(mx.array(rhs_matrix[:, 0], dtype=mx.float32))
    native_matrix = _native.csr_ilu0_preconditioner_apply(
        M.L.data,
        M.L.indices,
        M.L.indptr,
        M.U.data,
        M.U.indices,
        M.U.indptr,
        mx.array(rhs_matrix, dtype=mx.float32),
        A.shape,
    )
    expected = _solve_lu_dense(L_ref, U_ref, rhs_matrix)

    np.testing.assert_allclose(to_numpy(got_matrix), expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        to_numpy(got_vector), expected[:, 0], rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(to_numpy(native_matrix), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.cpu_only
def test_ilu0_apply_cpu_rank1_rank2_correctness(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    _assert_ilu0_apply_rank1_rank2_correctness(mx, to_numpy, scipy_sparse)


@pytest.mark.gpu
def test_ilu0_apply_gpu_rank1_rank2_correctness(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    _assert_ilu0_apply_rank1_rank2_correctness(mx, to_numpy, scipy_sparse)


def test_ilu0_reuse_analysis_preserves_apply_result(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _ilu0_4x4_scipy(scipy_sparse)
    A = _csr_from_scipy(mx, scipy_A)
    rhs = mx.array([1.0, -2.0, 0.5, 3.0], dtype=mx.float32)

    plain = preconditioners.ilu0(A, reuse_analysis=False)
    analyzed = preconditioners.ilu0(A, reuse_analysis=True)

    assert analyzed.setup_info["reuse_analysis"] is True
    np.testing.assert_allclose(
        to_numpy(analyzed(rhs)), to_numpy(plain(rhs)), rtol=1e-6, atol=1e-6
    )


def test_ilu0_shift_is_explicit_and_does_not_fill_missing_diagonal(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    zero_diag = ms.csr_array(
        (
            mx.array([0.0, 2.0], dtype=mx.float32),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )
    missing_diag = ms.csr_array(
        (
            mx.array([1.0, 2.0], dtype=mx.float32),
            mx.array([1, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )

    with pytest.raises(RuntimeError, match="pivot"):
        preconditioners.ilu0(zero_diag)
    shifted = preconditioners.ilu0(zero_diag, shift=2.0)
    np.testing.assert_allclose(
        to_numpy(shifted.U.todense()), np.diag([2.0, 4.0]), rtol=1e-6
    )
    with pytest.raises(RuntimeError, match="diagonal"):
        preconditioners.ilu0(missing_diag, shift=2.0)


def test_ilu0_failure_modes_are_explicit(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    rectangular = ms.csr_array(
        (
            mx.array([1.0, 2.0], dtype=mx.float32),
            mx.array([0, 2], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 3),
        canonical=True,
    )
    singular_structure = ms.csr_array(
        (
            mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float32),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            mx.array([0, 2, 4], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )
    complex_data = ms.csr_array(
        (
            mx.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=mx.complex64),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )

    with pytest.raises(ValueError, match="square"):
        preconditioners.ilu0(rectangular)
    with pytest.raises(RuntimeError, match="pivot"):
        preconditioners.ilu0(singular_structure)
    with pytest.raises(TypeError, match="real float"):
        preconditioners.ilu0(complex_data)
    with pytest.raises(ValueError, match="finite"):
        preconditioners.ilu0(_spd_2x2(mx), shift=np.inf)


def test_ilu0_setup_does_not_mutate_input_csr_metadata(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _ilu0_4x4_scipy(scipy_sparse)
    A = _csr_from_scipy(mx, scipy_A)
    before = (
        to_numpy(A.data).copy(),
        to_numpy(A.indices).copy(),
        to_numpy(A.indptr).copy(),
        A.sorted_indices,
        A.has_canonical_format,
    )

    M = preconditioners.ilu0(A)

    assert M.shape == A.shape
    assert A.sorted_indices == before[3]
    assert A.has_canonical_format == before[4]
    np.testing.assert_array_equal(to_numpy(A.data), before[0])
    np.testing.assert_array_equal(to_numpy(A.indices), before[1])
    np.testing.assert_array_equal(to_numpy(A.indptr), before[2])


def test_ichol0_setup_matches_internal_reference_and_preserves_lower_pattern(
    mx, to_numpy, scipy_sparse
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _poisson_2d_scipy(scipy_sparse, 3)
    A = _csr_from_scipy(mx, scipy_A)

    M = preconditioners.ichol0(A)
    L_ref = _reference_ic0_lower_from_csr(scipy_A)

    np.testing.assert_allclose(to_numpy(M.L.todense()), L_ref, rtol=2e-6, atol=2e-6)
    assert M.nnz_L == int(np.count_nonzero(np.tril(scipy_A.toarray())))
    assert M.setup_info["ordering"] == "natural"
    assert M.setup_info["fill"] == 0
    assert M.setup_info["factor"] == "lower"


def _assert_ichol0_apply_rank1_rank2_correctness(mx, to_numpy, scipy_sparse):
    scipy_A = _poisson_2d_scipy(scipy_sparse, 3)
    A = _csr_from_scipy(mx, scipy_A)
    M = preconditioners.ichol0(A)
    L_ref = _reference_ic0_lower_from_csr(scipy_A)
    rhs_matrix = np.column_stack(
        [
            np.linspace(0.25, 1.25, A.shape[0], dtype=np.float32),
            np.cos(np.linspace(0.0, 1.0, A.shape[0], dtype=np.float32)),
        ]
    ).astype(np.float32)
    upper = M._upper()

    got_matrix = M(mx.array(rhs_matrix, dtype=mx.float32))
    got_vector = M(mx.array(rhs_matrix[:, 0], dtype=mx.float32))
    native_matrix = _native.csr_ic0_preconditioner_apply(
        M.L.data,
        M.L.indices,
        M.L.indptr,
        upper.data,
        upper.indices,
        upper.indptr,
        mx.array(rhs_matrix, dtype=mx.float32),
        A.shape,
    )
    expected = _solve_cholesky_dense(L_ref, rhs_matrix)

    np.testing.assert_allclose(to_numpy(got_matrix), expected, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(
        to_numpy(got_vector), expected[:, 0], rtol=3e-5, atol=3e-5
    )
    np.testing.assert_allclose(to_numpy(native_matrix), expected, rtol=3e-5, atol=3e-5)


@pytest.mark.cpu_only
def test_ichol0_apply_cpu_rank1_rank2_correctness(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    _assert_ichol0_apply_rank1_rank2_correctness(mx, to_numpy, scipy_sparse)


@pytest.mark.gpu
def test_ichol0_apply_gpu_rank1_rank2_correctness(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    _assert_ichol0_apply_rank1_rank2_correctness(mx, to_numpy, scipy_sparse)


def test_ichol0_supports_upper_only_symmetric_storage(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    dense = np.array(
        [[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]],
        dtype=np.float32,
    )
    scipy_A = scipy_sparse.csr_array(np.triu(dense))
    A = _csr_from_scipy(mx, scipy_A)

    M = preconditioners.ichol0(A)
    L_ref = _reference_ic0_lower_from_csr(scipy_A)

    np.testing.assert_allclose(to_numpy(M.L.todense()), L_ref, rtol=2e-6, atol=2e-6)
    assert M.nnz_L == int(np.count_nonzero(np.tril(dense)))


def test_ichol0_shift_is_explicit_and_can_recover_near_spd_input(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    weak = ms.csr_array(
        (
            mx.array([1.0e-7, 1.0], dtype=mx.float32),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )

    with pytest.raises(RuntimeError, match="pivot"):
        preconditioners.ichol0(weak)
    shifted = preconditioners.ichol0(weak, shift=1.0e-5)

    np.testing.assert_allclose(
        np.diag(to_numpy(shifted.L.todense())) ** 2,
        [1.01e-5, 1.00001],
        rtol=2e-5,
        atol=2e-8,
    )


def test_ichol0_failure_modes_are_explicit(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    missing_diag = ms.csr_array(
        (
            mx.array([1.0, 2.0], dtype=mx.float32),
            mx.array([1, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )
    indefinite = ms.csr_array(
        (
            mx.array([1.0, 2.0, 2.0, 1.0], dtype=mx.float32),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            mx.array([0, 2, 4], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )
    nonsymmetric = ms.csr_array(
        (
            mx.array([4.0, 2.0, 1.0, 3.0], dtype=mx.float32),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            mx.array([0, 2, 4], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )
    rectangular = ms.csr_array(
        (
            mx.array([1.0, 2.0], dtype=mx.float32),
            mx.array([0, 2], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 3),
        canonical=True,
    )
    complex_data = ms.csr_array(
        (
            mx.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=mx.complex64),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )

    with pytest.raises(RuntimeError, match="diagonal"):
        preconditioners.ichol0(missing_diag)
    with pytest.raises(RuntimeError, match="pivot"):
        preconditioners.ichol0(indefinite)
    with pytest.raises(RuntimeError, match="symmetric"):
        preconditioners.ichol0(nonsymmetric)
    with pytest.raises(ValueError, match="square"):
        preconditioners.ichol0(rectangular)
    with pytest.raises(TypeError, match="real float"):
        preconditioners.ichol0(complex_data)
    with pytest.raises(ValueError, match="finite"):
        preconditioners.ichol0(_spd_2x2(mx), shift=np.inf)
    with pytest.raises(ValueError, match="non-negative"):
        preconditioners.ichol0(_spd_2x2(mx), shift=-1.0)


def test_ichol0_setup_does_not_mutate_input_csr_metadata(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _poisson_2d_scipy(scipy_sparse, 3)
    A = _csr_from_scipy(mx, scipy_A)
    before = (
        to_numpy(A.data).copy(),
        to_numpy(A.indices).copy(),
        to_numpy(A.indptr).copy(),
        A.sorted_indices,
        A.has_canonical_format,
    )

    M = preconditioners.ichol0(A)

    assert M.shape == A.shape
    assert A.sorted_indices == before[3]
    assert A.has_canonical_format == before[4]
    np.testing.assert_array_equal(to_numpy(A.data), before[0])
    np.testing.assert_array_equal(to_numpy(A.indices), before[1])
    np.testing.assert_array_equal(to_numpy(A.indptr), before[2])


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


def _assert_chebyshev_apply_matches_dense_reference(mx, to_numpy, scipy_sparse):
    dense = np.array(
        [
            [5.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -0.5, 0.0],
            [0.0, -0.5, 3.5, -0.75],
            [0.0, 0.0, -0.75, 3.0],
        ],
        dtype=np.float32,
    )
    eigvals = np.linalg.eigvalsh(dense.astype(np.float64))
    scipy_A = scipy_sparse.csr_array(dense)
    A = _csr_from_scipy(mx, scipy_A)
    rhs = mx.array([1.0, -2.0, 0.5, 3.0], dtype=mx.float32)
    rhs_matrix_np = np.array(
        [[1.0, 0.5], [-2.0, 1.0], [0.5, -1.5], [3.0, 2.0]], dtype=np.float32
    )
    rhs_matrix = mx.array(rhs_matrix_np, dtype=mx.float32)

    M = preconditioners.chebyshev(
        A,
        degree=4,
        lambda_min=float(0.95 * eigvals[0]),
        lambda_max=float(1.05 * eigvals[-1]),
        estimate=False,
    )

    expected_vector = _reference_chebyshev_apply(
        dense,
        to_numpy(rhs),
        degree=M.degree,
        lambda_min=M.lambda_min,
        lambda_max=M.lambda_max,
    )
    expected_matrix = _reference_chebyshev_apply(
        dense,
        rhs_matrix_np,
        degree=M.degree,
        lambda_min=M.lambda_min,
        lambda_max=M.lambda_max,
    )
    np.testing.assert_allclose(to_numpy(M(rhs)), expected_vector, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        to_numpy(M(rhs_matrix)), expected_matrix, rtol=2e-6, atol=2e-6
    )


@pytest.mark.cpu_only
def test_chebyshev_apply_cpu_rank1_rank2_matches_dense_reference(
    mx, to_numpy, scipy_sparse
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    _assert_chebyshev_apply_matches_dense_reference(mx, to_numpy, scipy_sparse)


@pytest.mark.gpu
def test_chebyshev_apply_gpu_rank1_rank2_matches_dense_reference(
    mx, to_numpy, scipy_sparse
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    _assert_chebyshev_apply_matches_dense_reference(mx, to_numpy, scipy_sparse)


def test_chebyshev_uses_native_spectral_estimates_for_poisson(mx, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _poisson_2d_scipy(scipy_sparse, 6)
    A = _csr_from_scipy(mx, scipy_A)

    M = preconditioners.chebyshev(A, degree=2, estimate=True)

    assert M.lambda_min > 0.0
    assert M.lambda_max > M.lambda_min
    assert M.spectral_info["gershgorin_min"] == pytest.approx(0.0)
    assert M.spectral_info["lambda_min_source"] == "lanczos_0.5"
    assert M.spectral_info["lambda_max_source"] == "gershgorin"
    assert M.spectral_info["estimate_steps"] > 0


def test_chebyshev_rejects_invalid_or_unsafe_intervals(mx, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _poisson_2d_scipy(scipy_sparse, 4)
    A = _csr_from_scipy(mx, scipy_A)

    with pytest.raises(ValueError, match="lower spectral bound"):
        preconditioners.chebyshev(A, estimate=False)
    with pytest.raises(ValueError, match="0 < lambda_min < lambda_max"):
        preconditioners.chebyshev(A, lambda_min=2.0, lambda_max=1.0)
    with pytest.raises(ValueError, match="degree"):
        preconditioners.chebyshev(A, degree=0)


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


def test_gmres_ilu0_matches_scipy_spilu_quality_on_no_fill_system(
    mx, to_numpy, scipy_sparse
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    n = 10
    lower = -1.25 * np.ones(n - 1, dtype=np.float32)
    diag = 4.0 * np.ones(n, dtype=np.float32)
    upper = -0.5 * np.ones(n - 1, dtype=np.float32)
    scipy_A = scipy_sparse.diags(
        [lower, diag, upper], offsets=[-1, 0, 1], format="csr", dtype=np.float32
    )
    A = _csr_from_scipy(mx, scipy_A)
    b_np = np.linspace(1.0, -1.0, n, dtype=np.float32)
    M = preconditioners.ilu0(A)

    x_mlx, info_mlx = linalg.gmres(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=M,
        rtol=1e-6,
        atol=1e-7,
        restart=4,
        maxiter=40,
    )

    spilu = scipy_linalg.spilu(
        scipy_A.tocsc(),
        drop_tol=0.0,
        fill_factor=1.0,
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
    )
    M_scipy = scipy_linalg.LinearOperator(
        scipy_A.shape, matvec=spilu.solve, dtype=np.float32
    )
    x_scipy, info_scipy = _scipy_gmres(
        scipy_linalg,
        scipy_A,
        b_np,
        M=M_scipy,
        rtol=1e-6,
        atol=1e-7,
        restart=4,
        maxiter=40,
    )

    x_np = to_numpy(x_mlx)
    assert info_mlx == 0
    assert info_scipy == 0
    assert _relative_residual(scipy_A.toarray(), x_np, b_np) <= 2e-6
    np.testing.assert_allclose(x_np, x_scipy, rtol=3e-4, atol=3e-4)


def test_gmres_ilu0_uses_native_path_and_reduces_diagonal_dominant_iterations(
    mx, to_numpy, monkeypatch, scipy_sparse
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    import mlx_sparse.linalg._iterative as iterative_module

    n = 12
    scipy_A = scipy_sparse.diags(
        [
            -0.8 * np.ones(n - 1, dtype=np.float32),
            3.5 * np.ones(n, dtype=np.float32),
            -1.4 * np.ones(n - 1, dtype=np.float32),
        ],
        offsets=[-1, 0, 1],
        format="csr",
        dtype=np.float32,
    )
    A = _csr_from_scipy(mx, scipy_A)
    b = mx.array(np.sin(np.linspace(0.1, 1.3, n)).astype(np.float32))
    x0 = mx.zeros((n,), dtype=mx.float32)
    M = preconditioners.ilu0(A)

    def forbidden_host_fallback(*args, **kwargs):
        raise AssertionError("ILU0 GMRES used the Python host fallback")

    monkeypatch.setattr(iterative_module, "_left_pgmres_host", forbidden_host_fallback)
    _, base_info, _, base_iterations = _native.csr_gmres(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        A.shape,
        rtol=1e-6,
        atol=1e-7,
        restart=4,
        maxiter=48,
    )
    x, ilu_info, _, ilu_iterations = _native.csr_gmres_ilu0(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        M.L.data,
        M.L.indices,
        M.L.indptr,
        M.U.data,
        M.U.indices,
        M.U.indptr,
        A.shape,
        rtol=1e-6,
        atol=1e-7,
        restart=4,
        maxiter=48,
    )
    x_public, info_public = linalg.gmres(
        A, b, M=M, rtol=1e-6, atol=1e-7, restart=4, maxiter=48
    )

    base_iter = int(np.asarray(to_numpy(base_iterations)).item())
    ilu_iter = int(np.asarray(to_numpy(ilu_iterations)).item())
    assert int(np.asarray(to_numpy(base_info)).item()) == 0
    assert int(np.asarray(to_numpy(ilu_info)).item()) == 0
    assert info_public == 0
    assert ilu_iter < base_iter
    np.testing.assert_allclose(to_numpy(x_public), to_numpy(x), rtol=1e-6, atol=1e-6)
    assert _relative_residual(scipy_A.toarray(), to_numpy(x), to_numpy(b)) <= 2e-6


def test_gmres_ilu0_reduces_convection_diffusion_iterations(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    n = 16
    diffusion = 0.15
    convection = 1.4
    h = 1.0 / (n + 1)
    lower = (-diffusion / h**2 - convection / h) * np.ones(n - 1, dtype=np.float32)
    diag = (2.0 * diffusion / h**2 + convection / h + 1.0) * np.ones(
        n, dtype=np.float32
    )
    upper = (-diffusion / h**2) * np.ones(n - 1, dtype=np.float32)
    scipy_A = scipy_sparse.diags(
        [lower, diag, upper], offsets=[-1, 0, 1], format="csr", dtype=np.float32
    )
    A = _csr_from_scipy(mx, scipy_A)
    b = mx.array(np.cos(np.linspace(0.0, 2.0, n)).astype(np.float32))
    x0 = mx.zeros((n,), dtype=mx.float32)
    M = preconditioners.ilu0(A)

    _, base_info, _, base_iterations = _native.csr_gmres(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        A.shape,
        rtol=2e-6,
        atol=1e-7,
        restart=4,
        maxiter=64,
    )
    x, ilu_info, _, ilu_iterations = _native.csr_gmres_ilu0(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        M.L.data,
        M.L.indices,
        M.L.indptr,
        M.U.data,
        M.U.indices,
        M.U.indptr,
        A.shape,
        rtol=2e-6,
        atol=1e-7,
        restart=4,
        maxiter=64,
    )

    base_info_value = int(np.asarray(to_numpy(base_info)).item())
    base_iter_value = int(np.asarray(to_numpy(base_iterations)).item())
    ilu_iter_value = int(np.asarray(to_numpy(ilu_iterations)).item())
    assert int(np.asarray(to_numpy(ilu_info)).item()) == 0
    if base_info_value == 0:
        assert ilu_iter_value < base_iter_value
    else:
        assert ilu_iter_value < base_info_value
    assert _relative_residual(scipy_A.toarray(), to_numpy(x), to_numpy(b)) <= 5e-6


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


def test_cg_chebyshev_matches_dense_numpy_on_small_spd(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    dense = np.array(
        [
            [5.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -0.5, 0.0],
            [0.0, -0.5, 3.5, -0.75],
            [0.0, 0.0, -0.75, 3.0],
        ],
        dtype=np.float32,
    )
    eigvals = np.linalg.eigvalsh(dense.astype(np.float64))
    A = _csr_from_scipy(mx, scipy_sparse.csr_array(dense))
    b_np = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    M = preconditioners.chebyshev(
        A,
        degree=4,
        lambda_min=float(0.95 * eigvals[0]),
        lambda_max=float(1.05 * eigvals[-1]),
        estimate=False,
    )

    x, info = linalg.cg(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=M,
        rtol=1e-5,
        atol=1e-7,
        maxiter=32,
    )

    x_np = to_numpy(x)
    assert info == 0
    np.testing.assert_allclose(
        x_np,
        np.linalg.solve(dense.astype(np.float64), b_np.astype(np.float64)),
        rtol=5e-5,
        atol=5e-5,
    )
    assert _relative_residual(dense, x_np, b_np) <= 5e-5


def test_native_pcg_chebyshev_reduces_poisson_iterations_vs_jacobi(
    mx, to_numpy, scipy_sparse
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _poisson_2d_scipy(scipy_sparse, 8)
    A = _csr_from_scipy(mx, scipy_A)
    b = mx.ones((A.shape[0],), dtype=mx.float32)
    x0 = mx.zeros((A.shape[0],), dtype=mx.float32)
    jacobi = preconditioners.jacobi(A, check=True)
    chebyshev = preconditioners.chebyshev(A, degree=2)

    _, jacobi_info, _, jacobi_iterations = _native.csr_pcg_jacobi(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        jacobi.inverse_diagonal,
        A.shape,
        rtol=1e-4,
        atol=1e-7,
        maxiter=500,
    )
    x, cheb_info, residual, cheb_iterations = _native.csr_pcg_chebyshev(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        chebyshev.A.data,
        chebyshev.A.indices,
        chebyshev.A.indptr,
        A.shape,
        degree=chebyshev.degree,
        lambda_min=chebyshev.lambda_min,
        lambda_max=chebyshev.lambda_max,
        rtol=1e-4,
        atol=1e-7,
        maxiter=500,
    )

    assert int(np.asarray(to_numpy(jacobi_info)).item()) == 0
    assert int(np.asarray(to_numpy(cheb_info)).item()) == 0
    assert int(np.asarray(to_numpy(cheb_iterations)).item()) < int(
        np.asarray(to_numpy(jacobi_iterations)).item()
    )
    residual_np = float(np.asarray(to_numpy(residual)).item())
    assert residual_np == pytest.approx(
        np.linalg.norm(scipy_A @ to_numpy(x) - np.ones(A.shape[0], dtype=np.float32)),
        abs=2e-5,
    )
    assert residual_np / np.sqrt(A.shape[0]) <= 1.5e-4


@pytest.mark.gpu
def test_native_pcg_chebyshev_gpu_solves_poisson(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _poisson_2d_scipy(scipy_sparse, 6)
    A = _csr_from_scipy(mx, scipy_A)
    b = mx.ones((A.shape[0],), dtype=mx.float32)
    x0 = mx.zeros((A.shape[0],), dtype=mx.float32)
    M = preconditioners.chebyshev(A, degree=2)

    x, info, residual, iterations = _native.csr_pcg_chebyshev(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        M.A.data,
        M.A.indices,
        M.A.indptr,
        A.shape,
        degree=M.degree,
        lambda_min=M.lambda_min,
        lambda_max=M.lambda_max,
        rtol=1e-4,
        atol=1e-7,
        maxiter=256,
    )

    assert int(np.asarray(to_numpy(info)).item()) == 0
    assert int(np.asarray(to_numpy(iterations)).item()) > 0
    residual_np = float(np.asarray(to_numpy(residual)).item())
    assert residual_np == pytest.approx(
        np.linalg.norm(scipy_A @ to_numpy(x) - np.ones(A.shape[0], dtype=np.float32)),
        abs=2e-5,
    )
    assert residual_np / np.sqrt(A.shape[0]) <= 1.5e-4


def test_cg_ichol0_matches_dense_numpy_on_small_spd(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    dense = np.array(
        [
            [6.0, -1.0, 0.0, 0.0],
            [-1.0, 5.0, -1.0, 0.0],
            [0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, -1.0, 3.5],
        ],
        dtype=np.float32,
    )
    A = _csr_from_scipy(mx, scipy_sparse.csr_array(dense))
    b_np = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)

    x, info = linalg.cg(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=preconditioners.ichol0(A),
        rtol=1e-5,
        atol=1e-7,
        maxiter=32,
    )

    x_np = to_numpy(x)
    assert info == 0
    np.testing.assert_allclose(
        x_np,
        np.linalg.solve(dense.astype(np.float64), b_np.astype(np.float64)),
        rtol=2e-5,
        atol=2e-5,
    )
    assert _relative_residual(dense, x_np, b_np) <= 2e-5


def test_native_pcg_ichol0_reduces_poisson_iterations_vs_jacobi(
    mx, to_numpy, scipy_sparse
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _poisson_2d_scipy(scipy_sparse, 8)
    A = _csr_from_scipy(mx, scipy_A)
    b = mx.ones((A.shape[0],), dtype=mx.float32)
    x0 = mx.zeros((A.shape[0],), dtype=mx.float32)
    jacobi = preconditioners.jacobi(A, check=True)
    ichol0 = preconditioners.ichol0(A)
    upper = ichol0._upper()

    _, jacobi_info, _, jacobi_iterations = _native.csr_pcg_jacobi(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        jacobi.inverse_diagonal,
        A.shape,
        rtol=1e-4,
        atol=1e-7,
        maxiter=500,
    )
    x, ic0_info, residual, ic0_iterations = _native.csr_pcg_ic0(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        ichol0.L.data,
        ichol0.L.indices,
        ichol0.L.indptr,
        upper.data,
        upper.indices,
        upper.indptr,
        A.shape,
        rtol=1e-4,
        atol=1e-7,
        maxiter=500,
    )

    assert int(np.asarray(to_numpy(jacobi_info)).item()) == 0
    assert int(np.asarray(to_numpy(ic0_info)).item()) == 0
    assert int(np.asarray(to_numpy(ic0_iterations)).item()) < int(
        np.asarray(to_numpy(jacobi_iterations)).item()
    )
    residual_np = float(np.asarray(to_numpy(residual)).item())
    assert residual_np == pytest.approx(
        np.linalg.norm(scipy_A @ to_numpy(x) - np.ones(A.shape[0], dtype=np.float32)),
        abs=2e-5,
    )
    assert residual_np / np.sqrt(A.shape[0]) <= 1.5e-4


def test_cg_ichol0_handles_anisotropic_diffusion(mx, to_numpy, scipy_sparse):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_A = _anisotropic_diffusion_2d_scipy(scipy_sparse, 5)
    A = _csr_from_scipy(mx, scipy_A)
    x_true = np.sin(np.linspace(0.2, 1.3, A.shape[0], dtype=np.float32))
    b_np = np.asarray(scipy_A @ x_true, dtype=np.float32)

    x, info = linalg.cg(
        A,
        mx.array(b_np, dtype=mx.float32),
        M=preconditioners.ichol0(A),
        rtol=2e-4,
        atol=1e-7,
        maxiter=128,
    )

    x_np = to_numpy(x)
    assert info == 0
    assert _relative_residual(scipy_A.toarray(), x_np, b_np) <= 2.5e-4
    np.testing.assert_allclose(x_np, x_true, rtol=2e-3, atol=2e-3)


def test_native_pcg_ichol0_scaled_diagonal_converges_in_one_iteration(
    mx, to_numpy, scipy_sparse
):
    if not extension_available():
        pytest.skip("native extension unavailable")
    diag = np.geomspace(1.0e-6, 1.0e3, 16).astype(np.float32)
    scipy_A = scipy_sparse.diags(diag, 0, format="csr", dtype=np.float32)
    A = _csr_from_scipy(mx, scipy_A)
    x_true = np.linspace(-1.0, 1.0, A.shape[0], dtype=np.float32)
    b = mx.array(diag * x_true, dtype=mx.float32)
    x0 = mx.zeros((A.shape[0],), dtype=mx.float32)
    M = preconditioners.ichol0(A)
    upper = M._upper()

    x, info, residual, iterations = _native.csr_pcg_ic0(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        M.L.data,
        M.L.indices,
        M.L.indptr,
        upper.data,
        upper.indices,
        upper.indptr,
        A.shape,
        rtol=1e-6,
        atol=1e-8,
        maxiter=16,
    )

    assert int(np.asarray(to_numpy(info)).item()) == 0
    assert int(np.asarray(to_numpy(iterations)).item()) == 1
    assert float(np.asarray(to_numpy(residual)).item()) <= 2e-4
    np.testing.assert_allclose(to_numpy(x), x_true, rtol=2e-4, atol=2e-4)


def test_cg_ichol0_shifted_near_singular_spd_stays_finite(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    diag = np.array([1.0e-7, 1.0, 2.0, 3.0, 5.0], dtype=np.float32)
    A = ms.csr_array(
        (
            mx.array(diag, dtype=mx.float32),
            mx.array(np.arange(diag.size, dtype=np.int32), dtype=mx.int32),
            mx.array(np.arange(diag.size + 1, dtype=np.int32), dtype=mx.int32),
        ),
        shape=(diag.size, diag.size),
        canonical=True,
    )
    x_true = np.linspace(0.5, 1.5, diag.size, dtype=np.float32)
    b = mx.array((diag + 1.0e-5) * x_true, dtype=mx.float32)

    M = preconditioners.ichol0(A, shift=1.0e-5)
    shifted_A = ms.csr_array(
        (
            mx.array(diag + 1.0e-5, dtype=mx.float32),
            mx.array(np.arange(diag.size, dtype=np.int32), dtype=mx.int32),
            mx.array(np.arange(diag.size + 1, dtype=np.int32), dtype=mx.int32),
        ),
        shape=A.shape,
        canonical=True,
    )
    x, info = linalg.cg(
        shifted_A,
        b,
        M=M,
        rtol=1e-6,
        atol=1e-8,
        maxiter=16,
    )

    assert info == 0
    assert np.all(np.isfinite(to_numpy(x)))
    np.testing.assert_allclose(to_numpy(x), x_true, rtol=2e-4, atol=2e-4)


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
    solve_rhs = mx.array([1.0, -2.0, 0.5], dtype=mx.float32)
    solve_rhs_np = to_numpy(solve_rhs)
    expected_solution = np.linalg.solve(dense.astype(np.float64), solve_rhs_np)

    for sparse in (coo.tocsr(), coo, coo.tocsc()):
        M = preconditioners.jacobi(sparse, check=True)
        assert M.is_positive_definite is True
        got = M(mx.array(rhs_np, dtype=mx.float32))
        np.testing.assert_allclose(to_numpy(got), expected, rtol=1e-6, atol=1e-6)
        solution, info = linalg.gmres(
            sparse,
            solve_rhs,
            M=M,
            rtol=1e-6,
            atol=1e-7,
            restart=3,
            maxiter=24,
        )
        solution_np = to_numpy(solution)
        assert info == 0
        np.testing.assert_allclose(solution_np, expected_solution, rtol=2e-5, atol=2e-5)
        assert _relative_residual(dense, solution_np, solve_rhs_np) <= 2e-6
