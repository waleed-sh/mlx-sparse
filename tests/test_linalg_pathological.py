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


def _csr_from_dense(mx, dense, *, canonical=True):
    dense = np.asarray(dense)
    row, col = np.nonzero(dense)
    data = dense[row, col]
    indptr = np.zeros(dense.shape[0] + 1, dtype=np.int32)
    np.add.at(indptr, row + 1, 1)
    np.cumsum(indptr, out=indptr)
    return ms.csr_array(
        (
            mx.array(data.astype(dense.dtype, copy=False)),
            mx.array(col.astype(np.int32, copy=False)),
            mx.array(indptr),
        ),
        shape=dense.shape,
        sorted_indices=True,
        canonical=canonical,
        validate="full",
    )


def _noncanonical_spd(mx):
    dense = np.array(
        [
            [6.0, 2.0, 0.0, 0.0],
            [2.0, 5.0, -1.0, 0.0],
            [0.0, -1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    data = np.array(
        [
            3.0,
            6.0,
            -1.0,
            -1.0,
            2.0,
            5.0,
            1.0,
            -1.0,
            4.0,
            0.25,
            3.0,
            0.75,
        ],
        dtype=np.float32,
    )
    indices = np.array([1, 0, 1, 2, 0, 1, 3, 1, 2, 2, 3, 2], dtype=np.int32)
    indptr = np.array([0, 3, 6, 9, 12], dtype=np.int32)
    csr = ms.csr_array(
        (mx.array(data), mx.array(indices), mx.array(indptr)),
        shape=dense.shape,
        sorted_indices=False,
        canonical=False,
        validate="full",
    )
    np.testing.assert_allclose(np.array(csr.canonicalize().todense()), dense)
    return csr, dense


def _star_spd(n: int = 12):
    dense = np.eye(n, dtype=np.float32) * 6.0
    dense[0, 0] = 12.0
    for col in range(1, n):
        value = 0.05 * ((col % 5) + 1)
        dense[0, col] = value
        dense[col, 0] = value
    return dense


def _assert_solution_residual(dense, x, b, *, rtol=1e-4, atol=1e-4):
    residual = dense @ x - b
    scale = max(np.linalg.norm(b), 1.0)
    assert np.all(np.isfinite(x))
    assert np.linalg.norm(residual) <= atol + rtol * scale


def test_iterative_solvers_canonicalize_duplicate_unsorted_csr(mx, to_numpy):
    _require_native()
    csr, dense = _noncanonical_spd(mx)
    b_np = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    b = mx.array(b_np)
    expected = np.linalg.solve(dense, b_np)

    for solver in (linalg.cg, linalg.gmres, linalg.minres):
        x, info = solver(csr, b, rtol=1e-6, atol=1e-7, maxiter=80)
        assert info == 0, solver.__name__
        got = to_numpy(x)
        np.testing.assert_allclose(got, expected, rtol=5e-4, atol=5e-4)
        _assert_solution_residual(dense, got, b_np, rtol=1e-5, atol=1e-5)


def test_direct_solvers_canonicalize_duplicate_unsorted_csr(mx, to_numpy):
    _require_native()
    csr, dense = _noncanonical_spd(mx)
    b_np = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
    b = mx.array(b_np)
    expected = np.linalg.solve(dense, b_np)

    chol = linalg.sparse_cholesky(csr)
    lu = linalg.sparse_lu(csr)
    np.testing.assert_allclose(to_numpy(chol.solve(b)), expected, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(to_numpy(lu.solve(b)), expected, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(
        to_numpy(linalg.spsolve(csr, b)), expected, rtol=5e-4, atol=5e-4
    )


def test_spectral_solvers_canonicalize_duplicate_unsorted_csr(mx, to_numpy):
    _require_native()
    csr, dense = _noncanonical_spd(mx)

    got_eigsh = linalg.eigsh(csr, k=2, which="LM", ncv=4, return_eigenvectors=False)
    expected_eigsh = np.linalg.eigvalsh(dense)[-2:]
    np.testing.assert_allclose(
        np.sort(to_numpy(got_eigsh)),
        np.sort(expected_eigsh),
        rtol=5e-2,
        atol=5e-2,
    )

    got_svds = linalg.svds(csr, k=2, which="LM", ncv=4, return_singular_vectors=False)
    expected_svds = np.linalg.svd(dense, compute_uv=False)[:2]
    np.testing.assert_allclose(
        np.sort(to_numpy(got_svds)),
        np.sort(expected_svds),
        rtol=5e-2,
        atol=5e-2,
    )


def test_solvers_handle_highly_imbalanced_csr_row_lengths(mx, to_numpy):
    _require_native()
    dense = _star_spd()
    csr = _csr_from_dense(mx, dense)
    row_lengths = np.diff(to_numpy(csr.indptr))
    assert row_lengths.max() >= 6 * row_lengths[1:].max()

    b_np = np.linspace(-1.0, 1.0, dense.shape[0], dtype=np.float32)
    b = mx.array(b_np)
    expected = np.linalg.solve(dense, b_np)

    for solver in (linalg.cg, linalg.gmres, linalg.minres):
        x, info = solver(csr, b, rtol=1e-6, atol=1e-7, maxiter=128)
        assert info == 0, solver.__name__
        got = to_numpy(x)
        np.testing.assert_allclose(got, expected, rtol=5e-4, atol=5e-4)
        _assert_solution_residual(dense, got, b_np, rtol=1e-5, atol=1e-5)

    np.testing.assert_allclose(
        to_numpy(linalg.spsolve(csr, b)),
        expected,
        rtol=5e-4,
        atol=5e-4,
    )


def test_direct_solvers_reject_empty_row_singular_matrix(mx):
    _require_native()
    data = mx.array([2.0, 3.0, 4.0], dtype=mx.float32)
    indices = mx.array([0, 2, 3], dtype=mx.int32)
    indptr = mx.array([0, 1, 1, 2, 3], dtype=mx.int32)
    csr = ms.csr_array(
        (data, indices, indptr),
        shape=(4, 4),
        sorted_indices=True,
        canonical=True,
        validate="full",
    )
    b = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float32)

    with pytest.raises(RuntimeError):
        linalg.sparse_cholesky(csr)
    with pytest.raises(RuntimeError):
        linalg.sparse_lu(csr)
    with pytest.raises(RuntimeError):
        linalg.spsolve(csr, b)


def test_iterative_solvers_report_nonconvergence_on_inconsistent_empty_row(
    mx, to_numpy
):
    _require_native()
    data = mx.array([2.0, 3.0, 4.0], dtype=mx.float32)
    indices = mx.array([0, 2, 3], dtype=mx.int32)
    indptr = mx.array([0, 1, 1, 2, 3], dtype=mx.int32)
    csr = ms.csr_array(
        (data, indices, indptr),
        shape=(4, 4),
        sorted_indices=True,
        canonical=True,
        validate="full",
    )
    dense = to_numpy(csr.todense())
    b_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    b = mx.array(b_np)

    for solver in (linalg.cg, linalg.gmres, linalg.minres):
        x, info = solver(csr, b, rtol=1e-7, atol=0.0, maxiter=12)
        got = to_numpy(x)
        assert info != 0, solver.__name__
        assert np.all(np.isfinite(got)), solver.__name__
        assert np.linalg.norm(dense @ got - b_np) > 1e-2, solver.__name__


@pytest.mark.parametrize(
    "solver",
    [
        linalg.cg,
        linalg.gmres,
        pytest.param(
            linalg.minres,
            marks=pytest.mark.xfail(
                reason=(
                    "MINRES does not converge to the requested tolerance on "
                    "this near-singular diagonal system."
                ),
                strict=True,
            ),
        ),
    ],
)
def test_near_singular_diagonal_iterative_solver_converges(solver, mx, to_numpy):
    _require_native()
    dense = np.diag(np.array([1e-5, 1.0, 2.0, 3.0], dtype=np.float32))
    csr = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    b = mx.array(b_np)

    x, info = solver(csr, b, rtol=1e-6, atol=1e-7, maxiter=64)
    got = to_numpy(x)
    assert info == 0, solver.__name__
    _assert_solution_residual(dense, got, b_np, rtol=1e-5, atol=1e-5)


def test_jacobi_pcg_near_singular_diagonal_converges(mx, to_numpy):
    _require_native()
    dense = np.diag(np.array([1e-8, 1.0, 2.0, 3.0], dtype=np.float32))
    csr = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    b = mx.array(b_np)

    x, info = linalg.cg(
        csr,
        b,
        M=preconditioners.jacobi(csr, check=True),
        rtol=1e-6,
        atol=1e-7,
        maxiter=8,
    )

    got = to_numpy(x)
    assert info == 0
    assert np.all(np.isfinite(got))
    _assert_solution_residual(dense, got, b_np, rtol=1e-5, atol=1e-5)


def test_jacobi_pcg_singular_incompatible_system_reports_failure_finitely(mx, to_numpy):
    _require_native()
    dense = np.diag(np.array([0.0, 1.0], dtype=np.float32))
    csr = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, 1.0], dtype=np.float32)
    b = mx.array(b_np)

    x, info = linalg.cg(
        csr,
        b,
        M=preconditioners.jacobi(csr, zero_policy="unit"),
        rtol=1e-7,
        atol=0.0,
        maxiter=8,
    )

    got = to_numpy(x)
    assert info != 0
    assert np.all(np.isfinite(got))
    assert np.linalg.norm(dense @ got - b_np) > 1e-2


def test_near_singular_diagonal_direct_solver_residual_is_small(mx, to_numpy):
    _require_native()
    dense = np.diag(np.array([1e-5, 1.0, 2.0, 3.0], dtype=np.float32))
    csr = _csr_from_dense(mx, dense)
    b_np = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    b = mx.array(b_np)

    _assert_solution_residual(
        dense,
        to_numpy(linalg.spsolve(csr, b)),
        b_np,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.xfail(
    reason=(
        "GMRES does not converge to the requested tolerance on this "
        "ill-conditioned Hilbert-like system, although its residual remains "
        "bounded."
    ),
    strict=True,
)
def test_ill_conditioned_hilbert_like_gmres_converges(mx, to_numpy):
    _require_native()
    n = 5
    i = np.arange(n, dtype=np.float32)
    dense = (1.0 / (i[:, None] + i[None, :] + 1.0)).astype(np.float32)
    csr = _csr_from_dense(mx, dense)
    b_np = np.linspace(0.25, 1.25, n, dtype=np.float32)
    b = mx.array(b_np)

    x_gmres, info = linalg.gmres(csr, b, rtol=1e-6, atol=1e-7, restart=5, maxiter=64)
    assert info == 0
    _assert_solution_residual(dense, to_numpy(x_gmres), b_np, rtol=1e-5, atol=1e-5)


def test_ill_conditioned_hilbert_like_direct_solver_residual_is_small(mx, to_numpy):
    _require_native()
    n = 5
    i = np.arange(n, dtype=np.float32)
    dense = (1.0 / (i[:, None] + i[None, :] + 1.0)).astype(np.float32)
    csr = _csr_from_dense(mx, dense)
    b_np = np.linspace(0.25, 1.25, n, dtype=np.float32)
    b = mx.array(b_np)

    x_direct = linalg.spsolve(csr, b)
    _assert_solution_residual(dense, to_numpy(x_direct), b_np, rtol=5e-3, atol=5e-3)


def test_singular_matrix_direct_solvers_raise(mx):
    _require_native()
    dense = np.array(
        [
            [2.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    csr = _csr_from_dense(mx, dense)
    b = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)

    with pytest.raises(RuntimeError):
        linalg.sparse_lu(csr)
    with pytest.raises(RuntimeError):
        linalg.spsolve(csr, b)


def test_complex_hermitian_solver_inputs_are_explicitly_unsupported(mx):
    _require_native()
    dense = np.array(
        [
            [4.0 + 0.0j, 1.0 - 2.0j, 0.0 + 0.0j],
            [1.0 + 2.0j, 5.0 + 0.0j, -0.5 + 0.25j],
            [0.0 + 0.0j, -0.5 - 0.25j, 3.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    csr = _csr_from_dense(mx, dense)
    b = mx.array(np.array([1.0 + 0.5j, -2.0 + 0.0j, 0.25 - 1.0j], dtype=np.complex64))

    np.testing.assert_allclose(dense, dense.conj().T)
    for solver in (linalg.cg, linalg.gmres, linalg.minres):
        with pytest.raises(TypeError, match="real float data"):
            solver(csr, b)
    for routine in (
        linalg.eigsh,
        linalg.eigs,
        linalg.svds,
        linalg.sparse_cholesky,
        linalg.sparse_lu,
    ):
        with pytest.raises(TypeError, match="real float data"):
            routine(csr)
