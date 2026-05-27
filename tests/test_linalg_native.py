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

pytestmark = pytest.mark.native


def _spd(mx):
    return ms.csr_array(
        (
            mx.array([4.0, 1.0, 1.0, 3.0], dtype=mx.float32),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            mx.array([0, 2, 4], dtype=mx.int32),
        ),
        shape=(2, 2),
        validate="full",
        canonical=True,
    )


def _nonsymmetric(mx):
    return ms.csr_array(
        (
            mx.array([2.0, 1.0, 3.0, -1.0, 4.0], dtype=mx.float32),
            mx.array([0, 1, 1, 0, 2], dtype=mx.int32),
            mx.array([0, 2, 3, 5], dtype=mx.int32),
        ),
        shape=(3, 3),
        validate="full",
        canonical=True,
    )


def test_sparse_dot_and_vdot_match_numpy(mx, to_numpy):
    pytest.importorskip("mlx.core")
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = _spd(mx)
    b = ms.csr_array(
        (
            mx.array([1.0, 2.0, 5.0], dtype=mx.float32),
            mx.array([0, 1, 1], dtype=mx.int32),
            mx.array([0, 2, 3], dtype=mx.int32),
        ),
        shape=(2, 2),
        validate="full",
        canonical=True,
    )
    expected = float(np.sum(to_numpy(a.todense()) * to_numpy(b.todense())))
    np.testing.assert_allclose(to_numpy(a.vdot(b)), expected, rtol=1e-6)
    np.testing.assert_allclose(to_numpy(linalg.dot(a, b)), expected, rtol=1e-6)


def test_sparse_complex_dot_and_vdot_match_numpy_semantics(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = ms.csr_array(
        (
            mx.array(np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        validate="full",
        canonical=True,
    )
    b = ms.csr_array(
        (
            mx.array(np.array([4.0 - 1.0j, 2.0 + 3.0j], dtype=np.complex64)),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        validate="full",
        canonical=True,
    )

    dense_a = to_numpy(a.todense())
    dense_b = to_numpy(b.todense())
    np.testing.assert_allclose(to_numpy(a.vdot(b)), np.vdot(dense_a, dense_b))
    np.testing.assert_allclose(to_numpy(linalg.vdot(a, b)), np.vdot(dense_a, dense_b))
    np.testing.assert_allclose(to_numpy(a.dot(b)), np.sum(dense_a * dense_b))
    np.testing.assert_allclose(to_numpy(linalg.dot(a, b)), np.sum(dense_a * dense_b))


def test_cg_gmres_minres_match_numpy_solve(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = _spd(mx)
    b = mx.array([1.0, 2.0], dtype=mx.float32)
    expected = np.linalg.solve(to_numpy(a.todense()), to_numpy(b))
    for solver in (linalg.cg, linalg.gmres, linalg.minres):
        x, info = solver(a, b, rtol=1e-6, atol=1e-7, maxiter=20)
        assert info == 0
        np.testing.assert_allclose(to_numpy(x), expected, rtol=5e-4, atol=5e-4)


def test_csc_iterative_solvers_match_numpy_solve(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = _spd(mx).tocsc(canonical=True)
    b = mx.array([1.0, 2.0], dtype=mx.float32)
    expected = np.linalg.solve(to_numpy(a.todense()), to_numpy(b))
    for solver in (linalg.cg, linalg.gmres, linalg.minres):
        x, info = solver(a, b, rtol=1e-6, atol=1e-7, maxiter=20)
        assert info == 0
        np.testing.assert_allclose(to_numpy(x), expected, rtol=5e-4, atol=5e-4)


def test_iterative_solvers_match_scipy_sparse(mx, scipy_sparse, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    scipy_a = scipy_sparse.diags(
        [
            -np.ones(4, dtype=np.float32),
            4.0 * np.ones(5, dtype=np.float32),
            -np.ones(4, dtype=np.float32),
        ],
        offsets=[-1, 0, 1],
        format="csr",
    )
    a = ms.from_scipy(scipy_a)
    b_np = np.arange(1, 6, dtype=np.float32)
    b = mx.array(b_np)
    expected, scipy_info = scipy_linalg.cg(scipy_a, b_np, rtol=1e-7, maxiter=64)
    assert scipy_info == 0
    x, info = linalg.cg(a, b, rtol=1e-6, maxiter=64)
    assert info == 0
    np.testing.assert_allclose(to_numpy(x), expected, rtol=1e-4, atol=1e-4)


def test_sparse_cholesky_factor_and_solve(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = _spd(mx)
    factor = linalg.sparse_cholesky(a)
    dense_l = to_numpy(factor.L.todense())
    np.testing.assert_allclose(
        dense_l @ dense_l.T, to_numpy(a.todense()), rtol=5e-5, atol=5e-5
    )
    b = mx.array([1.0, 2.0], dtype=mx.float32)
    np.testing.assert_allclose(
        to_numpy(factor.solve(b)),
        np.linalg.solve(to_numpy(a.todense()), to_numpy(b)),
        rtol=5e-4,
        atol=5e-4,
    )


def test_csc_sparse_cholesky_and_lu_solve(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    spd = _spd(mx).tocsc(canonical=True)
    b_spd = mx.array([1.0, 2.0], dtype=mx.float32)
    chol = linalg.sparse_cholesky(spd)
    np.testing.assert_allclose(
        to_numpy(chol.solve(b_spd)),
        np.linalg.solve(to_numpy(spd.todense()), to_numpy(b_spd)),
        rtol=5e-4,
        atol=5e-4,
    )

    general = _nonsymmetric(mx).tocsc(canonical=True)
    b = mx.array([1.0, -2.0, 0.5], dtype=mx.float32)
    np.testing.assert_allclose(
        to_numpy(linalg.spsolve(general, b)),
        np.linalg.solve(to_numpy(general.todense()), to_numpy(b)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_sparse_lu_factor_and_solve(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = _nonsymmetric(mx)
    factor = linalg.sparse_lu(a)
    dense_a = to_numpy(a.todense())
    dense_l = to_numpy(factor.L.todense())
    dense_u = to_numpy(factor.U.todense())
    perm = to_numpy(factor.perm).astype(np.int64)
    np.testing.assert_allclose(
        dense_a[perm, :], dense_l @ dense_u, rtol=1e-5, atol=1e-5
    )
    b = mx.array([1.0, -2.0, 0.5], dtype=mx.float32)
    np.testing.assert_allclose(
        to_numpy(factor.solve(b)),
        np.linalg.solve(dense_a, to_numpy(b)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_factorized_lu_solve_reuses_factor_and_accepts_matrix_rhs(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = _nonsymmetric(mx)
    dense_a = to_numpy(a.todense())
    rhs_np = np.array([[1.0, 0.25], [-2.0, 1.5], [0.5, -0.75]], dtype=np.float32)
    solver = linalg.factorized(a, method="lu")

    assert solver.shape == a.shape
    assert solver.method == "lu"
    assert solver.backend in {"native", "accelerate"}
    assert solver.rhs_size == 3
    assert solver.solution_size == 3
    np.testing.assert_allclose(
        to_numpy(solver.solve(mx.array(rhs_np))),
        np.linalg.solve(dense_a, rhs_np),
        rtol=1e-4,
        atol=1e-4,
    )


def test_factorized_cholesky_matches_sparse_cholesky(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = _spd(mx)
    b = mx.array([1.0, 2.0], dtype=mx.float32)
    solver = linalg.factorized(a, method="cholesky")

    np.testing.assert_allclose(
        to_numpy(solver(b)),
        to_numpy(linalg.sparse_cholesky(a).solve(b)),
        rtol=5e-4,
        atol=5e-4,
    )


def test_spsolve_rejects_rectangular_sparse_matrix(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    rectangular = ms.csr_array(
        (
            mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
            mx.array([0, 1, 1], dtype=mx.int32),
            mx.array([0, 2, 3], dtype=mx.int32),
        ),
        shape=(2, 3),
        validate="full",
    )
    with pytest.raises(ValueError, match="square"):
        linalg.spsolve(rectangular, mx.ones((2,), dtype=mx.float32))


def test_eigsh_eigs_svds_match_dense_references(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = _spd(mx)
    vals, vecs = linalg.eigsh(a, k=1, which="LM", ncv=2)
    dense = to_numpy(a.todense())
    expected_vals, _ = np.linalg.eigh(dense)
    np.testing.assert_allclose(
        np.sort(to_numpy(vals)), np.sort(expected_vals)[-1:], rtol=5e-3, atol=5e-3
    )
    residual = dense @ to_numpy(vecs)[:, 0] - to_numpy(vals)[0] * to_numpy(vecs)[:, 0]
    assert np.linalg.norm(residual) < 5e-2

    general = _nonsymmetric(mx)
    evals = linalg.eigs(general, k=1, which="LM", ncv=3, return_eigenvectors=False)
    expected_general = np.linalg.eigvals(to_numpy(general.todense()))
    assert np.min(np.abs(expected_general - to_numpy(evals)[0])) < 2e-1

    u, s, vh = linalg.svds(general, k=1, which="LM", ncv=3)
    expected_s = np.linalg.svd(to_numpy(general.todense()), compute_uv=False)[0]
    np.testing.assert_allclose(to_numpy(s)[0], expected_s, rtol=5e-2, atol=5e-2)
    assert to_numpy(u).shape == (3, 1)
    assert to_numpy(vh).shape == (1, 3)


def test_csc_spectral_routines_match_dense_references(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    a = _spd(mx).tocsc(canonical=True)
    vals = linalg.eigsh(a, k=1, which="LM", ncv=2, return_eigenvectors=False)
    expected_vals, _ = np.linalg.eigh(to_numpy(a.todense()))
    np.testing.assert_allclose(
        np.sort(to_numpy(vals)), np.sort(expected_vals)[-1:], rtol=5e-3, atol=5e-3
    )

    general = _nonsymmetric(mx).tocsc(canonical=True)
    evals = linalg.eigs(general, k=1, which="LM", ncv=3, return_eigenvectors=False)
    expected_general = np.linalg.eigvals(to_numpy(general.todense()))
    assert np.min(np.abs(expected_general - to_numpy(evals)[0])) < 2e-1

    s = linalg.svds(general, k=1, which="LM", ncv=3, return_singular_vectors=False)
    expected_s = np.linalg.svd(to_numpy(general.todense()), compute_uv=False)[0]
    np.testing.assert_allclose(to_numpy(s)[0], expected_s, rtol=5e-2, atol=5e-2)


def test_spectral_routines_match_scipy_sparse(mx, scipy_sparse, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    scipy_a = scipy_sparse.diags(
        [
            -np.ones(4, dtype=np.float32),
            2.0 * np.ones(5, dtype=np.float32),
            -np.ones(4, dtype=np.float32),
        ],
        offsets=[-1, 0, 1],
        format="csr",
    )
    a = ms.from_scipy(scipy_a)
    expected_eig, _ = scipy_linalg.eigsh(scipy_a, k=1, which="LM")
    got_eig = linalg.eigsh(a, k=1, which="LM", ncv=5, return_eigenvectors=False)
    np.testing.assert_allclose(
        np.sort(to_numpy(got_eig)), np.sort(expected_eig), rtol=5e-2, atol=5e-2
    )

    expected_s = scipy_linalg.svds(scipy_a, k=1, return_singular_vectors=False)[0]
    got_s = linalg.svds(a, k=1, which="LM", ncv=5, return_singular_vectors=False)
    np.testing.assert_allclose(to_numpy(got_s)[0], expected_s, rtol=5e-2, atol=5e-2)
