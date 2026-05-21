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

"""Error-path and branch coverage for linalg._iterative, _eigen, _factorizations,
_sparse_ops.  Tests that do not hit native kernels run unconditionally; those
that run actual solvers skip when the native extension is unavailable."""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse._ext_loader import extension_available

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spd_2x2(mx_module):
    return ms.csr_array(
        (
            mx_module.array([4.0, 1.0, 1.0, 3.0], dtype=mx_module.float32),
            mx_module.array([0, 1, 0, 1], dtype=mx_module.int32),
            mx_module.array([0, 2, 4], dtype=mx_module.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )


def _coo_2x2():
    return ms.coo_array(
        (
            mx.array([4.0, 1.0, 1.0, 3.0], dtype=mx.float32),
            (
                mx.array([0, 0, 1, 1], dtype=mx.int32),
                mx.array([0, 1, 0, 1], dtype=mx.int32),
            ),
        ),
        shape=(2, 2),
    )


def _float16_csr():
    """2×2 CSRArray with float16 data."""
    return ms.csr_array(
        (
            mx.array([4.0, 1.0, 1.0, 3.0], dtype=mx.float16),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            mx.array([0, 2, 4], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )


# ---------------------------------------------------------------------------
# _iterative._as_csr
# ---------------------------------------------------------------------------


class TestIterativeAsCsr:
    def test_csr_passthrough(self):
        from mlx_sparse.linalg._iterative import _as_csr

        csr = _spd_2x2(mx)
        result = _as_csr(csr)
        assert result.has_canonical_format

    def test_coo_converts(self):
        from mlx_sparse.linalg._iterative import _as_csr

        coo = _coo_2x2()
        result = _as_csr(coo)
        assert isinstance(result, ms.CSRArray)

    def test_dense_raises(self):
        from mlx_sparse.linalg._iterative import _as_csr

        with pytest.raises(TypeError, match="sparse iterative solvers"):
            _as_csr(mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32))


# ---------------------------------------------------------------------------
# _iterative._float32_csr
# ---------------------------------------------------------------------------


class TestIterativeFloat32Csr:
    def test_float32_passthrough(self):
        from mlx_sparse.linalg._iterative import _float32_csr

        csr = _spd_2x2(mx)
        result = _float32_csr(csr)
        assert result is csr

    def test_float16_promotes(self):
        from mlx_sparse.linalg._iterative import _float32_csr

        csr = _float16_csr()
        result = _float32_csr(csr)
        assert result.data.dtype == mx.float32

    def test_bfloat16_promotes(self):
        from mlx_sparse.linalg._iterative import _float32_csr

        csr = ms.csr_array(
            (
                mx.array([4.0, 1.0, 1.0, 3.0], dtype=mx.bfloat16),
                mx.array([0, 1, 0, 1], dtype=mx.int32),
                mx.array([0, 2, 4], dtype=mx.int32),
            ),
            shape=(2, 2),
            canonical=True,
        )
        result = _float32_csr(csr)
        assert result.data.dtype == mx.float32

    def test_complex64_raises(self):
        from mlx_sparse.linalg._iterative import _float32_csr

        csr = ms.csr_array(
            (
                mx.array(np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex64)),
                mx.array([0, 1], dtype=mx.int32),
                mx.array([0, 1, 2], dtype=mx.int32),
            ),
            shape=(2, 2),
            canonical=True,
        )
        with pytest.raises(TypeError, match="real float"):
            _float32_csr(csr)


# ---------------------------------------------------------------------------
# _iterative._float32_array
# ---------------------------------------------------------------------------


class TestIterativeFloat32Array:
    def test_float16_promotes(self):
        from mlx_sparse.linalg._iterative import _float32_array

        x = mx.array([1.0, 2.0], dtype=mx.float16)
        result = _float32_array(x)
        assert result.dtype == mx.float32

    def test_bfloat16_promotes(self):
        from mlx_sparse.linalg._iterative import _float32_array

        x = mx.array([1.0, 2.0], dtype=mx.bfloat16)
        result = _float32_array(x)
        assert result.dtype == mx.float32

    def test_complex64_raises(self):
        from mlx_sparse.linalg._iterative import _float32_array

        x = mx.array(np.array([1.0 + 1.0j], dtype=np.complex64))
        with pytest.raises(TypeError, match="real float"):
            _float32_array(x)


# ---------------------------------------------------------------------------
# _iterative._guess
# ---------------------------------------------------------------------------


class TestIterativeGuess:
    def test_b_not_rank1_raises(self):
        from mlx_sparse.linalg._iterative import _guess

        csr = _spd_2x2(mx)
        b = mx.array([[1.0, 2.0]], dtype=mx.float32)
        with pytest.raises(ValueError, match="rank-1"):
            _guess(csr, b, None)

    def test_b_wrong_length_raises(self):
        from mlx_sparse.linalg._iterative import _guess

        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)  # length 3, expected 2
        with pytest.raises(ValueError, match="length"):
            _guess(csr, b, None)

    def test_x0_none_returns_zeros(self):
        from mlx_sparse.linalg._iterative import _guess

        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x0 = _guess(csr, b, None)
        mx.eval(x0)
        np.testing.assert_allclose(np.array(x0), [0.0, 0.0])

    def test_x0_valid_is_returned(self):
        from mlx_sparse.linalg._iterative import _guess

        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x0_in = mx.array([0.5, 0.5], dtype=mx.float32)
        x0_out = _guess(csr, b, x0_in)
        mx.eval(x0_out)
        np.testing.assert_allclose(np.array(x0_out), [0.5, 0.5])

    def test_x0_wrong_shape_raises(self):
        from mlx_sparse.linalg._iterative import _guess

        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x0 = mx.array([0.5, 0.5, 0.5], dtype=mx.float32)  # length 3, expected 2
        with pytest.raises(ValueError, match="shape"):
            _guess(csr, b, x0)

    def test_x0_rank2_raises(self):
        from mlx_sparse.linalg._iterative import _guess

        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x0 = mx.array([[0.5, 0.5]], dtype=mx.float32)
        with pytest.raises(ValueError, match="shape"):
            _guess(csr, b, x0)


# ---------------------------------------------------------------------------
# _iterative._maxiter
# ---------------------------------------------------------------------------


class TestIterativeMaxiter:
    def test_none_returns_10n(self):
        from mlx_sparse.linalg._iterative import _maxiter

        csr = _spd_2x2(mx)
        assert _maxiter(csr, None) == 10 * csr.shape[1]

    def test_explicit_value(self):
        from mlx_sparse.linalg._iterative import _maxiter

        csr = _spd_2x2(mx)
        assert _maxiter(csr, 50) == 50

    def test_negative_raises(self):
        from mlx_sparse.linalg._iterative import _maxiter

        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="non-negative"):
            _maxiter(csr, -1)


# ---------------------------------------------------------------------------
# Public solver API — error paths that don't touch native kernels
# ---------------------------------------------------------------------------


class TestSolverAPIErrors:
    def test_cg_with_M_raises(self):
        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="callbacks"):
            linalg.cg(csr, b, M=lambda x: x)

    def test_cg_with_callback_raises(self):
        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="callbacks"):
            linalg.cg(csr, b, callback=lambda x: None)

    def test_gmres_with_M_raises(self):
        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="callbacks"):
            linalg.gmres(csr, b, M=lambda x: x)

    def test_gmres_with_callback_raises(self):
        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="callbacks"):
            linalg.gmres(csr, b, callback=lambda x: None)

    def test_gmres_bad_callback_type_raises(self):
        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        with pytest.raises(ValueError, match="callback_type"):
            linalg.gmres(csr, b, callback_type="invalid")

    def test_gmres_bad_restart_raises(self):
        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        with pytest.raises(ValueError, match="restart"):
            linalg.gmres(csr, b, restart=0)

    def test_minres_with_callback_raises(self):
        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="callbacks"):
            linalg.minres(csr, b, callback=lambda x: None)

    def test_cg_accepts_coo_input(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        coo = _coo_2x2()
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x, info = linalg.cg(coo, b, rtol=1e-6, maxiter=50)
        assert info == 0

    def test_cg_with_x0(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x0 = mx.array([0.1, 0.1], dtype=mx.float32)
        x, info = linalg.cg(csr, b, x0=x0, rtol=1e-6, maxiter=50)
        assert info == 0

    def test_cg_with_float16_matrix_promoted(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _float16_csr()
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x, info = linalg.cg(csr, b, rtol=1e-5, maxiter=50)
        assert info == 0


# ---------------------------------------------------------------------------
# _eigen._as_csr and _float32_csr
# ---------------------------------------------------------------------------


class TestEigenAsCsr:
    def test_csr_passthrough(self):
        from mlx_sparse.linalg._eigen import _as_csr

        csr = _spd_2x2(mx)
        result = _as_csr(csr)
        assert result.has_canonical_format

    def test_coo_converts(self):
        from mlx_sparse.linalg._eigen import _as_csr

        coo = _coo_2x2()
        result = _as_csr(coo)
        assert isinstance(result, ms.CSRArray)

    def test_dense_raises(self):
        from mlx_sparse.linalg._eigen import _as_csr

        with pytest.raises(TypeError, match="sparse eigen"):
            _as_csr(mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32))


class TestEigenFloat32Csr:
    def test_float32_passthrough(self):
        from mlx_sparse.linalg._eigen import _float32_csr

        csr = _spd_2x2(mx)
        result = _float32_csr(csr)
        assert result is csr

    def test_float16_promotes(self):
        from mlx_sparse.linalg._eigen import _float32_csr

        csr = _float16_csr()
        result = _float32_csr(csr)
        assert result.data.dtype == mx.float32

    def test_complex64_raises(self):
        from mlx_sparse.linalg._eigen import _float32_csr

        csr = ms.csr_array(
            (
                mx.array(np.array([1.0 + 0.0j], dtype=np.complex64)),
                mx.array([0], dtype=mx.int32),
                mx.array([0, 1, 1], dtype=mx.int32),
            ),
            shape=(2, 2),
            canonical=True,
        )
        with pytest.raises(TypeError, match="real float"):
            _float32_csr(csr)


# ---------------------------------------------------------------------------
# _eigen._ncv
# ---------------------------------------------------------------------------


class TestNcv:
    def test_none_ncv(self):
        from mlx_sparse.linalg._eigen import _ncv

        assert _ncv(10, 3, None) == min(10, max(4, 7))

    def test_explicit_ncv(self):
        from mlx_sparse.linalg._eigen import _ncv

        assert _ncv(10, 3, 5) == 5

    def test_ncv_clamped_to_n(self):
        from mlx_sparse.linalg._eigen import _ncv

        assert _ncv(4, 2, 100) == 4


# ---------------------------------------------------------------------------
# lanczos
# ---------------------------------------------------------------------------


class TestLanczos:
    def test_v0_not_none_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(NotImplementedError, match="start vector"):
            linalg.lanczos(csr, k=1, v0=mx.array([1.0, 0.0]))

    def test_k_zero_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="k must satisfy"):
            linalg.lanczos(csr, k=0)

    def test_k_too_large_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="k must satisfy"):
            linalg.lanczos(csr, k=3)  # k must be < n=2... wait k <= n

    def test_valid_lanczos(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        alphas, betas, basis = linalg.lanczos(csr, k=1, return_basis=True)
        mx.eval(alphas, betas, basis)
        assert np.array(alphas).shape[0] == 1

    def test_lanczos_no_basis(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        alphas, betas = linalg.lanczos(csr, k=1, return_basis=False)
        mx.eval(alphas, betas)
        assert np.array(alphas).shape[0] == 1


# ---------------------------------------------------------------------------
# eigsh error paths
# ---------------------------------------------------------------------------


class TestEigshErrors:
    def test_non_square_raises(self):
        csr = ms.csr_array(
            (
                mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
                mx.array([0, 1, 2], dtype=mx.int32),
                mx.array([0, 1, 2, 3], dtype=mx.int32),
            ),
            shape=(3, 4),
            canonical=True,
        )
        with pytest.raises(ValueError, match="square"):
            linalg.eigsh(csr, k=1)

    def test_k_zero_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="k must satisfy"):
            linalg.eigsh(csr, k=0)

    def test_k_equals_n_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="k must satisfy"):
            linalg.eigsh(csr, k=2)  # k must be < n=2

    def test_v0_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(NotImplementedError):
            linalg.eigsh(csr, k=1, v0=mx.array([1.0, 0.0]))

    def test_maxiter_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(NotImplementedError):
            linalg.eigsh(csr, k=1, maxiter=100)

    def test_tol_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(NotImplementedError):
            linalg.eigsh(csr, k=1, tol=1e-3)

    def test_return_eigenvectors_false(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        vals = linalg.eigsh(csr, k=1, return_eigenvectors=False)
        mx.eval(vals)
        assert np.array(vals).shape[0] == 1


# ---------------------------------------------------------------------------
# eigs error paths
# ---------------------------------------------------------------------------


class TestEigsErrors:
    def test_non_square_raises(self):
        csr = ms.csr_array(
            (
                mx.array([1.0, 2.0], dtype=mx.float32),
                mx.array([0, 1], dtype=mx.int32),
                mx.array([0, 1, 2, 2], dtype=mx.int32),
            ),
            shape=(3, 4),
            canonical=True,
        )
        with pytest.raises(ValueError, match="square"):
            linalg.eigs(csr, k=1)

    def test_k_zero_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="k must satisfy"):
            linalg.eigs(csr, k=0)

    def test_k_equals_n_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="k must satisfy"):
            linalg.eigs(csr, k=2)

    def test_v0_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(NotImplementedError):
            linalg.eigs(csr, k=1, v0=mx.array([1.0, 0.0]))

    def test_maxiter_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(NotImplementedError):
            linalg.eigs(csr, k=1, maxiter=100)

    def test_tol_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(NotImplementedError):
            linalg.eigs(csr, k=1, tol=1e-3)

    def test_return_eigenvectors_false(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        vals = linalg.eigs(csr, k=1, return_eigenvectors=False)
        mx.eval(vals)
        assert np.array(vals).shape[0] == 1


# ---------------------------------------------------------------------------
# svds error paths
# ---------------------------------------------------------------------------


class TestSvdsErrors:
    def test_k_zero_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="k must satisfy"):
            linalg.svds(csr, k=0)

    def test_k_equals_min_shape_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="k must satisfy"):
            linalg.svds(csr, k=2)  # k < min(2,2)=2 fails

    def test_tol_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(NotImplementedError):
            linalg.svds(csr, k=1, tol=1e-3)

    def test_bad_return_singular_vectors_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(ValueError, match="return_singular_vectors"):
            linalg.svds(csr, k=1, return_singular_vectors="bad")

    def test_return_singular_vectors_false(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        s = linalg.svds(csr, k=1, return_singular_vectors=False)
        mx.eval(s)
        assert np.array(s).shape[0] == 1

    def test_return_singular_vectors_u(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        u, s, vh = linalg.svds(csr, k=1, return_singular_vectors="u")
        mx.eval(u, s)
        assert vh is None
        assert np.array(u).shape == (2, 1)

    def test_return_singular_vectors_vh(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        u, s, vh = linalg.svds(csr, k=1, return_singular_vectors="vh")
        mx.eval(vh, s)
        assert u is None
        assert np.array(vh).shape == (1, 2)


# ---------------------------------------------------------------------------
# _factorizations._as_csr and _float32_csr
# ---------------------------------------------------------------------------


class TestFactorizationsAsCsr:
    def test_csr_passthrough(self):
        from mlx_sparse.linalg._factorizations import _as_csr

        csr = _spd_2x2(mx)
        result = _as_csr(csr)
        assert result.has_canonical_format

    def test_coo_converts(self):
        from mlx_sparse.linalg._factorizations import _as_csr

        coo = _coo_2x2()
        result = _as_csr(coo)
        assert isinstance(result, ms.CSRArray)

    def test_dense_raises(self):
        from mlx_sparse.linalg._factorizations import _as_csr

        with pytest.raises(TypeError, match="sparse factorization"):
            _as_csr(mx.array([[1.0, 0.0], [0.0, 1.0]], dtype=mx.float32))


class TestFactorizationsFloat32Csr:
    def test_float32_passthrough(self):
        from mlx_sparse.linalg._factorizations import _float32_csr

        csr = _spd_2x2(mx)
        result = _float32_csr(csr)
        assert result is csr

    def test_float16_promotes(self):
        from mlx_sparse.linalg._factorizations import _float32_csr

        csr = _float16_csr()
        result = _float32_csr(csr)
        assert result.data.dtype == mx.float32

    def test_complex64_raises(self):
        from mlx_sparse.linalg._factorizations import _float32_csr

        csr = ms.csr_array(
            (
                mx.array(np.array([1.0 + 0.0j], dtype=np.complex64)),
                mx.array([0], dtype=mx.int32),
                mx.array([0, 1, 1], dtype=mx.int32),
            ),
            shape=(2, 2),
            canonical=True,
        )
        with pytest.raises(TypeError, match="real float"):
            _float32_csr(csr)


# ---------------------------------------------------------------------------
# _factorizations._triangular_solve
# ---------------------------------------------------------------------------


class TestTriangularSolveErrors:
    def test_rank2_raises_not_implemented(self):
        from mlx_sparse.linalg._factorizations import _triangular_solve

        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        b2d = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="rank-1"):
            _triangular_solve(csr, b2d, lower=True, unit_diagonal=False)

    def test_rank0_raises_value_error(self):
        from mlx_sparse.linalg._factorizations import _triangular_solve

        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        b0d = mx.array(1.0)
        with pytest.raises(ValueError, match="rank-1 or rank-2"):
            _triangular_solve(csr, b0d, lower=True, unit_diagonal=False)


# ---------------------------------------------------------------------------
# SparseCholesky properties and callable interface
# ---------------------------------------------------------------------------


class TestSparseCholeskyExtended:
    def test_shape_property(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        chol = linalg.sparse_cholesky(csr)
        assert chol.shape == (2, 2)

    def test_callable_interface(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        chol = linalg.sparse_cholesky(csr)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x1 = chol.solve(b)
        x2 = chol(b)
        mx.eval(x1, x2)
        np.testing.assert_allclose(np.array(x1), np.array(x2))

    def test_upper_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(NotImplementedError, match="lower"):
            linalg.sparse_cholesky(csr, upper=True)

    def test_cholesky_alias(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        result = linalg.cholesky(csr)
        assert result.shape == (2, 2)

    def test_coo_input(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        coo = _coo_2x2()
        chol = linalg.sparse_cholesky(coo)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x = chol.solve(b)
        mx.eval(x)
        A_dense = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = np.linalg.solve(A_dense, [1.0, 2.0])
        np.testing.assert_allclose(np.array(x), expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# SparseLU properties and callable interface
# ---------------------------------------------------------------------------


class TestSparseLUExtended:
    def test_shape_property(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        lu = linalg.sparse_lu(csr)
        assert lu.shape == (2, 2)

    def test_callable_interface(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        lu = linalg.sparse_lu(csr)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x1 = lu.solve(b)
        x2 = lu(b)
        mx.eval(x1, x2)
        np.testing.assert_allclose(np.array(x1), np.array(x2))

    def test_solve_rank2_raises(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        lu = linalg.sparse_lu(csr)
        b2d = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        with pytest.raises(NotImplementedError, match="rank-1"):
            lu.solve(b2d)

    def test_splu_alias(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        lu = linalg.splu(csr)
        assert lu.shape == (2, 2)

    def test_spsolve(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        b = mx.array([1.0, 2.0], dtype=mx.float32)
        x = linalg.spsolve(csr, b)
        mx.eval(x)
        A_dense = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = np.linalg.solve(A_dense, [1.0, 2.0])
        np.testing.assert_allclose(np.array(x), expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# _sparse_ops COO path and error
# ---------------------------------------------------------------------------


class TestSparseOpsExtended:
    def test_vdot_with_coo(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        coo = _coo_2x2()
        result = linalg.vdot(csr, coo)
        mx.eval(result)
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = float(np.sum(A * A))
        np.testing.assert_allclose(float(np.array(result)), expected, rtol=1e-5)

    def test_dot_with_coo(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _spd_2x2(mx)
        coo = _coo_2x2()
        result = linalg.dot(csr, coo)
        mx.eval(result)
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = float(np.sum(A * A))
        np.testing.assert_allclose(float(np.array(result)), expected, rtol=1e-5)

    def test_vdot_bad_type_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(TypeError):
            linalg.vdot(csr, mx.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_dot_bad_type_raises(self):
        csr = _spd_2x2(mx)
        with pytest.raises(TypeError):
            linalg.dot(mx.array([[1.0, 0.0], [0.0, 1.0]]), csr)
