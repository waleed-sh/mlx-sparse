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


def test_preconditioner_namespace_is_public():
    assert "preconditioners" in linalg.__all__
    assert linalg.preconditioners is preconditioners
    assert callable(preconditioners.jacobi)
    assert callable(preconditioners.diagonal)
    assert callable(preconditioners.identity)


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
