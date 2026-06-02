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


def _csr_from_dense(mx, dense: np.ndarray) -> ms.CSRArray:
    rows, cols = np.nonzero(dense)
    data = dense[rows, cols].astype(np.float32, copy=False)
    indptr = np.zeros((dense.shape[0] + 1,), dtype=np.int32)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr, dtype=np.int32)
    order = np.argsort(rows * dense.shape[1] + cols)
    return ms.csr_array(
        (
            mx.array(data[order]),
            mx.array(cols[order].astype(np.int32, copy=False)),
            mx.array(indptr),
        ),
        shape=dense.shape,
        canonical=True,
    )


def test_spsolve_triangular_lower_rank1_rank2_matches_numpy(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    lower = np.array(
        [[2.0, 0.0, 0.0], [-1.0, 3.0, 0.0], [4.0, 1.0, -2.0]],
        dtype=np.float32,
    )
    csr = _csr_from_dense(mx, lower)
    rhs_vector = mx.array([2.0, 5.0, -3.0], dtype=mx.float32)
    rhs_matrix_np = np.array([[2.0, 1.0], [5.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    rhs_matrix = mx.array(rhs_matrix_np)

    x_vector = linalg.spsolve_triangular(csr, rhs_vector, lower=True)
    x_matrix = linalg.spsolve_triangular(csr, rhs_matrix, lower=True)
    mx.eval(x_vector, x_matrix)

    np.testing.assert_allclose(
        np.array(x_vector), np.linalg.solve(lower, [2.0, 5.0, -3.0]), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.array(x_matrix), np.linalg.solve(lower, rhs_matrix_np), rtol=1e-5
    )


def test_spsolve_triangular_upper_unit_diagonal_matches_numpy(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    upper_unit = np.array(
        [[1.0, -2.0, 1.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    csr = _csr_from_dense(mx, upper_unit)
    rhs = mx.array([4.0, -1.0, 2.0], dtype=mx.float32)

    x = linalg.spsolve_triangular(
        csr,
        rhs,
        lower=False,
        unit_diagonal=True,
    )
    mx.eval(x)

    np.testing.assert_allclose(
        np.array(x), np.linalg.solve(upper_unit, np.array([4.0, -1.0, 2.0])), rtol=1e-5
    )


def test_spsolve_triangular_rejects_public_analysis_placeholder(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    csr = _csr_from_dense(mx, np.eye(2, dtype=np.float32))
    with pytest.raises(NotImplementedError, match="analysis"):
        linalg.spsolve_triangular(
            csr, mx.ones((2,), dtype=mx.float32), analyzed=object()
        )


def test_matrix_free_cg_with_callable_preconditioner_matches_numpy(mx):
    dense = np.array(
        [[5.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 3.0]],
        dtype=np.float32,
    )
    rhs_np = np.array([1.0, 2.0, -1.0], dtype=np.float32)
    operator = linalg.LinearOperator(
        dense.shape,
        matvec=lambda x: mx.array(dense) @ x,
        dtype=mx.float32,
    )
    inv_diag = mx.array(1.0 / np.diag(dense), dtype=mx.float32)

    x, info = linalg.cg(
        operator,
        mx.array(rhs_np),
        M=lambda r: inv_diag * r,
        rtol=1e-6,
        maxiter=32,
    )
    mx.eval(x)

    assert info == 0
    np.testing.assert_allclose(
        np.array(x), np.linalg.solve(dense, rhs_np), rtol=2e-5, atol=2e-5
    )


def test_matrix_free_gmres_left_preconditioner_matches_numpy(mx):
    dense = np.array(
        [[4.0, 2.0, 0.0], [-1.0, 3.0, 1.0], [0.0, -2.0, 5.0]],
        dtype=np.float32,
    )
    rhs_np = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    operator = linalg.LinearOperator(
        dense.shape,
        matvec=lambda x: mx.array(dense) @ x,
        dtype=mx.float32,
    )
    inv_diag = mx.array(1.0 / np.diag(dense), dtype=mx.float32)

    x, info = linalg.gmres(
        operator,
        mx.array(rhs_np),
        M=lambda r: inv_diag * r,
        restart=3,
        rtol=1e-6,
        maxiter=24,
    )
    mx.eval(x)

    assert info == 0
    np.testing.assert_allclose(
        np.array(x), np.linalg.solve(dense, rhs_np), rtol=2e-5, atol=2e-5
    )


def test_spectral_v0_matches_full_basis_reference(mx):
    if not extension_available():
        pytest.skip("native extension unavailable")
    diagonal = np.diag([1.0, 2.0, 4.0, 8.0]).astype(np.float32)
    csr = _csr_from_dense(mx, diagonal)
    v0 = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)

    eigsh_values = linalg.eigsh(
        csr,
        k=2,
        v0=v0,
        ncv=4,
        which="LM",
        return_eigenvectors=False,
    )
    eigs_values = linalg.eigs(
        csr,
        k=2,
        v0=v0,
        ncv=4,
        which="LM",
        return_eigenvectors=False,
    )
    svd_values = linalg.svds(
        csr,
        k=2,
        v0=v0,
        ncv=4,
        which="LM",
        return_singular_vectors=False,
    )
    mx.eval(eigsh_values, eigs_values, svd_values)

    np.testing.assert_allclose(np.sort(np.array(eigsh_values)), [4.0, 8.0], rtol=2e-4)
    np.testing.assert_allclose(np.sort(np.array(eigs_values)), [4.0, 8.0], rtol=2e-4)
    np.testing.assert_allclose(np.sort(np.array(svd_values)), [4.0, 8.0], rtol=2e-4)
