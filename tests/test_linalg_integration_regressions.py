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


def _require_native():
    if not extension_available():
        pytest.skip("native extension unavailable")


def _csr_from_dense(mx, dense: np.ndarray):
    dense = np.asarray(dense)
    row, col = np.nonzero(dense)
    data = dense[row, col].astype(dense.dtype, copy=False)
    indptr = np.zeros(dense.shape[0] + 1, dtype=np.int32)
    np.add.at(indptr, row + 1, 1)
    np.cumsum(indptr, out=indptr)
    return ms.csr_array(
        (
            mx.array(data),
            mx.array(col.astype(np.int32, copy=False)),
            mx.array(indptr),
        ),
        shape=dense.shape,
        sorted_indices=True,
        canonical=True,
        validate="full",
    )


def _csc_from_dense(mx, dense: np.ndarray):
    dense = np.asarray(dense)
    data_parts = []
    index_parts = []
    indptr = np.zeros(dense.shape[1] + 1, dtype=np.int32)
    for col in range(dense.shape[1]):
        rows = np.nonzero(dense[:, col])[0].astype(np.int32, copy=False)
        index_parts.append(rows)
        data_parts.append(dense[rows, col].astype(dense.dtype, copy=False))
        indptr[col + 1] = indptr[col] + rows.size
    data = (
        np.concatenate(data_parts) if data_parts else np.empty((0,), dtype=dense.dtype)
    )
    indices = (
        np.concatenate(index_parts) if index_parts else np.empty((0,), dtype=np.int32)
    )
    return ms.csc_array(
        (mx.array(data), mx.array(indices), mx.array(indptr)),
        shape=dense.shape,
        sorted_indices=True,
        canonical=True,
        validate="full",
    )


def _poisson_2d(n: int) -> np.ndarray:
    size = n * n
    dense = np.zeros((size, size), dtype=np.float32)
    for row in range(n):
        for col in range(n):
            idx = row * n + col
            dense[idx, idx] = 4.0
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nrow = row + drow
                ncol = col + dcol
                if 0 <= nrow < n and 0 <= ncol < n:
                    dense[idx, nrow * n + ncol] = -1.0
    return dense


def _shifted_path_laplacian(n: int, shift: float) -> np.ndarray:
    dense = np.eye(n, dtype=np.float32) * np.float32(shift)
    for idx in range(n):
        degree = 0
        if idx > 0:
            dense[idx, idx - 1] = -1.0
            degree += 1
        if idx + 1 < n:
            dense[idx, idx + 1] = -1.0
            degree += 1
        dense[idx, idx] += degree
    return dense


def _relative_residual(dense: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(dense @ x - b) / max(np.linalg.norm(b), 1.0))


def test_pde_poisson_system_solvers_and_factorizations(mx, to_numpy):
    _require_native()
    dense = _poisson_2d(4)
    csr = _csr_from_dense(mx, dense)
    x_true = np.sin(np.arange(dense.shape[0], dtype=np.float32) / 5.0)
    rhs = dense @ x_true

    for solver in (linalg.cg, linalg.gmres, linalg.minres):
        x, info = solver(csr, mx.array(rhs), rtol=1e-6, atol=1e-7, maxiter=256)
        got = to_numpy(x)
        assert info == 0, solver.__name__
        assert _relative_residual(dense, got, rhs) < 2e-5
        np.testing.assert_allclose(got, x_true, rtol=2e-4, atol=2e-4)

    chol = linalg.sparse_cholesky(csr)
    lu = linalg.sparse_lu(csr)
    for got in (to_numpy(chol.solve(mx.array(rhs))), to_numpy(lu.solve(mx.array(rhs)))):
        assert _relative_residual(dense, got, rhs) < 2e-5
        np.testing.assert_allclose(got, x_true, rtol=2e-4, atol=2e-4)


def test_shifted_graph_laplacian_csc_solver_and_hermitian_eigsh(mx, to_numpy):
    _require_native()
    dense = _shifted_path_laplacian(12, shift=0.25)
    csc = _csc_from_dense(mx, dense)
    x_true = np.cos(np.arange(dense.shape[0], dtype=np.float32) / 3.0)
    rhs = dense @ x_true

    for solver in (linalg.cg, linalg.minres):
        x, info = solver(csc, mx.array(rhs), rtol=1e-6, atol=1e-7, maxiter=128)
        got = to_numpy(x)
        assert info == 0, solver.__name__
        assert _relative_residual(dense, got, rhs) < 2e-5

    smallest = linalg.eigsh(csc, k=1, which="SM", ncv=8, return_eigenvectors=False)
    expected = np.linalg.eigvalsh(dense)
    np.testing.assert_allclose(to_numpy(smallest), expected[:1], rtol=5e-3, atol=5e-3)


def test_symmetric_indefinite_system_minres_lu_and_eigsh(mx, to_numpy):
    _require_native()
    dense = np.array(
        [
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [2.0, -1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, -3.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 4.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, 2.0],
        ],
        dtype=np.float32,
    )
    csr = _csr_from_dense(mx, dense)
    x_true = np.array([1.0, -0.5, 0.75, 2.0, -1.5], dtype=np.float32)
    rhs = dense @ x_true

    x_minres, info = linalg.minres(
        csr, mx.array(rhs), rtol=1e-6, atol=1e-7, maxiter=128
    )
    assert info == 0
    assert _relative_residual(dense, to_numpy(x_minres), rhs) < 2e-5

    x_lu = linalg.spsolve(csr, mx.array(rhs))
    np.testing.assert_allclose(to_numpy(x_lu), x_true, rtol=2e-4, atol=2e-4)
    with pytest.raises(RuntimeError):
        linalg.sparse_cholesky(csr)

    expected = np.linalg.eigvalsh(dense)
    got_sa = linalg.eigsh(csr, k=1, which="SA", ncv=5, return_eigenvectors=False)
    got_la = linalg.eigsh(csr, k=1, which="LA", ncv=5, return_eigenvectors=False)
    np.testing.assert_allclose(to_numpy(got_sa), expected[:1], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(to_numpy(got_la), expected[-1:], rtol=5e-3, atol=5e-3)


def test_rectangular_sparse_svds_and_normal_equation_solver(mx, to_numpy):
    _require_native()
    dense = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, -1.0, 0.0, 0.5],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, -1.0, 0.0],
            [0.5, 0.0, 0.0, 1.5],
            [0.0, -0.25, 1.0, 0.0],
            [2.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    csc = _csc_from_dense(mx, dense)
    singular = linalg.svds(csc, k=2, ncv=5, return_singular_vectors=False)
    expected_singular = np.linalg.svd(dense, compute_uv=False)[:2]
    np.testing.assert_allclose(
        np.sort(to_numpy(singular)),
        np.sort(expected_singular),
        rtol=5e-3,
        atol=5e-3,
    )

    x_true = np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float32)
    rhs = dense @ x_true
    normal = dense.T @ dense
    normal_rhs = dense.T @ rhs
    normal_csr = _csr_from_dense(mx, normal)
    x_cg, info = linalg.cg(
        normal_csr, mx.array(normal_rhs), rtol=1e-6, atol=1e-7, maxiter=128
    )

    assert info == 0
    np.testing.assert_allclose(to_numpy(x_cg), x_true, rtol=5e-4, atol=5e-4)


def test_nearly_singular_ill_conditioned_system_residual_contracts(mx, to_numpy):
    _require_native()
    q, _ = np.linalg.qr(
        np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, 1.0, 0.0],
                [0.0, 1.0, -1.0, 1.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            dtype=np.float32,
        )
    )
    spectrum = np.array([1e-4, 2e-3, 0.5, 3.0], dtype=np.float32)
    dense = (q @ np.diag(spectrum) @ q.T).astype(np.float32)
    csr = _csr_from_dense(mx, dense)
    b = np.array([1.0, -0.25, 0.5, -1.5], dtype=np.float32)

    x_direct = to_numpy(linalg.spsolve(csr, mx.array(b)))
    assert np.linalg.cond(dense) > 1e4
    assert _relative_residual(dense, x_direct, b) < 2e-3

    x_gmres, _ = linalg.gmres(
        csr, mx.array(b), rtol=1e-5, atol=1e-6, restart=4, maxiter=64
    )
    residual = _relative_residual(dense, to_numpy(x_gmres), b)
    assert np.isfinite(residual)
    assert residual < 5e-2


@pytest.mark.parametrize("format_name", ["csr", "csc"])
def test_autodiff_optimization_learns_sparse_operator_then_solver_uses_it(
    mx, to_numpy, format_name
):
    _require_native()
    dense = _poisson_2d(3)
    row, col = np.nonzero(dense)
    true_csr_values = dense[row, col].astype(np.float32)
    csr_indptr = np.zeros(dense.shape[0] + 1, dtype=np.int32)
    np.add.at(csr_indptr, row + 1, 1)
    np.cumsum(csr_indptr, out=csr_indptr)

    csc_rows = []
    csc_values = []
    csc_indptr = np.zeros(dense.shape[1] + 1, dtype=np.int32)
    for j in range(dense.shape[1]):
        rows = np.nonzero(dense[:, j])[0].astype(np.int32, copy=False)
        csc_rows.append(rows)
        csc_values.append(dense[rows, j].astype(np.float32, copy=False))
        csc_indptr[j + 1] = csc_indptr[j] + rows.size
    true_csc_values = np.concatenate(csc_values)

    if format_name == "csr":
        true_values = true_csr_values
        indices = mx.array(col.astype(np.int32, copy=False))
        indptr = mx.array(csr_indptr)

        def make_sparse(values):
            return ms.csr_array(
                (values, indices, indptr),
                shape=dense.shape,
                sorted_indices=True,
                canonical=True,
            )

    else:
        true_values = true_csc_values
        indices = mx.array(np.concatenate(csc_rows))
        indptr = mx.array(csc_indptr)

        def make_sparse(values):
            return ms.csc_array(
                (values, indices, indptr),
                shape=dense.shape,
                sorted_indices=True,
                canonical=True,
            )

    probes = mx.array(np.eye(dense.shape[1], dtype=np.float32))
    target = mx.array(dense.astype(np.float32))
    values = mx.array(
        true_values + np.linspace(-0.35, 0.35, true_values.size, dtype=np.float32)
    )

    def loss_fn(values):
        residual = make_sparse(values) @ probes - target
        return mx.sum(residual * residual)

    initial_loss = float(to_numpy(loss_fn(values)))
    grad_fn = mx.grad(loss_fn)
    for _ in range(40):
        values = values - np.float32(0.2) * grad_fn(values)
        mx.eval(values)
    final_loss = float(to_numpy(loss_fn(values)))

    assert final_loss < initial_loss * 1e-6
    np.testing.assert_allclose(to_numpy(values), true_values, rtol=2e-4, atol=2e-4)

    learned = make_sparse(values)
    x_true = np.linspace(-1.0, 1.0, dense.shape[0], dtype=np.float32)
    rhs = dense @ x_true
    x, info = linalg.cg(learned, mx.array(rhs), rtol=1e-6, atol=1e-7, maxiter=128)

    assert info == 0
    np.testing.assert_allclose(to_numpy(x), x_true, rtol=5e-4, atol=5e-4)
