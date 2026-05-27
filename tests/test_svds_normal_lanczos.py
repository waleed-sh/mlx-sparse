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
import mlx_sparse._native as native
from mlx_sparse import linalg


def _dense_to_csr_buffers(mx, dense: np.ndarray, index_dtype):
    data = []
    indices = []
    indptr = [0]
    for row in dense:
        cols = np.flatnonzero(row)
        data.extend(row[cols].astype(np.float32, copy=False))
        indices.extend(cols.astype(index_dtype, copy=False))
        indptr.append(len(data))
    return (
        mx.array(np.asarray(data, dtype=np.float32)),
        mx.array(np.asarray(indices, dtype=index_dtype)),
        mx.array(np.asarray(indptr, dtype=index_dtype)),
    )


def _reference_normal_lanczos(dense: np.ndarray, steps: int):
    normal = dense.T @ dense
    n_cols = normal.shape[0]
    basis = np.zeros((n_cols, steps), dtype=np.float32)
    alphas = np.zeros((steps,), dtype=np.float32)
    betas = np.zeros((steps,), dtype=np.float32)
    basis[:, 0] = np.float32(1.0 / np.sqrt(n_cols))

    beta_prev = np.float32(0.0)
    used = 0
    for j in range(steps):
        w = (normal @ basis[:, j]).astype(np.float32, copy=False)
        if j > 0:
            w = (w - beta_prev * basis[:, j - 1]).astype(np.float32, copy=False)
        alpha = np.float32(np.dot(basis[:, j].astype(np.float64), w.astype(np.float64)))
        alphas[j] = alpha
        w = (w - alpha * basis[:, j]).astype(np.float32, copy=False)

        for _ in range(2):
            for col in range(j + 1):
                correction = np.float32(
                    np.dot(basis[:, col].astype(np.float64), w.astype(np.float64))
                )
                w = (w - correction * basis[:, col]).astype(np.float32, copy=False)

        beta = np.float32(np.sqrt(max(float(np.dot(w, w)), 0.0)))
        betas[j] = beta
        used = j + 1
        if j + 1 == steps or beta <= np.finfo(np.float32).eps:
            break
        basis[:, j + 1] = w / beta
        beta_prev = beta

    return alphas, betas, basis, used


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_native_csr_normal_lanczos_matches_dense_reference(
    mx, to_numpy, index_dtype
):
    dense = np.array(
        [
            [3.0, 0.0, 1.0, 0.0],
            [0.0, -2.0, 0.0, 4.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, -1.0],
            [2.0, 0.0, -3.0, 0.0],
        ],
        dtype=np.float32,
    )
    shape = dense.shape
    steps = shape[1]
    data, indices, indptr = _dense_to_csr_buffers(mx, dense, index_dtype)

    alphas, betas, basis, actual = native.csr_normal_lanczos(
        data, indices, indptr, shape, k=steps
    )
    mx.eval(alphas, betas, basis, actual)

    expected_alphas, expected_betas, expected_basis, expected_used = (
        _reference_normal_lanczos(dense, steps)
    )
    used = int(to_numpy(actual).item())

    assert used == expected_used
    np.testing.assert_allclose(
        to_numpy(alphas)[:used], expected_alphas[:used], rtol=3e-5, atol=3e-5
    )
    np.testing.assert_allclose(
        to_numpy(betas)[:used], expected_betas[:used], rtol=3e-5, atol=3e-5
    )
    np.testing.assert_allclose(
        to_numpy(basis)[:, :used],
        expected_basis[:, :used],
        rtol=3e-5,
        atol=3e-5,
    )


def test_svds_ill_conditioned_rectangular_singular_values_match_dense(mx, to_numpy):
    dense = np.array(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.01],
            [0.2, 0.0, 0.0, 0.0],
            [0.0, -0.05, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    csr = ms.fromdense(mx.array(dense))

    singular = linalg.svds(csr, k=2, which="LM", ncv=4, return_singular_vectors=False)
    expected = np.linalg.svd(dense, compute_uv=False)[:2]

    np.testing.assert_allclose(
        np.sort(to_numpy(singular)),
        np.sort(expected),
        rtol=2e-4,
        atol=2e-4,
    )


def test_svds_rectangular_vectors_satisfy_singular_equations(mx, to_numpy):
    dense = np.array(
        [
            [4.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.25, 0.0, 0.0],
            [0.0, -0.125, 0.0],
        ],
        dtype=np.float32,
    )
    csr = ms.fromdense(mx.array(dense))

    u, singular, vh = linalg.svds(csr, k=2, which="LM", ncv=3)
    mx.eval(u, singular, vh)

    u_np = to_numpy(u)
    s_np = to_numpy(singular)
    vh_np = to_numpy(vh)
    for col, sigma in enumerate(s_np):
        right = vh_np[col]
        left = u_np[:, col]
        np.testing.assert_allclose(dense @ right, sigma * left, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(
            dense.T @ left, sigma * right, rtol=2e-4, atol=2e-4
        )


def test_svds_smallest_mode_uses_normal_operator_lanczos(mx, to_numpy):
    dense = np.diag(np.array([8.0, 4.0, 1.0, 0.25], dtype=np.float32))
    csr = ms.fromdense(mx.array(dense))

    singular = linalg.svds(csr, k=1, which="SM", ncv=4, return_singular_vectors=False)

    np.testing.assert_allclose(to_numpy(singular), [0.25], rtol=2e-4, atol=2e-4)
