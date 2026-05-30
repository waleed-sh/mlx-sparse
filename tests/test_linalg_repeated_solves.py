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
from mlx_sparse._ext_loader import extension_available
from mlx_sparse._host import to_numpy

pytestmark = [pytest.mark.native, pytest.mark.cpu_only]


def _spd(mx, index_dtype=np.int32):
    return ms.csr_array(
        (
            mx.array(
                np.array(
                    [5.0, -1.0, -1.0, 4.0, -0.5, -0.5, 3.0],
                    dtype=np.float32,
                )
            ),
            mx.array(np.array([0, 1, 0, 1, 2, 1, 2], dtype=index_dtype)),
            mx.array(np.array([0, 2, 5, 7], dtype=index_dtype)),
        ),
        shape=(3, 3),
        validate="full",
        canonical=True,
    )


def _general(mx, index_dtype=np.int32):
    return ms.csr_array(
        (
            mx.array(
                np.array(
                    [2.0, 1.0, 3.0, -1.0, 4.0],
                    dtype=np.float32,
                )
            ),
            mx.array(np.array([0, 1, 1, 0, 2], dtype=index_dtype)),
            mx.array(np.array([0, 2, 3, 5], dtype=index_dtype)),
        ),
        shape=(3, 3),
        validate="full",
        canonical=True,
    )


def _require_native():
    if not extension_available():
        pytest.skip("native extension unavailable")


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_sparse_cholesky_matrix_rhs_matches_numpy_and_serial(mx, to_numpy, index_dtype):
    _require_native()
    a = _spd(mx, index_dtype=index_dtype)
    dense = to_numpy(a.todense())
    rhs_np = np.array(
        [[1.0, 0.25, -2.0], [-2.0, 1.5, 0.5], [0.5, -0.75, 3.0]],
        dtype=np.float32,
    )
    factor = linalg.sparse_cholesky(a)

    with ms.runtime.context(n_threads=1):
        serial = factor.solve(mx.array(rhs_np))
    with ms.runtime.context(n_threads=3, solver_parallel=True, solver_threads=3):
        parallel = factor.solve(mx.array(rhs_np))

    expected = np.linalg.solve(dense, rhs_np)
    np.testing.assert_allclose(to_numpy(serial), expected, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(to_numpy(parallel), expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_sparse_lu_matrix_rhs_matches_numpy_and_factorized(mx, to_numpy, index_dtype):
    _require_native()
    a = _general(mx, index_dtype=index_dtype)
    dense = to_numpy(a.todense())
    rhs_np = np.array(
        [[1.0, 0.25, -2.0, 4.0], [-2.0, 1.5, 0.5, -1.0], [0.5, -0.75, 3.0, 2.0]],
        dtype=np.float32,
    )
    factor = linalg.sparse_lu(a)
    solver = linalg.factorized(a, method="lu")

    with ms.runtime.context(n_threads=1):
        serial = factor.solve(mx.array(rhs_np))
    with ms.runtime.context(n_threads=3, solver_parallel=True, solver_threads=3):
        parallel = factor.solve(mx.array(rhs_np))

    expected = np.linalg.solve(dense, rhs_np)
    np.testing.assert_allclose(to_numpy(serial), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(to_numpy(parallel), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        to_numpy(solver.solve(mx.array(rhs_np))), expected, rtol=1e-4, atol=1e-4
    )


def test_rank2_single_rhs_matches_rank1_solve(mx, to_numpy):
    _require_native()
    a = _general(mx)
    factor = linalg.sparse_lu(a)
    rhs_np = np.array([1.0, -2.0, 0.5], dtype=np.float32)

    vector = factor.solve(mx.array(rhs_np))
    matrix = factor.solve(mx.array(rhs_np[:, None]))

    np.testing.assert_allclose(
        to_numpy(matrix)[:, 0], to_numpy(vector), rtol=1e-6, atol=1e-6
    )


def test_native_permutation_accepts_matrix_rhs(mx, to_numpy):
    _require_native()
    rhs_np = np.array(
        [[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]],
        dtype=np.float32,
    )
    perm_np = np.array([2, 0, 1], dtype=np.int32)

    out = native.csr_permute_vector(mx.array(rhs_np), mx.array(perm_np))

    np.testing.assert_array_equal(to_numpy(out), rhs_np[perm_np])


def test_matrix_rhs_shape_validation(mx):
    _require_native()
    factor = linalg.sparse_lu(_general(mx))

    with pytest.raises(ValueError, match="first dimension"):
        factor.solve(mx.ones((2, 2), dtype=mx.float32))
    with pytest.raises(ValueError, match="at least one column"):
        factor.solve(mx.ones((3, 0), dtype=mx.float32))
