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
from mlx_sparse._host import to_numpy

import mlx_sparse as ms


def test_csr_matmul_matches_hand_computed(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    rhs = mx.array(
        np.array(
            [
                [3.0, 1.0],
                [10.0, -2.0],
                [7.0, 4.0],
                [-2.0, 6.0],
            ],
            dtype=np.float32,
        )
    )

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    out = csr @ rhs

    expected = np.array([[-1.0, -2.0], [0.0, 0.0], [30.0, 22.0]], dtype=np.float32)
    np.testing.assert_allclose(to_numpy(out), expected)


def test_csr_matmul_matches_scipy_random(mx, scipy_sparse):
    rng = np.random.default_rng(789)
    scipy_csr = scipy_sparse.random(
        64,
        80,
        density=0.04,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    rhs_np = rng.normal(size=(80, 7)).astype(np.float32)

    csr = ms.csr_array(
        (
            mx.array(scipy_csr.data.astype(np.float32)),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
        canonical=True,
    )

    np.testing.assert_allclose(
        to_numpy(csr @ mx.array(rhs_np)),
        scipy_csr @ rhs_np,
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_matmul_batched_rhs_matches_dense_mlx(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    rhs_np = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) / 10

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    out = csr @ mx.array(rhs_np)

    expected = to_numpy(csr.todense()) @ rhs_np
    np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-5)


def test_csr_sparse_sparse_matmul_matches_dense(mx):
    lhs = ms.csr_array(
        (
            mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32)),
            mx.array(np.array([0, 2, 1, 3], dtype=np.int32)),
            mx.array(np.array([0, 2, 2, 4], dtype=np.int32)),
        ),
        shape=(3, 4),
        sorted_indices=True,
        canonical=True,
    )
    rhs = ms.csr_array(
        (
            mx.array(np.array([3.0, 7.0, -2.0, 6.0], dtype=np.float32)),
            mx.array(np.array([0, 1, 1, 2], dtype=np.int32)),
            mx.array(np.array([0, 1, 2, 3, 4], dtype=np.int32)),
        ),
        shape=(4, 3),
        sorted_indices=True,
        canonical=True,
    )

    out = lhs @ rhs

    assert isinstance(out, ms.CSRArray)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        to_numpy(lhs.todense()) @ to_numpy(rhs.todense()),
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_matmul_rejects_wrong_rhs_shape(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
        ),
        shape=(1, 2),
    )

    try:
        csr @ mx.array(np.ones((1, 2), dtype=np.float32))
    except ValueError as exc:
        assert "n_cols" in str(exc)
    else:
        raise AssertionError("expected csr_matmul shape validation to fail")
