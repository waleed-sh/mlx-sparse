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
from mlx_sparse._host import to_numpy


def test_csr_matvec_matches_hand_computed(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    x = mx.array(np.array([3.0, 10.0, 7.0, -2.0], dtype=np.float32))

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    y = csr @ x

    np.testing.assert_allclose(
        to_numpy(y), np.array([-1.0, 0.0, 30.0], dtype=np.float32)
    )


def test_csr_matvec_matches_scipy_random(mx, scipy_sparse):
    rng = np.random.default_rng(456)
    scipy_csr = scipy_sparse.random(
        96,
        128,
        density=0.03,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    x_np = rng.normal(size=(128,)).astype(np.float32)

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
    y = csr @ mx.array(x_np)

    np.testing.assert_allclose(to_numpy(y), scipy_csr @ x_np, rtol=1e-5, atol=1e-5)


def test_csr_matvec_accepts_noncontiguous_inputs(mx):
    data = mx.array(
        np.array([99.0, 2.0, 99.0, -1.0, 99.0, 4.0, 99.0, 5.0], dtype=np.float32)
    )[1::2]
    indices = mx.array(np.array([9, 0, 9, 2, 9, 1, 9, 3], dtype=np.int32))[1::2]
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    x = mx.array(
        np.array([99.0, 3.0, 99.0, 10.0, 99.0, 7.0, 99.0, -2.0], dtype=np.float32)
    )[1::2]

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    y = csr @ x

    np.testing.assert_allclose(
        to_numpy(y), np.array([-1.0, 0.0, 30.0], dtype=np.float32)
    )


def test_csr_matvec_rejects_wrong_rhs_shape(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
        ),
        shape=(1, 2),
    )

    with pytest.raises(ValueError, match="n_cols"):
        csr @ mx.array(np.array([1.0], dtype=np.float32))


def test_csr_matvec_rejects_unsupported_dtype(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
        ),
        shape=(1, 1),
    )

    with pytest.raises(TypeError, match="dtype must be one"):
        csr @ mx.array(np.array([1], dtype=np.int32))


def test_no_hidden_eager_eval_composes_with_mlx_ops(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0, 2.0], dtype=np.float32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 2], dtype=np.int32)),
        ),
        shape=(1, 2),
    )
    x = mx.array(np.array([3.0, 4.0], dtype=np.float32))

    z = mx.sin(csr @ x) + 2.0

    np.testing.assert_allclose(
        to_numpy(z), np.sin(np.array([11.0], dtype=np.float32)) + 2.0
    )
