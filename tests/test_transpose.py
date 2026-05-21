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


def test_csr_transpose_matches_dense_transpose(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4), sorted_indices=True)
    transposed = csr.T

    assert transposed.shape == (4, 3)
    assert transposed.sorted_indices
    np.testing.assert_allclose(
        to_numpy(transposed.todense()), to_numpy(csr.todense()).T
    )


def test_hermitian_conjugates_values_and_transposes(mx):
    data = mx.array(np.array([1.0 + 2.0j, 3.0 - 4.0j, -2.0 + 1.0j], dtype=np.complex64))
    indices = mx.array(np.array([0, 2, 1], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 3], dtype=np.int32))

    csr = ms.csr_array((data, indices, indptr), shape=(2, 3), sorted_indices=True)
    hermitian = csr.H

    assert hermitian.shape == (3, 2)
    np.testing.assert_allclose(
        to_numpy(hermitian.todense()),
        np.conjugate(to_numpy(csr.todense())).T,
    )


def test_transpose_preserves_canonical_flag_when_known(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0, 2.0], dtype=np.float32)),
            mx.array(np.array([0, 2], dtype=np.int32)),
            mx.array(np.array([0, 1, 2], dtype=np.int32)),
        ),
        shape=(2, 3),
        sorted_indices=True,
        canonical=True,
    )

    assert csr.T.has_canonical_format
