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


def test_csr_todense_sums_duplicate_entries(mx):
    data = mx.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    indices = mx.array(np.array([1, 1, 0, 2], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 3, 4], dtype=np.int32))

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    dense = csr.todense()

    expected = np.array(
        [
            [0.0, 3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(to_numpy(dense), expected)


def test_csr_todense_matches_scipy_random(mx, scipy_sparse):
    rng = np.random.default_rng(123)
    scipy_csr = scipy_sparse.random(
        32,
        48,
        density=0.08,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )

    csr = ms.csr_array(
        (
            mx.array(scipy_csr.data.astype(np.float32)),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        validate="metadata",
        sorted_indices=True,
        canonical=True,
    )

    np.testing.assert_allclose(to_numpy(csr.todense()), scipy_csr.toarray(), atol=1e-6)
