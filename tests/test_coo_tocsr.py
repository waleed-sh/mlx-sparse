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
from conftest import to_numpy

import mlx_sparse as ms


def test_coo_tocsr_sorts_by_row_then_column(mx):
    data = mx.array(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
    row = mx.array(np.array([2, 0, 1, 0], dtype=np.int32))
    col = mx.array(np.array([1, 3, 2, 0], dtype=np.int32))

    csr = ms.coo_array((data, (row, col)), shape=(3, 4)).tocsr()

    np.testing.assert_array_equal(to_numpy(csr.indptr), np.array([0, 2, 3, 4]))
    np.testing.assert_array_equal(to_numpy(csr.indices), np.array([0, 3, 2, 1]))
    np.testing.assert_array_equal(
        to_numpy(csr.data),
        np.array([40.0, 20.0, 30.0, 10.0], dtype=np.float32),
    )


def test_canonicalize_sums_duplicate_coordinates(mx):
    data = mx.array(np.array([2.0, 3.0, 5.0], dtype=np.float32))
    row = mx.array(np.array([0, 0, 0], dtype=np.int32))
    col = mx.array(np.array([1, 1, 0], dtype=np.int32))

    csr = ms.coo_array((data, (row, col)), shape=(1, 3)).tocsr().canonicalize()

    np.testing.assert_array_equal(to_numpy(csr.indptr), np.array([0, 2]))
    np.testing.assert_array_equal(to_numpy(csr.indices), np.array([0, 1]))
    np.testing.assert_allclose(
        to_numpy(csr.data), np.array([5.0, 5.0], dtype=np.float32)
    )
    assert csr.has_canonical_format


def test_empty_coo_to_csr(mx):
    data = mx.array(np.array([], dtype=np.float32))
    row = mx.array(np.array([], dtype=np.int32))
    col = mx.array(np.array([], dtype=np.int32))

    csr = ms.coo_array((data, (row, col)), shape=(4, 5)).tocsr()

    np.testing.assert_array_equal(to_numpy(csr.indptr), np.zeros(5, dtype=np.int32))
    assert csr.nnz == 0
