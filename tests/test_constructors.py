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
from mlx_sparse._host import to_numpy

import mlx_sparse as ms


def test_coo_constructor_metadata(mx):
    data = mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    row = mx.array(np.array([0, 0, 2], dtype=np.int32))
    col = mx.array(np.array([1, 3, 0], dtype=np.int32))

    coo = ms.coo_array((data, (row, col)), shape=(3, 4))

    assert coo.shape == (3, 4)
    assert coo.nnz == 3
    assert coo.dtype == mx.float32
    assert coo.index_dtype == mx.int32


def test_csr_constructor_metadata(mx):
    data = mx.array(np.array([4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([1, 0], dtype=np.int32))
    indptr = mx.array(np.array([0, 1, 1, 2], dtype=np.int32))

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))

    assert csr.shape == (3, 4)
    assert csr.nnz == 2
    assert "CSRArray" in repr(csr)


def test_csr_constructor_preserves_int64_index_dtype(mx):
    data = mx.array(np.array([1.0], dtype=np.float32))
    indices = mx.array(np.array([0], dtype=np.int64))
    indptr = mx.array(np.array([0, 1], dtype=np.int64))

    csr = ms.csr_array((data, indices, indptr), shape=(1, 1))

    assert csr.index_dtype == mx.int64


def test_csr_rejects_bad_indptr_length(mx):
    data = mx.array(np.array([1.0], dtype=np.float32))
    indices = mx.array(np.array([0], dtype=np.int32))
    indptr = mx.array(np.array([0, 1], dtype=np.int32))

    with pytest.raises(ValueError, match="indptr"):
        ms.csr_array((data, indices, indptr), shape=(3, 4))


def test_csr_full_validation_rejects_nonmonotonic_indptr(mx):
    data = mx.array(np.array([1.0, 2.0], dtype=np.float32))
    indices = mx.array(np.array([0, 1], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 1, 2], dtype=np.int32))

    with pytest.raises(ValueError, match="monotonically"):
        ms.csr_array((data, indices, indptr), shape=(3, 3), validate="full")


def test_coo_full_validation_rejects_out_of_bounds(mx):
    data = mx.array(np.array([1.0], dtype=np.float32))
    row = mx.array(np.array([2], dtype=np.int32))
    col = mx.array(np.array([0], dtype=np.int32))

    with pytest.raises(ValueError, match="row coordinates"):
        ms.coo_array((data, (row, col)), shape=(2, 3), validate="full")


def test_shape_must_be_rank_two(mx):
    data = mx.array(np.array([], dtype=np.float32))
    indices = mx.array(np.array([], dtype=np.int32))
    indptr = mx.array(np.array([0], dtype=np.int32))

    with pytest.raises(ValueError, match="rank-2"):
        ms.csr_array((data, indices, indptr), shape=(0,))


def test_eye_constructs_shifted_diagonal(mx):
    csr = ms.eye(3, 5, k=1)

    expected = np.zeros((3, 5), dtype=np.float32)
    expected[0, 1] = 1.0
    expected[1, 2] = 1.0
    expected[2, 3] = 1.0
    np.testing.assert_allclose(to_numpy(csr.todense()), expected)
    assert csr.has_canonical_format


def test_diags_constructs_multiple_offsets(mx):
    csr = ms.diags(
        [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])],
        offsets=[0, 2],
        shape=(3, 5),
    )

    expected = np.zeros((3, 5), dtype=np.float32)
    expected[0, 0] = 1.0
    expected[1, 1] = 2.0
    expected[2, 2] = 3.0
    expected[0, 2] = 4.0
    expected[1, 3] = 5.0
    np.testing.assert_allclose(to_numpy(csr.todense()), expected)


def test_fromdense_roundtrip_and_threshold(mx):
    dense_np = np.array(
        [[0.0, 1e-4, 2.0], [-3.0, 0.0, 4e-5]],
        dtype=np.float32,
    )
    csr = ms.fromdense(mx.array(dense_np), threshold=1e-3)

    expected = np.array([[0.0, 0.0, 2.0], [-3.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(to_numpy(csr.todense()), expected)
    assert csr.sorted_indices
    assert csr.has_canonical_format
