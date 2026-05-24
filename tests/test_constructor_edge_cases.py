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


def test_eye_handles_offsets_empty_outputs_and_invalid_dtypes(mx):
    lower = ms.eye(3, 4, k=-1, dtype=mx.complex64, index_dtype=mx.int64)
    expected_lower = np.zeros((3, 4), dtype=np.complex64)
    expected_lower[1, 0] = 1.0
    expected_lower[2, 1] = 1.0

    assert lower.dtype == mx.complex64
    assert lower.index_dtype == mx.int64
    np.testing.assert_allclose(to_numpy(lower.todense()), expected_lower)

    empty = ms.eye(2, 2, k=3)
    assert empty.nnz == 0
    np.testing.assert_allclose(to_numpy(empty.todense()), np.zeros((2, 2)))

    with pytest.raises(TypeError, match="dtype must be one"):
        ms.eye(1, dtype=mx.int32)
    with pytest.raises(TypeError, match="index_dtype"):
        ms.eye(1, index_dtype=mx.float32)


def test_diags_accepts_scalar_vectors_matrix_and_empty_inputs(mx):
    scalar = ms.diags(mx.array(5.0), shape=(1, 1))
    np.testing.assert_allclose(to_numpy(scalar.todense()), [[5.0]])

    vector = ms.diags(mx.array(np.array([1.0, 2.0], dtype=np.float32)), offsets=-1)
    expected_vector = np.zeros((3, 3), dtype=np.float32)
    expected_vector[1, 0] = 1.0
    expected_vector[2, 1] = 2.0
    np.testing.assert_allclose(to_numpy(vector.todense()), expected_vector)

    matrix_rows = ms.diags(
        mx.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        offsets=[0, 1],
        shape=(3, 3),
    )
    expected_matrix_rows = np.zeros((3, 3), dtype=np.float32)
    expected_matrix_rows[0, 0] = 1.0
    expected_matrix_rows[1, 1] = 2.0
    expected_matrix_rows[0, 1] = 3.0
    expected_matrix_rows[1, 2] = 4.0
    np.testing.assert_allclose(to_numpy(matrix_rows.todense()), expected_matrix_rows)

    sequence_scalars = ms.diags([1.0, 2.0, 3.0], offsets=0, shape=(3, 3))
    np.testing.assert_allclose(
        to_numpy(sequence_scalars.todense()), np.eye(3) * [1, 2, 3]
    )

    empty = ms.diags(
        [], offsets=[], shape=(2, 3), dtype=mx.float16, index_dtype=mx.int64
    )
    assert empty.nnz == 0
    assert empty.dtype == mx.float16
    assert empty.index_dtype == mx.int64
    np.testing.assert_allclose(to_numpy(empty.todense()), np.zeros((2, 3)))


def test_diags_rejects_inconsistent_or_impossible_inputs(mx):
    with pytest.raises(ValueError, match="same number"):
        ms.diags([np.array([1.0])], offsets=[0, 1])

    with pytest.raises(ValueError, match="repeated offsets"):
        ms.diags([np.array([1.0]), np.array([2.0])], offsets=[0, 0])

    with pytest.raises(ValueError, match="can hold at most"):
        ms.diags(np.array([1.0, 2.0]), offsets=2, shape=(2, 2))

    with pytest.raises(TypeError, match="index_dtype"):
        ms.diags(np.array([1.0]), index_dtype=mx.float32)


def test_fromdense_validation_complex_threshold_and_alias(mx):
    dense = mx.array(
        np.array(
            [[0.0 + 0.0j, 0.05 + 0.05j], [2.0 - 1.0j, 0.0 + 0.0j]],
            dtype=np.complex64,
        )
    )
    csr = ms.from_dense(dense, threshold=0.1)

    expected = np.array([[0.0, 0.0], [2.0 - 1.0j, 0.0]], dtype=np.complex64)
    np.testing.assert_allclose(to_numpy(csr.todense()), expected)
    assert csr.dtype == mx.complex64

    with pytest.raises(ValueError, match="rank-2"):
        ms.fromdense(mx.array(np.array([1.0], dtype=np.float32)))
    with pytest.raises(TypeError, match="input dtype"):
        ms.fromdense(mx.array(np.array([[1]], dtype=np.int32)))
    with pytest.raises(ValueError, match="non-negative"):
        ms.fromdense(mx.array(np.array([[1.0]], dtype=np.float32)), threshold=-1.0)


def test_from_scipy_error_paths_canonical_false_and_dtype_casts(mx, scipy_sparse):
    with pytest.raises(TypeError, match="scipy.sparse"):
        ms.from_scipy(np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="format"):
        ms.from_scipy(scipy_sparse.eye(2, dtype=np.float32), format="bsr")

    sp16 = scipy_sparse.csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32))
    csr16 = ms.from_scipy(sp16, dtype=mx.float16)
    assert csr16.dtype == mx.float16
    np.testing.assert_allclose(to_numpy(csr16.todense()), sp16.toarray())

    sp64 = scipy_sparse.csr_matrix(np.array([[0.0, 1.5]], dtype=np.float64))
    bf16 = ms.from_scipy(sp64, dtype=mx.bfloat16)
    assert bf16.dtype == mx.bfloat16
    np.testing.assert_allclose(to_numpy(bf16.todense()), sp64.toarray(), atol=1e-2)

    duplicate_coo = scipy_sparse.coo_matrix(
        (
            np.array([1.0, 2.0], dtype=np.float32),
            (np.array([0, 0]), np.array([1, 1])),
        ),
        shape=(1, 3),
    )
    coo = ms.from_scipy(duplicate_coo, format="coo", canonical=False)
    assert isinstance(coo, ms.COOArray)
    assert coo.nnz == 2
    assert not coo.has_canonical_format


def test_asarray_casts_existing_sparse_inputs(mx):
    csr = ms.eye(2, dtype=mx.float32)
    cast_csr = ms.asarray(csr, dtype=mx.float16)

    assert cast_csr is not csr
    assert cast_csr.dtype == mx.float16
    np.testing.assert_allclose(
        to_numpy(cast_csr.todense()), np.eye(2, dtype=np.float16)
    )

    coo = ms.coo_array(
        (
            mx.array(np.array([1.0, 2.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 1], dtype=np.int32)),
                mx.array(np.array([1, 0], dtype=np.int32)),
            ),
        ),
        shape=(2, 2),
    )
    cast_from_coo = ms.asarray(coo, dtype=mx.complex64)

    assert isinstance(cast_from_coo, ms.CSRArray)
    assert cast_from_coo.dtype == mx.complex64
    np.testing.assert_allclose(
        to_numpy(cast_from_coo.todense()),
        np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.complex64),
    )
