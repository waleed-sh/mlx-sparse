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
from mlx_sparse._validation import ensure_mx_array


def test_ensure_mx_array_casts_existing_arrays_and_python_inputs(mx):
    original = mx.array(np.array([1.0], dtype=np.float32))

    cast = ensure_mx_array(original, dtype=mx.float16)
    created = ensure_mx_array([1.0, 2.0], dtype=mx.float32)

    assert cast.dtype == mx.float16
    assert created.dtype == mx.float32


def test_csr_metadata_validation_errors_are_specific(mx):
    indices = mx.array(np.array([0], dtype=np.int32))
    indptr = mx.array(np.array([0, 1], dtype=np.int32))

    with pytest.raises(ValueError, match="rank-1"):
        ms.csr_array(
            (mx.array(np.ones((1, 1), dtype=np.float32)), indices, indptr), (1, 1)
        )

    with pytest.raises(TypeError, match="data dtype"):
        ms.csr_array(
            (
                mx.array(np.array([1], dtype=np.int32)),
                indices,
                indptr,
            ),
            shape=(1, 1),
        )

    with pytest.raises(TypeError, match="indices"):
        ms.csr_array(
            (
                mx.array(np.array([1.0], dtype=np.float32)),
                mx.array(np.array([0.0], dtype=np.float32)),
                indptr,
            ),
            shape=(1, 1),
        )

    with pytest.raises(TypeError, match="same dtype"):
        ms.csr_array(
            (
                mx.array(np.array([1.0], dtype=np.float32)),
                indices,
                mx.array(np.array([0, 1], dtype=np.int64)),
            ),
            shape=(1, 1),
        )

    with pytest.raises(ValueError, match="same length"):
        ms.csr_array(
            (
                mx.array(np.array([1.0, 2.0], dtype=np.float32)),
                indices,
                indptr,
            ),
            shape=(1, 1),
        )


def test_csr_full_validation_rejects_pointer_and_index_errors(mx):
    data = mx.array(np.array([1.0], dtype=np.float32))
    indices = mx.array(np.array([0], dtype=np.int32))

    with pytest.raises(ValueError, match=r"indptr\[0\]"):
        ms.csr_array(
            (data, indices, mx.array(np.array([1, 1], dtype=np.int32))),
            shape=(1, 1),
            validate="full",
        )

    with pytest.raises(ValueError, match=r"indptr\[-1\]"):
        ms.csr_array(
            (data, indices, mx.array(np.array([0, 0], dtype=np.int32))),
            shape=(1, 1),
            validate="full",
        )

    with pytest.raises(ValueError, match="in bounds"):
        ms.csr_array(
            (
                mx.array(np.array([1.0, 2.0], dtype=np.float32)),
                mx.array(np.array([-1, 3], dtype=np.int32)),
                mx.array(np.array([0, 2], dtype=np.int32)),
            ),
            shape=(1, 3),
            validate="full",
        )


def test_coo_metadata_and_value_validation_errors(mx):
    data = mx.array(np.array([1.0], dtype=np.float32))
    row = mx.array(np.array([0], dtype=np.int32))
    col = mx.array(np.array([0], dtype=np.int32))

    with pytest.raises(TypeError, match="data dtype"):
        ms.coo_array(
            (
                mx.array(np.array([1], dtype=np.int32)),
                (row, col),
            ),
            shape=(1, 1),
        )

    with pytest.raises(TypeError, match="row and col dtypes"):
        ms.coo_array(
            (
                data,
                (row, mx.array(np.array([0], dtype=np.int64))),
            ),
            shape=(1, 1),
        )

    with pytest.raises(ValueError, match="same length"):
        ms.coo_array(
            (
                mx.array(np.array([1.0, 2.0], dtype=np.float32)),
                (row, col),
            ),
            shape=(1, 1),
        )

    with pytest.raises(ValueError, match="col coordinates"):
        ms.coo_array(
            (
                data,
                (row, mx.array(np.array([2], dtype=np.int32))),
            ),
            shape=(1, 2),
            validate="full",
        )


def test_validate_false_allows_deferred_structural_checks(mx):
    csr = ms.csr_array(
        (
            mx.array(np.ones((1, 1), dtype=np.float32)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
        ),
        shape=(1, 1),
        validate=False,
    )
    coo = ms.coo_array(
        (
            mx.array(np.ones((1, 1), dtype=np.float32)),
            (
                mx.array(np.array([0], dtype=np.int32)),
                mx.array(np.array([0], dtype=np.int32)),
            ),
        ),
        shape=(1, 1),
        validate=False,
    )

    assert csr.data.ndim == 2
    assert coo.data.ndim == 2
