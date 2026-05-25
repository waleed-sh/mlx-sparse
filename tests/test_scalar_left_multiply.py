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


def _values(dtype_name: str) -> np.ndarray:
    data = np.array([2.0, -1.5, 0.25, 3.0, -4.0], dtype=np.float32)
    if dtype_name == "complex64":
        return data.astype(np.complex64) + 1j * np.array(
            [0.5, -2.0, 1.25, 0.0, -0.75], dtype=np.float32
        )
    return data


def _sample_sparse(mx, format_name: str, dtype_name: str, index_dtype) -> object:
    dtype = getattr(mx, dtype_name)
    data = mx.array(_values(dtype_name)).astype(dtype)

    if format_name == "csr":
        return ms.csr_array(
            (
                data,
                mx.array(np.array([3, 0, 2, 2, 1], dtype=index_dtype)),
                mx.array(np.array([0, 2, 2, 4, 5], dtype=index_dtype)),
            ),
            shape=(4, 4),
            sorted_indices=False,
            canonical=False,
        )

    if format_name == "csc":
        return ms.csc_array(
            (
                data,
                mx.array(np.array([3, 0, 2, 2, 1], dtype=index_dtype)),
                mx.array(np.array([0, 2, 2, 4, 5], dtype=index_dtype)),
            ),
            shape=(4, 4),
            sorted_indices=False,
            canonical=False,
        )

    if format_name == "coo":
        return ms.coo_array(
            (
                data,
                (
                    mx.array(np.array([2, 0, 2, 3, 1], dtype=index_dtype)),
                    mx.array(np.array([2, 3, 2, 0, 2], dtype=index_dtype)),
                ),
            ),
            shape=(4, 4),
            canonical=False,
        )

    raise AssertionError(f"Unknown sparse format {format_name!r}")


def _assert_structure_preserved(lhs, out):
    assert type(out) is type(lhs)
    assert out.shape == lhs.shape
    assert out.nnz == lhs.nnz
    assert out.has_canonical_format == lhs.has_canonical_format

    if isinstance(lhs, ms.COOArray):
        np.testing.assert_array_equal(to_numpy(out.row), to_numpy(lhs.row))
        np.testing.assert_array_equal(to_numpy(out.col), to_numpy(lhs.col))
    else:
        assert out.sorted_indices == lhs.sorted_indices
        np.testing.assert_array_equal(to_numpy(out.indices), to_numpy(lhs.indices))
        np.testing.assert_array_equal(to_numpy(out.indptr), to_numpy(lhs.indptr))


@pytest.mark.parametrize("format_name", ["csr", "csc", "coo"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("scalar", [2, -0.5, np.float32(1.25)])
def test_number_left_multiply_sparse_matches_dense_and_preserves_structure(
    mx, format_name, index_dtype, scalar
):
    sparse = _sample_sparse(mx, format_name, "float32", index_dtype)
    before_dense = to_numpy(sparse.todense())
    before_data = to_numpy(sparse.data)

    out = scalar * sparse

    _assert_structure_preserved(sparse, out)
    np.testing.assert_allclose(to_numpy(out.data), scalar * before_data)
    np.testing.assert_allclose(to_numpy(out.todense()), scalar * before_dense)
    np.testing.assert_allclose(to_numpy(sparse.data), before_data)
    np.testing.assert_allclose(to_numpy(sparse.todense()), before_dense)


@pytest.mark.parametrize("format_name", ["csr", "csc", "coo"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_complex_number_left_multiply_sparse_matches_dense(
    mx, format_name, index_dtype
):
    sparse = _sample_sparse(mx, format_name, "complex64", index_dtype)
    scalar = np.complex64(-1.25 + 0.5j)
    before_dense = to_numpy(sparse.todense())
    before_data = to_numpy(sparse.data)

    out = scalar * sparse

    _assert_structure_preserved(sparse, out)
    np.testing.assert_allclose(to_numpy(out.data), scalar * before_data)
    np.testing.assert_allclose(to_numpy(out.todense()), scalar * before_dense)
    np.testing.assert_allclose(to_numpy(sparse.data), before_data)
    np.testing.assert_allclose(to_numpy(sparse.todense()), before_dense)


@pytest.mark.parametrize("format_name", ["csr", "csc", "coo"])
def test_left_multiply_rejects_non_numeric_scalars(mx, format_name):
    sparse = _sample_sparse(mx, format_name, "float32", np.int32)

    with pytest.raises(TypeError, match="Expected a number"):
        sparse.__rmul__("not-a-number")
