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


def _index_dtype(mx, name):
    if name == "int32":
        return mx.int32
    if name == "int64":
        return mx.int64
    raise AssertionError(name)


def _value_dtype(mx, name):
    if name == "float32":
        return mx.float32
    if name == "complex64":
        return mx.complex64
    raise AssertionError(name)


def _lhs_coo(mx, *, dtype, index_dtype):
    values = np.array([2.0, -1.0, 4.0, 3.0, -2.0], dtype=np.float32)
    if dtype == mx.complex64:
        values = values.astype(np.complex64) + 1j * np.array(
            [0.5, -0.25, 1.0, 0.0, -0.75], dtype=np.float32
        )
    return ms.coo_array(
        (
            mx.array(values, dtype=dtype),
            (
                mx.array(np.array([0, 0, 1, 2, 2], dtype=np.int32), dtype=index_dtype),
                mx.array(np.array([1, 3, 0, 2, 3], dtype=np.int32), dtype=index_dtype),
            ),
        ),
        shape=(3, 4),
        canonical=True,
    )


def _rhs_coo(mx, *, dtype, index_dtype):
    values = np.array([5.0, -3.0, 1.0, 6.0, -4.0], dtype=np.float32)
    if dtype == mx.complex64:
        values = values.astype(np.complex64) + 1j * np.array(
            [-1.0, 0.25, 0.5, -0.5, 0.75], dtype=np.float32
        )
    return ms.coo_array(
        (
            mx.array(values, dtype=dtype),
            (
                mx.array(np.array([0, 1, 2, 3, 3], dtype=np.int32), dtype=index_dtype),
                mx.array(np.array([2, 0, 1, 1, 2], dtype=np.int32), dtype=index_dtype),
            ),
        ),
        shape=(4, 3),
        canonical=True,
    )


def _as_format(array, format_name):
    if format_name == "coo":
        return array
    if format_name == "csr":
        return array.tocsr(canonical=True)
    if format_name == "csc":
        return array.tocsc(canonical=True)
    raise AssertionError(format_name)


def _expected_left_type(format_name):
    if format_name == "coo":
        return ms.COOArray
    if format_name == "csr":
        return ms.CSRArray
    if format_name == "csc":
        return ms.CSCArray
    raise AssertionError(format_name)


def _assert_canonical_for_format(array):
    assert array.has_canonical_format
    if isinstance(array, (ms.CSRArray, ms.CSCArray)):
        assert array.sorted_indices


@pytest.mark.parametrize("lhs_format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("rhs_format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("dtype_name", ["float32", "complex64"])
@pytest.mark.parametrize("index_dtype_name", ["int32", "int64"])
def test_sparse_sparse_matmul_all_format_pairs_match_dense_reference(
    lhs_format,
    rhs_format,
    dtype_name,
    index_dtype_name,
    mx,
    to_numpy,
):
    dtype = _value_dtype(mx, dtype_name)
    index_dtype = _index_dtype(mx, index_dtype_name)
    lhs = _as_format(_lhs_coo(mx, dtype=dtype, index_dtype=index_dtype), lhs_format)
    rhs = _as_format(_rhs_coo(mx, dtype=dtype, index_dtype=index_dtype), rhs_format)

    out = lhs @ rhs

    assert isinstance(out, _expected_left_type(lhs_format))
    assert out.shape == (lhs.shape[0], rhs.shape[1])
    assert out.dtype == dtype
    assert out.index_dtype == index_dtype
    _assert_canonical_for_format(out)
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        to_numpy(lhs.todense()) @ to_numpy(rhs.todense()),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("lhs_format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("rhs_format", ["coo", "csr", "csc"])
def test_sparse_sparse_matmul_mixed_index_dtypes_promote_to_int64(
    lhs_format,
    rhs_format,
    mx,
    to_numpy,
):
    lhs = _as_format(_lhs_coo(mx, dtype=mx.float32, index_dtype=mx.int32), lhs_format)
    rhs = _as_format(_rhs_coo(mx, dtype=mx.float32, index_dtype=mx.int64), rhs_format)

    out = lhs @ rhs

    assert isinstance(out, _expected_left_type(lhs_format))
    assert out.index_dtype == mx.int64
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        to_numpy(lhs.todense()) @ to_numpy(rhs.todense()),
    )


@pytest.mark.parametrize("lhs_format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("rhs_format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("out_format", [None, "coo", "csr", "csc"])
def test_block_array_all_sparse_format_pairs_match_dense_reference(
    lhs_format,
    rhs_format,
    out_format,
    mx,
    to_numpy,
):
    lhs = _as_format(
        _lhs_coo(mx, dtype=mx.float32, index_dtype=mx.int32),
        lhs_format,
    )
    rhs = _as_format(
        _rhs_coo(mx, dtype=mx.float32, index_dtype=mx.int32),
        rhs_format,
    )

    out = ms.block_array([[lhs, None], [None, rhs]], format=out_format)
    expected = np.block(
        [
            [
                to_numpy(lhs.todense()),
                np.zeros((lhs.shape[0], rhs.shape[1]), dtype=np.float32),
            ],
            [
                np.zeros((rhs.shape[0], lhs.shape[1]), dtype=np.float32),
                to_numpy(rhs.todense()),
            ],
        ]
    )

    if out_format in (None, "coo"):
        assert isinstance(out, ms.COOArray)
    elif out_format == "csr":
        assert isinstance(out, ms.CSRArray)
    elif out_format == "csc":
        assert isinstance(out, ms.CSCArray)
    else:
        raise AssertionError(out_format)
    np.testing.assert_allclose(to_numpy(out.todense()), expected)
