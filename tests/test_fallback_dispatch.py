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
import mlx_sparse._construct as construct
import mlx_sparse._fallback as fallback
import mlx_sparse._native as native
from mlx_sparse._host import to_numpy


def _sample_csr(mx, *, index_dtype=np.int32):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=index_dtype))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=index_dtype))
    return data, indices, indptr, (3, 4)


def test_native_wrappers_fall_back_to_python_kernels(monkeypatch, mx):
    monkeypatch.setattr(native, "extension", lambda: None)
    data, indices, indptr, shape = _sample_csr(mx)

    dense = np.array(
        [[2.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 5.0]],
        dtype=np.float32,
    )
    x = mx.array(np.array([3.0, 10.0, 7.0, -2.0], dtype=np.float32))
    rhs = mx.array(np.arange(20, dtype=np.float32).reshape(4, 5) / 10.0)
    row_rhs = mx.array(np.arange(6, dtype=np.float32).reshape(3, 2) / 10.0)

    np.testing.assert_allclose(to_numpy(native.identity_like(x)), to_numpy(x))
    np.testing.assert_allclose(
        to_numpy(native.csr_todense(data, indices, indptr, shape)),
        dense,
    )
    np.testing.assert_allclose(
        to_numpy(native.csr_matvec(data, indices, indptr, x, shape)),
        dense @ to_numpy(x),
    )
    np.testing.assert_allclose(
        to_numpy(
            native.csr_matvec_transpose(data, indices, indptr, row_rhs[:, 0], shape)
        ),
        dense.T @ to_numpy(row_rhs[:, 0]),
    )
    np.testing.assert_allclose(
        to_numpy(native.csr_matmul(data, indices, indptr, rhs, shape)),
        dense @ to_numpy(rhs),
    )
    np.testing.assert_allclose(
        to_numpy(native.csr_matmul_transpose(data, indices, indptr, row_rhs, shape)),
        dense.T @ to_numpy(row_rhs),
    )

    dup_data = mx.array(np.array([1.0, 2.0, -1.0], dtype=np.float32))
    dup_indices = mx.array(np.array([0, 0, 2], dtype=np.int32))
    dup_indptr = mx.array(np.array([0, 3], dtype=np.int32))
    summed_data, summed_indices, summed_indptr = native.csr_sum_duplicates(
        dup_data, dup_indices, dup_indptr
    )
    np.testing.assert_allclose(to_numpy(summed_data), [3.0, -1.0])
    np.testing.assert_array_equal(to_numpy(summed_indices), [0, 2])
    np.testing.assert_array_equal(to_numpy(summed_indptr), [0, 2])

    dense_mx = mx.array(np.array([[0.0, 2.0], [3.0, 1e-4]], dtype=np.float32))
    fd_data, fd_indices, fd_indptr = native.csr_fromdense(
        dense_mx, index_dtype=mx.int64, threshold=1e-3
    )
    np.testing.assert_allclose(to_numpy(fd_data), [2.0, 3.0])
    np.testing.assert_array_equal(to_numpy(fd_indices), np.array([1, 0]))
    np.testing.assert_array_equal(to_numpy(fd_indptr), np.array([0, 1, 2]))


def test_fallback_coo_conversion_transpose_and_sort(monkeypatch, mx):
    monkeypatch.setattr(native, "extension", lambda: None)

    data = mx.array(np.array([-1.0, 2.0, 5.0, 4.0], dtype=np.float32))
    row = mx.array(np.array([0, 0, 2, 2], dtype=np.int32))
    col = mx.array(np.array([2, 0, 3, 1], dtype=np.int32))
    csr_data, csr_indices, csr_indptr = native.coo_tocsr(data, row, col, (3, 4))
    csr = ms.csr_array((csr_data, csr_indices, csr_indptr), shape=(3, 4))

    expected = np.array(
        [[2.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 5.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(to_numpy(csr.todense()), expected)

    transposed = native.csr_transpose(csr.data, csr.indices, csr.indptr, csr.shape)
    np.testing.assert_allclose(
        to_numpy(ms.csr_array(transposed, shape=(4, 3)).todense()),
        expected.T,
    )

    unsorted_data = mx.array(np.array([9.0, 2.0, 3.0], dtype=np.float32))
    unsorted_indices = mx.array(np.array([2, 0, 1], dtype=np.int32))
    unsorted_indptr = mx.array(np.array([0, 3], dtype=np.int32))
    sorted_data, sorted_indices, sorted_indptr = native.csr_sort_indices(
        unsorted_data, unsorted_indices, unsorted_indptr
    )

    np.testing.assert_allclose(to_numpy(sorted_data), [2.0, 3.0, 9.0])
    np.testing.assert_array_equal(to_numpy(sorted_indices), [0, 1, 2])
    assert sorted_indptr is unsorted_indptr


def test_sparse_sparse_fallback_errors_and_empty_mixed_index_output(mx):
    lhs = ms.csr_array(
        (
            mx.array(np.array([1.0, -1.0], dtype=np.float32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 2], dtype=np.int32)),
        ),
        shape=(1, 2),
        sorted_indices=True,
        canonical=True,
    )
    rhs_cancel = ms.csr_array(
        (
            mx.array(np.array([2.0, 2.0], dtype=np.float32)),
            mx.array(np.array([0, 0], dtype=np.int64)),
            mx.array(np.array([0, 1, 2], dtype=np.int64)),
        ),
        shape=(2, 1),
        sorted_indices=True,
        canonical=True,
    )

    out = ms.csr_matmat(lhs, rhs_cancel)

    assert out.shape == (1, 1)
    assert out.nnz == 0
    assert out.index_dtype == mx.int64
    np.testing.assert_allclose(
        to_numpy(out.todense()), np.zeros((1, 1), dtype=np.float32)
    )

    bad_shape_rhs = ms.eye(3, 1)
    with pytest.raises(ValueError, match="dimension mismatch"):
        ms.csr_matmat(lhs, bad_shape_rhs)

    bad_dtype_rhs = ms.csr_array(
        (
            mx.array(np.array([1.0], dtype=np.float16)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1, 1], dtype=np.int32)),
        ),
        shape=(2, 1),
    )
    with pytest.raises(TypeError, match="matching value dtypes"):
        ms.csr_matmat(lhs, bad_dtype_rhs)


def test_sparse_sparse_matmul_uses_native_extension(monkeypatch, mx):
    if not ms.is_available():
        pytest.skip("native extension is required for native dispatch check")

    def fail_if_used(*args, **kwargs):
        raise AssertionError("csr_matmat should not use the NumPy fallback")

    monkeypatch.setattr(fallback, "csr_matmat", fail_if_used)

    lhs = ms.csr_array(
        (
            mx.array(np.array([1.0, 2.0], dtype=np.float32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 2], dtype=np.int32)),
        ),
        shape=(1, 2),
        sorted_indices=True,
        canonical=True,
    )
    rhs = ms.csr_array(
        (
            mx.array(np.array([3.0, 4.0], dtype=np.float32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 1, 2], dtype=np.int32)),
        ),
        shape=(2, 2),
        sorted_indices=True,
        canonical=True,
    )

    out = ms.csr_matmat(lhs, rhs)

    assert out.has_canonical_format
    np.testing.assert_allclose(to_numpy(out.todense()), [[3.0, 8.0]])


def test_sum_duplicates_uses_native_extension(monkeypatch, mx):
    if not ms.is_available():
        pytest.skip("native extension is required for native dispatch check")

    def fail_if_used(*args, **kwargs):
        raise AssertionError("sum_duplicates should not use the NumPy fallback")

    monkeypatch.setattr(fallback, "sum_csr_duplicates", fail_if_used)

    csr = ms.csr_array(
        (
            mx.array(np.array([1.0, 2.0, -1.0], dtype=np.float32)),
            mx.array(np.array([2, 0, 2], dtype=np.int32)),
            mx.array(np.array([0, 3], dtype=np.int32)),
        ),
        shape=(1, 3),
    )

    out = csr.canonicalize()

    assert out.has_canonical_format
    np.testing.assert_allclose(to_numpy(out.data), [2.0, 0.0])
    np.testing.assert_array_equal(to_numpy(out.indices), [0, 2])
    np.testing.assert_allclose(to_numpy(out.todense()), [[2.0, 0.0, 0.0]])


def test_fromdense_uses_native_extension(monkeypatch, mx):
    if not ms.is_available():
        pytest.skip("native extension is required for native dispatch check")

    def fail_if_used(*args, **kwargs):
        raise AssertionError("fromdense should not use a NumPy host fallback")

    monkeypatch.setattr(fallback, "fromdense", fail_if_used)
    monkeypatch.setattr(construct, "to_numpy", fail_if_used)

    dense = mx.array(np.array([[0.0, 2.0], [3.0, 1e-4]], dtype=np.float32))
    out = ms.fromdense(dense, threshold=1e-3, index_dtype=mx.int64)

    assert out.has_canonical_format
    assert out.index_dtype == mx.int64
    np.testing.assert_allclose(to_numpy(out.data), [2.0, 3.0])
    np.testing.assert_array_equal(to_numpy(out.indices), np.array([1, 0]))
    np.testing.assert_array_equal(to_numpy(out.indptr), np.array([0, 1, 2]))
