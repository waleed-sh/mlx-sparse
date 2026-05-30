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


def _value_array(mx, values: np.ndarray, dtype_name: str):
    if dtype_name == "complex64":
        data = values.astype(np.complex64) + 1j * np.flip(values).astype(np.complex64)
        return mx.array(data)
    return mx.array(values).astype(getattr(mx, dtype_name))


def _assert_csr_exact(left, right) -> None:
    np.testing.assert_array_equal(to_numpy(left.indptr), to_numpy(right.indptr))
    np.testing.assert_array_equal(to_numpy(left.indices), to_numpy(right.indices))
    np.testing.assert_allclose(to_numpy(left.data), to_numpy(right.data), atol=0.0)


def _assert_csc_exact(left, right) -> None:
    np.testing.assert_array_equal(to_numpy(left.indptr), to_numpy(right.indptr))
    np.testing.assert_array_equal(to_numpy(left.indices), to_numpy(right.indices))
    np.testing.assert_allclose(to_numpy(left.data), to_numpy(right.data), atol=0.0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16", "complex64"])
def test_fromdense_cpu_host_assembly_matches_serial_and_expected(
    mx, dtype_name, index_dtype
):
    base = np.array(
        [
            [0.0, 0.2, -0.75, 0.0, 1.5],
            [1.0e-4, -2.0, 0.0, 3.0, 0.25],
            [0.0, 0.0, 4.0, -1.0e-5, 0.0],
            [5.0, -0.1, 0.0, 0.4, -6.0],
        ],
        dtype=np.float32,
    )
    threshold = 0.15
    dense = _value_array(mx, base, dtype_name)
    index_mx_dtype = mx.int32 if index_dtype == np.int32 else mx.int64

    with ms.runtime.context(n_threads=1):
        serial = ms.fromdense(dense, threshold=threshold, index_dtype=index_mx_dtype)
    with ms.runtime.context(n_threads=3):
        parallel = ms.fromdense(dense, threshold=threshold, index_dtype=index_mx_dtype)

    _assert_csr_exact(parallel, serial)

    dense_np = to_numpy(dense)
    mask = np.abs(dense_np) > threshold
    expected_indices: list[int] = []
    expected_indptr = [0]
    for row in range(dense_np.shape[0]):
        cols = np.flatnonzero(mask[row])
        expected_indices.extend(cols.tolist())
        expected_indptr.append(len(expected_indices))

    np.testing.assert_array_equal(to_numpy(serial.indptr), np.array(expected_indptr))
    np.testing.assert_array_equal(to_numpy(serial.indices), np.array(expected_indices))
    expected_dense = np.where(mask, dense_np, np.zeros((), dtype=dense_np.dtype))
    np.testing.assert_allclose(
        to_numpy(serial.todense()), expected_dense, rtol=2e-2, atol=2e-2
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_fromdense_cpu_host_assembly_empty_shapes(mx, shape, index_dtype):
    dense = mx.zeros(shape, dtype=mx.float32)
    index_mx_dtype = mx.int32 if index_dtype == np.int32 else mx.int64

    with ms.runtime.context(n_threads=1):
        serial = ms.fromdense(dense, index_dtype=index_mx_dtype)
    with ms.runtime.context(n_threads=3):
        parallel = ms.fromdense(dense, index_dtype=index_mx_dtype)

    _assert_csr_exact(parallel, serial)
    np.testing.assert_array_equal(
        to_numpy(serial.indptr), np.zeros(shape[0] + 1, dtype=index_dtype)
    )
    assert serial.nnz == 0
    assert serial.shape == shape


@pytest.mark.cpu_only
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16", "complex64"])
def test_sum_duplicates_cpu_staged_path_remains_serial_parallel_consistent(
    mx, dtype_name, index_dtype
):
    data_np = np.array(
        [1.0, -1.0, 2.0, 3.0, -0.5, 0.5, 4.0, -2.0, 1.25, -1.25, 6.0],
        dtype=np.float32,
    )
    data = _value_array(mx, data_np, dtype_name)
    index_mx_dtype = mx.int32 if index_dtype == np.int32 else mx.int64
    indices = mx.array(
        np.array([0, 0, 2, 1, 1, 1, 4, 3, 3, 3, 4], dtype=index_dtype),
        dtype=index_mx_dtype,
    )
    indptr = mx.array(
        np.array([0, 3, 3, 7, 11], dtype=index_dtype), dtype=index_mx_dtype
    )

    csr = ms.csr_array(
        (data, indices, indptr),
        shape=(4, 5),
        sorted_indices=True,
        canonical=False,
    )

    with ms.runtime.context(n_threads=1):
        csr_serial = csr.sum_duplicates()
    with ms.runtime.context(n_threads=3):
        csr_parallel = csr.sum_duplicates()

    _assert_csr_exact(csr_parallel, csr_serial)
    np.testing.assert_array_equal(
        to_numpy(csr_serial.indptr), np.array([0, 2, 2, 4, 6], dtype=index_dtype)
    )
    np.testing.assert_array_equal(
        to_numpy(csr_serial.indices), np.array([0, 2, 1, 4, 3, 4], dtype=index_dtype)
    )
    np.testing.assert_allclose(
        to_numpy(csr_parallel.todense()), to_numpy(csr.todense())
    )

    csc = csr.tocsc(canonical=False)
    with ms.runtime.context(n_threads=1):
        csc_serial = csc.sum_duplicates()
    with ms.runtime.context(n_threads=3):
        csc_parallel = csc.sum_duplicates()

    _assert_csc_exact(csc_parallel, csc_serial)
    np.testing.assert_allclose(
        to_numpy(csc_parallel.todense()), to_numpy(csc.todense())
    )
