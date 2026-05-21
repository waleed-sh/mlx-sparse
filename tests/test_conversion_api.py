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

import mlx_sparse as ms
from mlx_sparse._host import to_numpy


def test_from_scipy_csr_sums_duplicates_and_preserves_complex(mx, scipy_sparse):
    data = np.array([1.0 + 2.0j, 3.0 - 4.0j, -2.0 + 0.5j], dtype=np.complex64)
    row = np.array([0, 0, 1], dtype=np.int32)
    col = np.array([2, 2, 0], dtype=np.int32)
    sp = scipy_sparse.coo_matrix((data, (row, col)), shape=(3, 4))

    csr = ms.from_scipy(sp, dtype=mx.complex64, index_dtype=mx.int64)

    expected = sp.tocsr()
    expected.sum_duplicates()
    np.testing.assert_allclose(to_numpy(csr.todense()), expected.toarray())
    assert csr.dtype == mx.complex64
    assert csr.index_dtype == mx.int64
    assert csr.sorted_indices
    assert csr.has_canonical_format


def test_from_scipy_coo_output(mx, scipy_sparse):
    sp = scipy_sparse.dok_matrix((3, 3), dtype=np.float32)
    sp[2, 0] = 2.0
    sp[0, 1] = -1.0

    coo = ms.from_scipy(sp, format="coo")

    assert isinstance(coo, ms.COOArray)
    np.testing.assert_allclose(to_numpy(coo.todense()), sp.toarray())
    assert coo.has_canonical_format


def test_asarray_accepts_existing_sparse_scipy_and_dense(mx, scipy_sparse):
    dense_np = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    dense_csr = ms.asarray(dense_np)
    scipy_csr = ms.asarray(scipy_sparse.csr_matrix(dense_np))

    np.testing.assert_allclose(to_numpy(dense_csr.todense()), dense_np)
    np.testing.assert_allclose(to_numpy(scipy_csr.todense()), dense_np)
    assert ms.asarray(dense_csr) is dense_csr

    coo = ms.coo_array(
        (
            mx.array(np.array([3.0], dtype=np.float32)),
            (
                mx.array(np.array([1], dtype=np.int32)),
                mx.array(np.array([0], dtype=np.int32)),
            ),
        ),
        shape=(2, 2),
    )
    assert isinstance(ms.asarray(coo), ms.CSRArray)


def test_from_numpy_alias_supports_dtype_and_threshold(mx):
    dense_np = np.array([[1.0, 1e-5], [0.0, -2.0]], dtype=np.float32)

    csr = ms.from_numpy(dense_np, threshold=1e-4, dtype=mx.float16)

    expected = np.array([[1.0, 0.0], [0.0, -2.0]], dtype=np.float16)
    np.testing.assert_allclose(to_numpy(csr.todense()), expected)
    assert csr.dtype == mx.float16
