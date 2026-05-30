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
import mlx_sparse._native as native
from mlx_sparse._ext_loader import extension
from mlx_sparse._host import to_numpy


def _sample_csr(mx, index_dtype=np.int32):
    shape = (7, 9)
    indptr = np.array([0, 5, 5, 8, 16, 17, 23, 25], dtype=index_dtype)
    indices = np.array(
        [
            5,
            1,
            1,
            3,
            0,
            2,
            2,
            4,
            8,
            0,
            3,
            3,
            3,
            4,
            7,
            1,
            6,
            2,
            2,
            2,
            5,
            8,
            0,
            4,
            4,
        ],
        dtype=index_dtype,
    )
    data = np.array(
        [
            2.0,
            -1.0,
            0.5,
            4.0,
            -3.0,
            1.25,
            -0.25,
            3.0,
            -1.0,
            2.5,
            1.0,
            -0.5,
            0.75,
            4.5,
            -2.0,
            1.0,
            -3.5,
            2.0,
            -1.0,
            0.25,
            5.0,
            -2.25,
            0.75,
            1.5,
            -0.5,
        ],
        dtype=np.float32,
    )
    return ms.csr_array(
        (mx.array(data), mx.array(indices), mx.array(indptr)),
        shape=shape,
        sorted_indices=False,
        canonical=False,
    )


def _assert_csr_exact(left, right):
    np.testing.assert_array_equal(to_numpy(left.indptr), to_numpy(right.indptr))
    np.testing.assert_array_equal(to_numpy(left.indices), to_numpy(right.indices))
    np.testing.assert_allclose(to_numpy(left.data), to_numpy(right.data), atol=0.0)


def _assert_csc_exact(left, right):
    np.testing.assert_array_equal(to_numpy(left.indptr), to_numpy(right.indptr))
    np.testing.assert_array_equal(to_numpy(left.indices), to_numpy(right.indices))
    np.testing.assert_allclose(to_numpy(left.data), to_numpy(right.data), atol=0.0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_storage_aligned_reductions_and_todense_parallel_match_serial(mx, index_dtype):
    csr = _sample_csr(mx, index_dtype)
    csc = csr.tocsc(canonical=False)

    with ms.runtime.context(n_threads=1):
        serial = {
            "csr_row_sums": to_numpy(csr.row_sums()),
            "csr_row_norms": to_numpy(csr.row_norms()),
            "csr_diagonal": to_numpy(csr.diagonal()),
            "csr_todense": to_numpy(csr.todense()),
            "csc_col_sums": to_numpy(csc.col_sums()),
            "csc_col_norms": to_numpy(csc.col_norms()),
            "csc_diagonal": to_numpy(csc.diagonal()),
            "csc_todense": to_numpy(csc.todense()),
        }

    with ms.runtime.context(n_threads=3):
        parallel = {
            "csr_row_sums": to_numpy(csr.row_sums()),
            "csr_row_norms": to_numpy(csr.row_norms()),
            "csr_diagonal": to_numpy(csr.diagonal()),
            "csr_todense": to_numpy(csr.todense()),
            "csc_col_sums": to_numpy(csc.col_sums()),
            "csc_col_norms": to_numpy(csc.col_norms()),
            "csc_diagonal": to_numpy(csc.diagonal()),
            "csc_todense": to_numpy(csc.todense()),
        }

    for name, expected in serial.items():
        np.testing.assert_allclose(parallel[name], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.cpu_only
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_structural_kernels_parallel_match_serial_exactly(mx, index_dtype):
    csr = _sample_csr(mx, index_dtype)
    dense = csr.todense()
    indptr_np = to_numpy(csr.indptr)
    rows_np = np.repeat(
        np.arange(csr.shape[0], dtype=index_dtype), np.diff(indptr_np).astype(np.int64)
    )
    permutation = np.array(
        [
            16,
            0,
            7,
            21,
            4,
            12,
            1,
            24,
            8,
            3,
            19,
            11,
            2,
            15,
            10,
            5,
            23,
            14,
            6,
            18,
            9,
            20,
            13,
            22,
            17,
        ],
        dtype=np.int64,
    )
    coo = ms.coo_array(
        (
            mx.array(to_numpy(csr.data)[permutation]),
            (
                mx.array(rows_np[permutation]),
                mx.array(to_numpy(csr.indices)[permutation]),
            ),
        ),
        shape=csr.shape,
        canonical=False,
    )

    with ms.runtime.context(n_threads=1):
        coo_csr_serial = coo.tocsr(canonical=False)
        coo_csc_serial = coo.tocsc(canonical=False)
        transpose_serial = csr.transpose()
        csr_sorted_serial = csr.sort_indices()
        csr_canonical_serial = csr_sorted_serial.sum_duplicates()
        csc_serial = csr.tocsc(canonical=False)
        csc_sorted_serial = csc_serial.sort_indices()
        csc_canonical_serial = csc_sorted_serial.sum_duplicates()
        roundtrip_serial = csc_serial.tocsr(canonical=False)
        fromdense_serial = ms.fromdense(dense, index_dtype=csr.index_dtype)

    with ms.runtime.context(n_threads=3):
        coo_csr_parallel = coo.tocsr(canonical=False)
        coo_csc_parallel = coo.tocsc(canonical=False)
        transpose_parallel = csr.transpose()
        csr_sorted_parallel = csr.sort_indices()
        csr_canonical_parallel = csr_sorted_parallel.sum_duplicates()
        csc_parallel = csr.tocsc(canonical=False)
        csc_sorted_parallel = csc_parallel.sort_indices()
        csc_canonical_parallel = csc_sorted_parallel.sum_duplicates()
        roundtrip_parallel = csc_parallel.tocsr(canonical=False)
        fromdense_parallel = ms.fromdense(dense, index_dtype=csr.index_dtype)

    _assert_csr_exact(coo_csr_parallel, coo_csr_serial)
    _assert_csc_exact(coo_csc_parallel, coo_csc_serial)
    _assert_csr_exact(transpose_parallel, transpose_serial)
    _assert_csr_exact(csr_sorted_parallel, csr_sorted_serial)
    _assert_csr_exact(csr_canonical_parallel, csr_canonical_serial)
    _assert_csc_exact(csc_parallel, csc_serial)
    _assert_csc_exact(csc_sorted_parallel, csc_sorted_serial)
    _assert_csc_exact(csc_canonical_parallel, csc_canonical_serial)
    _assert_csr_exact(roundtrip_parallel, roundtrip_serial)
    _assert_csr_exact(fromdense_parallel, fromdense_serial)


@pytest.mark.cpu_only
def test_data_vjp_kernels_parallel_match_serial(mx):
    ext = extension()
    if ext is None:
        pytest.skip("native extension is required for data VJP parallel coverage")

    csr = _sample_csr(mx, np.int32).sort_indices().sum_duplicates()
    indptr_np = to_numpy(csr.indptr)
    rows_np = np.repeat(
        np.arange(csr.shape[0], dtype=np.int32), np.diff(indptr_np).astype(np.int64)
    )
    coo = ms.coo_array(
        (csr.data, (mx.array(rows_np), csr.indices)),
        shape=csr.shape,
        canonical=True,
    )
    csc = csr.tocsc(canonical=True)
    x = mx.array(np.linspace(-1.0, 1.0, csr.shape[1], dtype=np.float32))
    rhs = mx.array(
        np.arange(csr.shape[1] * 5, dtype=np.float32).reshape(csr.shape[1], 5) / 17.0
    )
    cot_vec = mx.array(np.linspace(0.25, 1.25, csr.shape[0], dtype=np.float32))
    cot_mat = mx.array(
        np.arange(csr.shape[0] * 5, dtype=np.float32).reshape(csr.shape[0], 5) / 13.0
    )

    def csr_vec_grad(values):
        array = ms.csr_array(
            (values, csr.indices, csr.indptr),
            shape=csr.shape,
            sorted_indices=True,
            canonical=True,
        )
        return mx.sum((array @ x) * cot_vec)

    def csr_mat_grad(values):
        array = ms.csr_array(
            (values, csr.indices, csr.indptr),
            shape=csr.shape,
            sorted_indices=True,
            canonical=True,
        )
        return mx.sum((array @ rhs) * cot_mat)

    with ms.runtime.context(n_threads=1):
        csr_vec_serial = to_numpy(mx.grad(csr_vec_grad)(csr.data))
        csr_mat_serial = to_numpy(mx.grad(csr_mat_grad)(csr.data))
        coo_serial = to_numpy(
            ext.coo_matmul_data_vjp(
                coo.row, coo.col, rhs, cot_mat, coo.shape[0], coo.shape[1]
            )
        )
        csc_serial = to_numpy(
            ext.csc_matmul_data_vjp(
                csc.indices, csc.indptr, rhs, cot_mat, csc.shape[0], csc.shape[1]
            )
        )

    with ms.runtime.context(n_threads=3):
        csr_vec_parallel = to_numpy(mx.grad(csr_vec_grad)(csr.data))
        csr_mat_parallel = to_numpy(mx.grad(csr_mat_grad)(csr.data))
        coo_parallel = to_numpy(
            ext.coo_matmul_data_vjp(
                coo.row, coo.col, rhs, cot_mat, coo.shape[0], coo.shape[1]
            )
        )
        csc_parallel = to_numpy(
            ext.csc_matmul_data_vjp(
                csc.indices, csc.indptr, rhs, cot_mat, csc.shape[0], csc.shape[1]
            )
        )

    np.testing.assert_allclose(csr_vec_parallel, csr_vec_serial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(csr_mat_parallel, csr_mat_serial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(coo_parallel, coo_serial, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(csc_parallel, csc_serial, rtol=1e-6, atol=1e-6)


@pytest.mark.cpu_only
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_scatter_reductions_and_products_parallel_match_serial(mx, index_dtype):
    csr = _sample_csr(mx, index_dtype).sort_indices().sum_duplicates()
    indptr_np = to_numpy(csr.indptr)
    rows_np = np.repeat(
        np.arange(csr.shape[0], dtype=index_dtype), np.diff(indptr_np).astype(np.int64)
    )
    coo = ms.coo_array(
        (csr.data, (mx.array(rows_np), csr.indices)),
        shape=csr.shape,
        canonical=True,
    )
    csc = csr.tocsc(canonical=True)
    x_cols = mx.array(np.linspace(-1.25, 0.75, csr.shape[1], dtype=np.float32))
    x_rows = mx.array(np.linspace(0.5, 1.5, csr.shape[0], dtype=np.float32))
    rhs_cols = mx.array(
        np.arange(csr.shape[1] * 4, dtype=np.float32).reshape(csr.shape[1], 4) / 11.0
    )
    rhs_rows = mx.array(
        np.arange(csr.shape[0] * 4, dtype=np.float32).reshape(csr.shape[0], 4) / 7.0
    )

    with ms.runtime.context(n_threads=1):
        serial = {
            "csr_col_sums": to_numpy(csr.col_sums()),
            "csc_row_sums": to_numpy(csc.row_sums()),
            "csc_row_norms": to_numpy(csc.row_norms()),
            "coo_row_sums": to_numpy(coo.row_sums()),
            "coo_col_sums": to_numpy(coo.col_sums()),
            "coo_row_norms": to_numpy(coo.row_norms()),
            "coo_col_norms": to_numpy(coo.col_norms()),
            "coo_matvec": to_numpy(coo @ x_cols),
            "coo_matmul": to_numpy(coo @ rhs_cols),
            "csc_matvec": to_numpy(csc @ x_cols),
            "csc_matmul": to_numpy(csc @ rhs_cols),
            "csr_transpose_matvec": to_numpy(
                native.csr_matvec_transpose(
                    csr.data, csr.indices, csr.indptr, x_rows, csr.shape
                )
            ),
            "csr_transpose_matmul": to_numpy(
                native.csr_matmul_transpose(
                    csr.data, csr.indices, csr.indptr, rhs_rows, csr.shape
                )
            ),
            "csc_transpose_matvec": to_numpy(
                native.csc_matvec_transpose(
                    csc.data, csc.indices, csc.indptr, x_rows, csc.shape
                )
            ),
            "csc_transpose_matmul": to_numpy(
                native.csc_matmul_transpose(
                    csc.data, csc.indices, csc.indptr, rhs_rows, csc.shape
                )
            ),
        }

    with ms.runtime.context(n_threads=3):
        parallel = {
            "csr_col_sums": to_numpy(csr.col_sums()),
            "csc_row_sums": to_numpy(csc.row_sums()),
            "csc_row_norms": to_numpy(csc.row_norms()),
            "coo_row_sums": to_numpy(coo.row_sums()),
            "coo_col_sums": to_numpy(coo.col_sums()),
            "coo_row_norms": to_numpy(coo.row_norms()),
            "coo_col_norms": to_numpy(coo.col_norms()),
            "coo_matvec": to_numpy(coo @ x_cols),
            "coo_matmul": to_numpy(coo @ rhs_cols),
            "csc_matvec": to_numpy(csc @ x_cols),
            "csc_matmul": to_numpy(csc @ rhs_cols),
            "csr_transpose_matvec": to_numpy(
                native.csr_matvec_transpose(
                    csr.data, csr.indices, csr.indptr, x_rows, csr.shape
                )
            ),
            "csr_transpose_matmul": to_numpy(
                native.csr_matmul_transpose(
                    csr.data, csr.indices, csr.indptr, rhs_rows, csr.shape
                )
            ),
            "csc_transpose_matvec": to_numpy(
                native.csc_matvec_transpose(
                    csc.data, csc.indices, csc.indptr, x_rows, csc.shape
                )
            ),
            "csc_transpose_matmul": to_numpy(
                native.csc_matmul_transpose(
                    csc.data, csc.indices, csc.indptr, rhs_rows, csc.shape
                )
            ),
        }

    for name, expected in serial.items():
        np.testing.assert_allclose(
            parallel[name], expected, rtol=1e-5, atol=1e-5, err_msg=name
        )
