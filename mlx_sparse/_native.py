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

import mlx.core as mx

import mlx_sparse._fallback as _fallback
from mlx_sparse._ext_loader import extension
from mlx_sparse._typing import Shape2D


def _index_dtype_bits(index_dtype) -> int:
    if index_dtype == mx.int32:
        return 32
    if index_dtype == mx.int64:
        return 64
    raise TypeError(f"index_dtype must be mx.int32 or mx.int64, got {index_dtype}.")


def identity_like(x):
    ext = extension()
    if ext is None:
        import mlx.core as mx

        return mx.array(x)
    return ext.identity_like(x)


def coo_tocsr(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        return _fallback.coo_to_csr(data, row, col, shape)
    return ext.coo_tocsr(data, row, col, shape[0], shape[1])


def coo_tocsc(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        return _fallback.coo_to_csc(data, row, col, shape)
    return ext.coo_tocsc(data, row, col, shape[0], shape[1])


def coo_kron(lhs, rhs):
    ext = extension()
    if ext is None:
        raise RuntimeError("coo_kron requires the native mlx_sparse extension.")
    return ext.coo_kron(
        lhs.data,
        lhs.row,
        lhs.col,
        rhs.data,
        rhs.row,
        rhs.col,
        lhs.shape[0],
        lhs.shape[1],
        rhs.shape[0],
        rhs.shape[1],
    )


def coo_block(blocks, row_offsets, col_offsets, shape: Shape2D):
    ext = extension()
    if ext is None:
        raise RuntimeError("coo_block requires the native mlx_sparse extension.")
    return ext.coo_block(
        [block.data for block in blocks],
        [block.row for block in blocks],
        [block.col for block in blocks],
        [int(offset) for offset in row_offsets],
        [int(offset) for offset in col_offsets],
        int(shape[0]),
        int(shape[1]),
    )


def coo_triangular(array, *, k: int, upper: bool):
    ext = extension()
    if ext is None:
        raise RuntimeError("coo_triangular requires the native mlx_sparse extension.")
    return ext.coo_triangular(
        array.data,
        array.row,
        array.col,
        array.shape[0],
        array.shape[1],
        int(k),
        bool(upper),
    )


def csr_triangular(array, *, k: int, upper: bool):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_triangular requires the native mlx_sparse extension.")
    return ext.csr_triangular(
        array.data,
        array.indices,
        array.indptr,
        array.shape[0],
        array.shape[1],
        int(k),
        bool(upper),
    )


def csc_triangular(array, *, k: int, upper: bool):
    ext = extension()
    if ext is None:
        raise RuntimeError("csc_triangular requires the native mlx_sparse extension.")
    return ext.csc_triangular(
        array.data,
        array.indices,
        array.indptr,
        array.shape[0],
        array.shape[1],
        int(k),
        bool(upper),
    )


def csr_todense(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_todense(data, indices, indptr, shape)
    return ext.csr_todense(data, indices, indptr, shape[0], shape[1])


def csc_todense(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csc_todense(data, indices, indptr, shape)
    return ext.csc_todense(data, indices, indptr, shape[0], shape[1])


def coo_todense(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.coo_todense(data, row, col, shape)
    return ext.coo_todense(data, row, col, shape[0], shape[1])


def csr_matvec(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_matvec(data, indices, indptr, x, shape)
    return ext.csr_matvec(data, indices, indptr, x, shape[0], shape[1])


def csc_matvec(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csc_matvec(data, indices, indptr, x, shape)
    return ext.csc_matvec(data, indices, indptr, x, shape[0], shape[1])


def coo_matvec(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.coo_matvec(data, row, col, x, shape)
    return ext.coo_matvec(data, row, col, x, shape[0], shape[1])


def csr_batched_matvec(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        dense = _fallback.csr_todense(data, indices, indptr, shape)
        return rhs @ mx.transpose(dense)
    return ext.csr_batched_matvec(data, indices, indptr, rhs, shape[0], shape[1])


def coo_batched_matvec(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        dense = _fallback.coo_todense(data, row, col, shape)
        return rhs @ mx.transpose(dense)
    return ext.coo_batched_matvec(data, row, col, rhs, shape[0], shape[1])


def csc_batched_matvec(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        dense = _fallback.csc_todense(data, indices, indptr, shape)
        return rhs @ mx.transpose(dense)
    return ext.csc_batched_matvec(data, indices, indptr, rhs, shape[0], shape[1])


def csr_matvec_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_matvec_transpose(data, indices, indptr, x, shape)
    return ext.csr_matvec_transpose(data, indices, indptr, x, shape[0], shape[1])


def csc_matvec_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csc_matvec_transpose(data, indices, indptr, x, shape)
    return ext.csc_matvec_transpose(data, indices, indptr, x, shape[0], shape[1])


def csr_matmul(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_matmul(data, indices, indptr, rhs, shape)
    return ext.csr_matmul(data, indices, indptr, rhs, shape[0], shape[1])


def coo_matmul(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.coo_matmul(data, row, col, rhs, shape)
    return ext.coo_matmul(data, row, col, rhs, shape[0], shape[1])


def csc_matmul(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csc_matmul(data, indices, indptr, rhs, shape)
    return ext.csc_matmul(data, indices, indptr, rhs, shape[0], shape[1])


def csr_batched_matmul(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        dense = _fallback.csr_todense(data, indices, indptr, shape)
        return mx.matmul(dense[None, :, :], rhs)
    return ext.csr_batched_matmul(data, indices, indptr, rhs, shape[0], shape[1])


def coo_batched_matmul(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        dense = _fallback.coo_todense(data, row, col, shape)
        return mx.matmul(dense[None, :, :], rhs)
    return ext.coo_batched_matmul(data, row, col, rhs, shape[0], shape[1])


def csc_batched_matmul(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        dense = _fallback.csc_todense(data, indices, indptr, shape)
        return mx.matmul(dense[None, :, :], rhs)
    return ext.csc_batched_matmul(data, indices, indptr, rhs, shape[0], shape[1])


def csr_matmul_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_matmul_transpose(data, indices, indptr, rhs, shape)
    return ext.csr_matmul_transpose(data, indices, indptr, rhs, shape[0], shape[1])


def csc_matmul_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csc_matmul_transpose(data, indices, indptr, rhs, shape)
    return ext.csc_matmul_transpose(data, indices, indptr, rhs, shape[0], shape[1])


def csr_matmat(lhs, rhs):
    ext = extension()
    if ext is None:
        return _fallback.csr_matmat(lhs, rhs)
    return ext.csr_matmat(
        lhs.data,
        lhs.indices,
        lhs.indptr,
        rhs.data,
        rhs.indices,
        rhs.indptr,
        lhs.shape[0],
        lhs.shape[1],
        rhs.shape[0],
        rhs.shape[1],
    )


def coo_matmat(lhs, rhs):
    ext = extension()
    if ext is None:
        return _fallback.coo_matmat(lhs, rhs)
    return ext.coo_matmat(
        lhs.data,
        lhs.row,
        lhs.col,
        rhs.data,
        rhs.row,
        rhs.col,
        lhs.shape[0],
        lhs.shape[1],
        rhs.shape[0],
        rhs.shape[1],
    )


def csc_matmat(lhs, rhs):
    ext = extension()
    if ext is None:
        return _fallback.csc_matmat(lhs, rhs)
    return ext.csc_matmat(
        lhs.data,
        lhs.indices,
        lhs.indptr,
        rhs.data,
        rhs.indices,
        rhs.indptr,
        lhs.shape[0],
        lhs.shape[1],
        rhs.shape[0],
        rhs.shape[1],
    )


def csr_add(lhs, rhs, *, subtract: bool = False):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_add requires the native mlx_sparse extension.")
    return ext.csr_add(
        lhs.data,
        lhs.indices,
        lhs.indptr,
        rhs.data,
        rhs.indices,
        rhs.indptr,
        lhs.shape[0],
        lhs.shape[1],
        bool(subtract),
    )


def csr_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        return _fallback.csr_transpose(data, indices, indptr, shape)
    return ext.csr_transpose(data, indices, indptr, shape[0], shape[1])


def csr_tocsc(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        return _fallback.csr_to_csc(data, indices, indptr, shape)
    return ext.csr_tocsc(data, indices, indptr, shape[0], shape[1])


def csc_tocsr(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        return _fallback.csc_to_csr(data, indices, indptr, shape)
    return ext.csc_tocsr(data, indices, indptr, shape[0], shape[1])


def csr_tocoo(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_tocoo requires the native mlx_sparse extension.")
    return ext.csr_tocoo(data, indices, indptr, shape[0], shape[1])


def csc_tocoo(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csc_tocoo requires the native mlx_sparse extension.")
    return ext.csc_tocoo(data, indices, indptr, shape[0], shape[1])


def csr_row_sums(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_row_sums(data, indices, indptr, shape)
    return ext.csr_row_sums(data, indices, indptr, shape[0], shape[1])


def csr_col_sums(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_col_sums(data, indices, indptr, shape)
    return ext.csr_col_sums(data, indices, indptr, shape[0], shape[1])


def csr_row_norms(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_row_norms(data, indices, indptr, shape)
    return ext.csr_row_norms(data, indices, indptr, shape[0], shape[1])


def csr_diagonal(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_diagonal(data, indices, indptr, shape)
    return ext.csr_diagonal(data, indices, indptr, shape[0], shape[1])


def csr_trace(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_trace(data, indices, indptr, shape)
    return ext.csr_trace(data, indices, indptr, shape[0], shape[1])


def coo_row_sums(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.coo_row_sums(data, row, col, shape)
    return ext.coo_row_sums(data, row, col, shape[0], shape[1])


def coo_col_sums(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.coo_col_sums(data, row, col, shape)
    return ext.coo_col_sums(data, row, col, shape[0], shape[1])


def coo_row_norms(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
    *,
    assume_canonical: bool = False,
) -> mx.array:
    if not assume_canonical:
        csr_data, csr_indices, csr_indptr = coo_tocsr(data, row, col, shape)
        csr_data, csr_indices, csr_indptr = csr_sum_duplicates(
            csr_data, csr_indices, csr_indptr
        )
        return csr_row_norms(csr_data, csr_indices, csr_indptr, shape)
    ext = extension()
    if ext is None:
        return _fallback.coo_row_norms(data, row, col, shape)
    return ext.coo_row_norms(data, row, col, shape[0], shape[1])


def coo_col_norms(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
    *,
    assume_canonical: bool = False,
) -> mx.array:
    if not assume_canonical:
        csc_data, csc_indices, csc_indptr = coo_tocsc(data, row, col, shape)
        csc_data, csc_indices, csc_indptr = csc_sum_duplicates(
            csc_data, csc_indices, csc_indptr
        )
        return csc_col_norms(
            csc_data, csc_indices, csc_indptr, shape, assume_canonical=True
        )
    ext = extension()
    if ext is None:
        return _fallback.coo_col_norms(data, row, col, shape)
    return ext.coo_col_norms(data, row, col, shape[0], shape[1])


def coo_diagonal(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.coo_diagonal(data, row, col, shape)
    return ext.coo_diagonal(data, row, col, shape[0], shape[1])


def coo_trace(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.coo_trace(data, row, col, shape)
    return ext.coo_trace(data, row, col, shape[0], shape[1])


def csc_row_sums(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csc_row_sums(data, indices, indptr, shape)
    return ext.csc_row_sums(data, indices, indptr, shape[0], shape[1])


def csc_col_sums(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csc_col_sums(data, indices, indptr, shape)
    return ext.csc_col_sums(data, indices, indptr, shape[0], shape[1])


def csc_row_norms(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    *,
    assume_canonical: bool = False,
) -> mx.array:
    if not assume_canonical:
        sorted_data, sorted_indices, sorted_indptr = csc_sort_indices(
            data, indices, indptr
        )
        csc_data, csc_indices, csc_indptr = csc_sum_duplicates(
            sorted_data, sorted_indices, sorted_indptr
        )
        return csc_row_norms(
            csc_data, csc_indices, csc_indptr, shape, assume_canonical=True
        )
    ext = extension()
    if ext is None:
        return _fallback.csc_row_norms(data, indices, indptr, shape)
    return ext.csc_row_norms(data, indices, indptr, shape[0], shape[1])


def csc_col_norms(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    *,
    assume_canonical: bool = False,
) -> mx.array:
    if not assume_canonical:
        sorted_data, sorted_indices, sorted_indptr = csc_sort_indices(
            data, indices, indptr
        )
        csc_data, csc_indices, csc_indptr = csc_sum_duplicates(
            sorted_data, sorted_indices, sorted_indptr
        )
        return csc_col_norms(
            csc_data, csc_indices, csc_indptr, shape, assume_canonical=True
        )
    ext = extension()
    if ext is None:
        return _fallback.csc_col_norms(data, indices, indptr, shape)
    return ext.csc_col_norms(data, indices, indptr, shape[0], shape[1])


def csc_diagonal(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csc_diagonal(data, indices, indptr, shape)
    return ext.csc_diagonal(data, indices, indptr, shape[0], shape[1])


def csc_trace(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csc_trace(data, indices, indptr, shape)
    return ext.csc_trace(data, indices, indptr, shape[0], shape[1])


def csr_sort_indices(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    ext = extension()
    if ext is None:
        return _fallback.sort_csr_indices(data, indices, indptr)
    return ext.csr_sort_indices(data, indices, indptr)


def csc_sort_indices(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    ext = extension()
    if ext is None:
        return _fallback.sort_csc_indices(data, indices, indptr)
    return ext.csc_sort_indices(data, indices, indptr)


def csr_sum_duplicates(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    ext = extension()
    if ext is None:
        return _fallback.sum_csr_duplicates(data, indices, indptr)
    return ext.csr_sum_duplicates(data, indices, indptr)


def csc_sum_duplicates(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    ext = extension()
    if ext is None:
        return _fallback.sum_csc_duplicates(data, indices, indptr)
    return ext.csc_sum_duplicates(data, indices, indptr)


def csr_fromdense(
    dense: mx.array,
    *,
    index_dtype,
    threshold: float,
):
    ext = extension()
    if ext is None:
        return _fallback.fromdense(dense, index_dtype=index_dtype, threshold=threshold)
    return ext.csr_fromdense(dense, _index_dtype_bits(index_dtype), float(threshold))


def random_coo_indices(
    key: mx.array,
    shape: Shape2D,
    nnz: int,
    *,
    index_dtype,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("random_coo_indices requires the native extension.")
    return ext.random_coo_indices(
        key,
        int(shape[0]),
        int(shape[1]),
        int(nnz),
        _index_dtype_bits(index_dtype),
    )


def random_csr_indices(
    key: mx.array,
    shape: Shape2D,
    nnz: int,
    *,
    index_dtype,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("random_csr_indices requires the native extension.")
    return ext.random_csr_indices(
        key,
        int(shape[0]),
        int(shape[1]),
        int(nnz),
        _index_dtype_bits(index_dtype),
    )


def random_csc_indices(
    key: mx.array,
    shape: Shape2D,
    nnz: int,
    *,
    index_dtype,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("random_csc_indices requires the native extension.")
    return ext.random_csc_indices(
        key,
        int(shape[0]),
        int(shape[1]),
        int(nnz),
        _index_dtype_bits(index_dtype),
    )


def csr_cg(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_cg requires the native mlx_sparse extension.")
    return ext.csr_cg(
        data,
        indices,
        indptr,
        b,
        x0,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(maxiter),
    )


def csr_pcg_jacobi(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    inv_diag: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_pcg_jacobi requires the native mlx_sparse extension.")
    return ext.csr_pcg_jacobi(
        data,
        indices,
        indptr,
        b,
        x0,
        inv_diag,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(maxiter),
    )


def csr_pcg_ic0(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    l_data: mx.array,
    l_indices: mx.array,
    l_indptr: mx.array,
    lt_data: mx.array,
    lt_indices: mx.array,
    lt_indptr: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_pcg_ic0 requires the native mlx_sparse extension.")
    return ext.csr_pcg_ic0(
        data,
        indices,
        indptr,
        b,
        x0,
        l_data,
        l_indices,
        l_indptr,
        lt_data,
        lt_indices,
        lt_indptr,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(maxiter),
    )


def csr_pcg_chebyshev(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    m_data: mx.array,
    m_indices: mx.array,
    m_indptr: mx.array,
    shape: Shape2D,
    *,
    degree: int,
    lambda_min: float,
    lambda_max: float,
    rtol: float,
    atol: float,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_pcg_chebyshev requires the native mlx_sparse extension."
        )
    return ext.csr_pcg_chebyshev(
        data,
        indices,
        indptr,
        b,
        x0,
        m_data,
        m_indices,
        m_indptr,
        shape[0],
        shape[1],
        int(degree),
        float(lambda_min),
        float(lambda_max),
        float(rtol),
        float(atol),
        int(maxiter),
    )


def diagonal_preconditioner_apply(inv_diag: mx.array, rhs: mx.array):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "diagonal_preconditioner_apply requires the native mlx_sparse extension."
        )
    return ext.diagonal_preconditioner_apply(inv_diag, rhs)


def csr_chebyshev_spectral_bounds(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    *,
    estimate: bool,
    estimate_steps: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_chebyshev_spectral_bounds requires the native mlx_sparse " "extension."
        )
    return ext.csr_chebyshev_spectral_bounds(
        data,
        indices,
        indptr,
        shape[0],
        shape[1],
        bool(estimate),
        int(estimate_steps),
    )


def csr_chebyshev_preconditioner_apply(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
    *,
    degree: int,
    lambda_min: float,
    lambda_max: float,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_chebyshev_preconditioner_apply requires the native "
            "mlx_sparse extension."
        )
    return ext.csr_chebyshev_preconditioner_apply(
        data,
        indices,
        indptr,
        rhs,
        shape[0],
        shape[1],
        int(degree),
        float(lambda_min),
        float(lambda_max),
    )


def csr_exact_lu_preconditioner_apply(
    perm: mx.array,
    l_data: mx.array,
    l_indices: mx.array,
    l_indptr: mx.array,
    u_data: mx.array,
    u_indices: mx.array,
    u_indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_exact_lu_preconditioner_apply requires the native mlx_sparse "
            "extension."
        )
    return ext.csr_exact_lu_preconditioner_apply(
        perm,
        l_data,
        l_indices,
        l_indptr,
        u_data,
        u_indices,
        u_indptr,
        rhs,
        shape[0],
        shape[1],
    )


def csr_exact_cholesky_preconditioner_apply(
    l_data: mx.array,
    l_indices: mx.array,
    l_indptr: mx.array,
    lt_data: mx.array,
    lt_indices: mx.array,
    lt_indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_exact_cholesky_preconditioner_apply requires the native "
            "mlx_sparse extension."
        )
    return ext.csr_exact_cholesky_preconditioner_apply(
        l_data,
        l_indices,
        l_indptr,
        lt_data,
        lt_indices,
        lt_indptr,
        rhs,
        shape[0],
        shape[1],
    )


def csr_ilu0(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    *,
    shift: float,
    check: bool,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_ilu0 requires the native mlx_sparse extension.")
    return ext.csr_ilu0(
        data,
        indices,
        indptr,
        shape[0],
        shape[1],
        float(shift),
        bool(check),
    )


def csr_ic0(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    *,
    shift: float,
    check: bool,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_ic0 requires the native mlx_sparse extension.")
    return ext.csr_ic0(
        data,
        indices,
        indptr,
        shape[0],
        shape[1],
        float(shift),
        bool(check),
    )


def csr_ilu0_preconditioner_apply(
    l_data: mx.array,
    l_indices: mx.array,
    l_indptr: mx.array,
    u_data: mx.array,
    u_indices: mx.array,
    u_indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_ilu0_preconditioner_apply requires the native mlx_sparse " "extension."
        )
    return ext.csr_ilu0_preconditioner_apply(
        l_data,
        l_indices,
        l_indptr,
        u_data,
        u_indices,
        u_indptr,
        rhs,
        shape[0],
        shape[1],
    )


def csr_ic0_preconditioner_apply(
    l_data: mx.array,
    l_indices: mx.array,
    l_indptr: mx.array,
    lt_data: mx.array,
    lt_indices: mx.array,
    lt_indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_ic0_preconditioner_apply requires the native mlx_sparse extension."
        )
    return ext.csr_ic0_preconditioner_apply(
        l_data,
        l_indices,
        l_indptr,
        lt_data,
        lt_indices,
        lt_indptr,
        rhs,
        shape[0],
        shape[1],
    )


def csr_lanczos(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    v0: mx.array,
    shape: Shape2D,
    *,
    k: int,
    reorthogonalize: bool,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_lanczos requires the native mlx_sparse extension.")
    return ext.csr_lanczos(
        data,
        indices,
        indptr,
        v0,
        shape[0],
        shape[1],
        int(k),
        bool(reorthogonalize),
    )


def csr_gmres(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_gmres requires the native mlx_sparse extension.")
    return ext.csr_gmres(
        data,
        indices,
        indptr,
        b,
        x0,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(restart),
        int(maxiter),
    )


def csr_gmres_jacobi(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    inv_diag: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_gmres_jacobi requires the native mlx_sparse extension.")
    return ext.csr_gmres_jacobi(
        data,
        indices,
        indptr,
        b,
        x0,
        inv_diag,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(restart),
        int(maxiter),
    )


def csr_gmres_exact_lu(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    perm: mx.array,
    l_data: mx.array,
    l_indices: mx.array,
    l_indptr: mx.array,
    u_data: mx.array,
    u_indices: mx.array,
    u_indptr: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_gmres_exact_lu requires the native mlx_sparse extension."
        )
    return ext.csr_gmres_exact_lu(
        data,
        indices,
        indptr,
        b,
        x0,
        perm,
        l_data,
        l_indices,
        l_indptr,
        u_data,
        u_indices,
        u_indptr,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(restart),
        int(maxiter),
    )


def csr_gmres_ilu0(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    l_data: mx.array,
    l_indices: mx.array,
    l_indptr: mx.array,
    u_data: mx.array,
    u_indices: mx.array,
    u_indptr: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_gmres_ilu0 requires the native mlx_sparse extension.")
    return ext.csr_gmres_ilu0(
        data,
        indices,
        indptr,
        b,
        x0,
        l_data,
        l_indices,
        l_indptr,
        u_data,
        u_indices,
        u_indptr,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(restart),
        int(maxiter),
    )


def csr_gmres_exact_cholesky(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    l_data: mx.array,
    l_indices: mx.array,
    l_indptr: mx.array,
    lt_data: mx.array,
    lt_indices: mx.array,
    lt_indptr: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_gmres_exact_cholesky requires the native mlx_sparse extension."
        )
    return ext.csr_gmres_exact_cholesky(
        data,
        indices,
        indptr,
        b,
        x0,
        l_data,
        l_indices,
        l_indptr,
        lt_data,
        lt_indices,
        lt_indptr,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(restart),
        int(maxiter),
    )


def csr_gmres_exact_accelerate(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    solver,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
):
    ext = extension()
    if ext is None or not hasattr(ext, "csr_gmres_exact_accelerate"):
        raise RuntimeError(
            "csr_gmres_exact_accelerate requires an Accelerate-enabled native "
            "mlx_sparse extension."
        )
    return ext.csr_gmres_exact_accelerate(
        data,
        indices,
        indptr,
        b,
        x0,
        solver,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(restart),
        int(maxiter),
    )


def csr_minres(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
    shift: float,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_minres requires the native mlx_sparse extension.")
    return ext.csr_minres(
        data,
        indices,
        indptr,
        b,
        x0,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(maxiter),
        float(shift),
    )


def csr_minres_jacobi(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    inv_diag: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
    shift: float,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_minres_jacobi requires the native mlx_sparse extension."
        )
    return ext.csr_minres_jacobi(
        data,
        indices,
        indptr,
        b,
        x0,
        inv_diag,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(maxiter),
        float(shift),
    )


def csr_arnoldi(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    v0: mx.array,
    shape: Shape2D,
    *,
    k: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_arnoldi requires the native mlx_sparse extension.")
    return ext.csr_arnoldi(
        data,
        indices,
        indptr,
        v0,
        shape[0],
        shape[1],
        int(k),
    )


def csr_eigsh(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    v0: mx.array,
    shape: Shape2D,
    *,
    k: int,
    ncv: int,
    which: str,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_eigsh requires the native mlx_sparse extension.")
    return ext.csr_eigsh(
        data,
        indices,
        indptr,
        v0,
        shape[0],
        shape[1],
        int(k),
        int(ncv),
        str(which),
    )


def csr_eigs(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    v0: mx.array,
    shape: Shape2D,
    *,
    k: int,
    ncv: int,
    which: str,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_eigs requires the native mlx_sparse extension.")
    return ext.csr_eigs(
        data,
        indices,
        indptr,
        v0,
        shape[0],
        shape[1],
        int(k),
        int(ncv),
        str(which),
    )


def csr_svds(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    v0: mx.array,
    shape: Shape2D,
    *,
    k: int,
    ncv: int,
    which: str,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_svds requires the native mlx_sparse extension.")
    return ext.csr_svds(
        data,
        indices,
        indptr,
        v0,
        shape[0],
        shape[1],
        int(k),
        int(ncv),
        str(which),
    )


def csr_normal_lanczos(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    v0: mx.array,
    shape: Shape2D,
    *,
    k: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_normal_lanczos requires the native mlx_sparse extension."
        )
    return ext.csr_normal_lanczos(
        data,
        indices,
        indptr,
        v0,
        shape[0],
        shape[1],
        int(k),
    )


def csr_cholesky(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_cholesky requires the native mlx_sparse extension.")
    return ext.csr_cholesky(data, indices, indptr, shape[0], shape[1])


def csr_lu(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_lu requires the native mlx_sparse extension.")
    return ext.csr_lu(data, indices, indptr, shape[0], shape[1])


def accelerate_solvers_available() -> bool:
    ext = extension()
    if ext is None:
        return False
    checker = getattr(ext, "_accelerate_solvers_available", None)
    return bool(checker()) if checker is not None else False


def is_accelerate_float_solve(solver) -> bool:
    """Return whether ``solver`` is the native Accelerate float solve type."""

    ext = extension()
    if ext is None:
        return False
    solver_type = getattr(ext, "_AccelerateFloatSolve", None)
    return solver_type is not None and isinstance(solver, solver_type)


def accelerate_lu_solvers_available() -> bool:
    ext = extension()
    if ext is None:
        return False
    checker = getattr(ext, "_accelerate_lu_solvers_available", None)
    return bool(checker()) if checker is not None else False


def accelerate_factorize_csr_float32(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    method: str,
):
    ext = extension()
    if ext is None or not hasattr(ext, "accelerate_factorize_csr_float32"):
        raise RuntimeError("Accelerate sparse direct solves are not available.")
    return ext.accelerate_factorize_csr_float32(
        data, indices, indptr, shape[0], shape[1], method
    )


def accelerate_factorize_csc_float32(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    method: str,
):
    ext = extension()
    if ext is None or not hasattr(ext, "accelerate_factorize_csc_float32"):
        raise RuntimeError("Accelerate sparse direct solves are not available.")
    return ext.accelerate_factorize_csc_float32(
        data, indices, indptr, shape[0], shape[1], method
    )


def accelerate_factorize_coo_float32(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
    method: str,
):
    ext = extension()
    if ext is None or not hasattr(ext, "accelerate_factorize_coo_float32"):
        raise RuntimeError("Accelerate sparse direct solves are not available.")
    return ext.accelerate_factorize_coo_float32(
        data, row, col, shape[0], shape[1], method
    )


def csr_triangular_solve(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    shape: Shape2D,
    *,
    lower: bool,
    unit_diagonal: bool,
    diagonal_positions: mx.array | None = None,
    level_schedule: tuple[mx.array, mx.array] | None = None,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_triangular_solve requires the native mlx_sparse extension."
        )
    if diagonal_positions is not None or level_schedule is not None:
        if diagonal_positions is None:
            diagonal_positions = csr_triangular_diagonal_positions(
                indices, indptr, shape
            )
        if level_schedule is None:
            level_offsets = mx.array([], dtype=mx.int32)
            level_rows = mx.array([], dtype=mx.int32)
        else:
            level_offsets, level_rows = level_schedule
        return ext.csr_triangular_solve_analyzed(
            data,
            indices,
            indptr,
            b,
            diagonal_positions,
            level_offsets,
            level_rows,
            shape[0],
            shape[1],
            bool(lower),
            bool(unit_diagonal),
        )
    return ext.csr_triangular_solve(
        data,
        indices,
        indptr,
        b,
        shape[0],
        shape[1],
        bool(lower),
        bool(unit_diagonal),
    )


def csr_triangular_diagonal_positions(
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_triangular_diagonal_positions requires the native mlx_sparse extension."
        )
    return ext.csr_triangular_diagonal_positions(
        indices,
        indptr,
        shape[0],
        shape[1],
    )


def csr_triangular_level_schedule(
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    *,
    lower: bool,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_triangular_level_schedule requires the native mlx_sparse extension."
        )
    return ext.csr_triangular_level_schedule(
        indices,
        indptr,
        shape[0],
        shape[1],
        bool(lower),
    )


def csr_vdot(
    lhs_data: mx.array,
    lhs_indices: mx.array,
    lhs_indptr: mx.array,
    rhs_data: mx.array,
    rhs_indices: mx.array,
    rhs_indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_vdot requires the native mlx_sparse extension.")
    return ext.csr_vdot(
        lhs_data,
        lhs_indices,
        lhs_indptr,
        rhs_data,
        rhs_indices,
        rhs_indptr,
        shape[0],
        shape[1],
    )


def csr_dot(
    lhs_data: mx.array,
    lhs_indices: mx.array,
    lhs_indptr: mx.array,
    rhs_data: mx.array,
    rhs_indices: mx.array,
    rhs_indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_dot requires the native mlx_sparse extension.")
    return ext.csr_dot(
        lhs_data,
        lhs_indices,
        lhs_indptr,
        rhs_data,
        rhs_indices,
        rhs_indptr,
        shape[0],
        shape[1],
    )


def csr_permute_vector(x: mx.array, perm: mx.array):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_permute_vector requires the native mlx_sparse extension."
        )
    return ext.csr_permute_vector(x, perm)
