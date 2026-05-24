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
import numpy as np

from mlx_sparse._host import to_mx, to_numpy
from mlx_sparse._typing import Shape2D


def coo_to_csr(data: mx.array, row: mx.array, col: mx.array, shape: Shape2D):
    data_np = to_numpy(data)
    row_np = to_numpy(row)
    col_np = to_numpy(col)

    order = np.lexsort((np.arange(row_np.size), col_np, row_np))
    sorted_row = row_np[order]
    sorted_col = col_np[order]
    sorted_data = data_np[order]

    indptr = np.zeros(shape[0] + 1, dtype=row_np.dtype)
    if sorted_row.size:
        counts = np.bincount(sorted_row.astype(np.int64), minlength=shape[0])
        indptr[1:] = np.cumsum(counts, dtype=indptr.dtype)

    return (
        to_mx(sorted_data, dtype=data.dtype),
        to_mx(sorted_col.astype(col_np.dtype, copy=False), dtype=col.dtype),
        to_mx(indptr, dtype=row.dtype),
    )


def coo_to_csc(data: mx.array, row: mx.array, col: mx.array, shape: Shape2D):
    data_np = to_numpy(data)
    row_np = to_numpy(row)
    col_np = to_numpy(col)

    order = np.lexsort((np.arange(col_np.size), row_np, col_np))
    sorted_col = col_np[order]
    sorted_row = row_np[order]
    sorted_data = data_np[order]

    indptr = np.zeros(shape[1] + 1, dtype=col_np.dtype)
    if sorted_col.size:
        counts = np.bincount(sorted_col.astype(np.int64), minlength=shape[1])
        indptr[1:] = np.cumsum(counts, dtype=indptr.dtype)

    return (
        to_mx(sorted_data, dtype=data.dtype),
        to_mx(sorted_row.astype(row_np.dtype, copy=False), dtype=row.dtype),
        to_mx(indptr, dtype=col.dtype),
    )


def csr_todense(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    dense = np.zeros(shape, dtype=data_np.dtype)
    for row in range(shape[0]):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        np.add.at(dense[row], indices_np[start:end], data_np[start:end])
    return to_mx(dense, dtype=data.dtype)


def csc_todense(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    dense = np.zeros(shape, dtype=data_np.dtype)
    for col in range(shape[1]):
        start = int(indptr_np[col])
        end = int(indptr_np[col + 1])
        np.add.at(dense[:, col], indices_np[start:end], data_np[start:end])
    return to_mx(dense, dtype=data.dtype)


def _reduction_accum_dtype(data: mx.array, data_np: np.ndarray):
    if data.dtype in {mx.float16, mx.bfloat16}:
        return np.float32
    return data_np.dtype


def csr_row_sums(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    del indices
    data_np = to_numpy(data)
    indptr_np = to_numpy(indptr)
    accum_dtype = _reduction_accum_dtype(data, data_np)
    out = np.zeros(shape[0], dtype=accum_dtype)
    for row in range(shape[0]):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        out[row] = data_np[start:end].sum(dtype=accum_dtype)
    return to_mx(out, dtype=data.dtype)


def csr_col_sums(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    accum_dtype = _reduction_accum_dtype(data, data_np)
    out = np.zeros(shape[1], dtype=accum_dtype)
    for row in range(shape[0]):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        np.add.at(out, indices_np[start:end], data_np[start:end])
    return to_mx(out, dtype=data.dtype)


def csr_row_norms(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    del indices
    data_np = to_numpy(data)
    indptr_np = to_numpy(indptr)
    out = np.zeros(shape[0], dtype=np.float32)
    for row in range(shape[0]):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        values = data_np[start:end]
        out[row] = np.sqrt(np.sum(np.abs(values) ** 2, dtype=np.float64))
    return to_mx(out, dtype=mx.float32)


def csr_diagonal(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    diag_size = min(shape)
    accum_dtype = _reduction_accum_dtype(data, data_np)
    out = np.zeros(diag_size, dtype=accum_dtype)
    for row in range(diag_size):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        mask = indices_np[start:end] == row
        if mask.any():
            out[row] = data_np[start:end][mask].sum(dtype=accum_dtype)
    return to_mx(out, dtype=data.dtype)


def csr_trace(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    diag_size = min(shape)
    accum_dtype = _reduction_accum_dtype(data, data_np)
    total = np.zeros((), dtype=accum_dtype)
    for row in range(diag_size):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        mask = indices_np[start:end] == row
        if mask.any():
            total[...] = total + data_np[start:end][mask].sum(dtype=accum_dtype)
    return to_mx(total, dtype=data.dtype)


def csr_matvec(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    x_np = to_numpy(x)
    out = np.zeros(shape[0], dtype=np.result_type(data_np.dtype, x_np.dtype))
    for row in range(shape[0]):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        out[row] = np.dot(data_np[start:end], x_np[indices_np[start:end]])
    return to_mx(out, dtype=data.dtype)


def csc_matvec(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    x_np = to_numpy(x)
    out = np.zeros(shape[0], dtype=np.result_type(data_np.dtype, x_np.dtype))
    for col in range(shape[1]):
        start = int(indptr_np[col])
        end = int(indptr_np[col + 1])
        np.add.at(out, indices_np[start:end], data_np[start:end] * x_np[col])
    return to_mx(out, dtype=data.dtype)


def csr_matvec_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    x_np = to_numpy(x)
    out = np.zeros(shape[1], dtype=np.result_type(data_np.dtype, x_np.dtype))
    for row in range(shape[0]):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        np.add.at(out, indices_np[start:end], data_np[start:end] * x_np[row])
    return to_mx(out, dtype=data.dtype)


def csc_matvec_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    x_np = to_numpy(x)
    out = np.zeros(shape[1], dtype=np.result_type(data_np.dtype, x_np.dtype))
    for col in range(shape[1]):
        start = int(indptr_np[col])
        end = int(indptr_np[col + 1])
        out[col] = np.dot(data_np[start:end], x_np[indices_np[start:end]])
    return to_mx(out, dtype=data.dtype)


def csr_matmul(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    rhs_np = to_numpy(rhs)
    out = np.zeros(
        (shape[0], rhs_np.shape[1]),
        dtype=np.result_type(data_np.dtype, rhs_np.dtype),
    )
    for row in range(shape[0]):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        for p in range(start, end):
            out[row] += data_np[p] * rhs_np[indices_np[p]]
    return to_mx(out, dtype=data.dtype)


def csr_matmat(lhs, rhs):
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(
            "CSR sparse-sparse matmul dimension mismatch: "
            f"{lhs.shape} @ {rhs.shape}."
        )
    if lhs.data.dtype != rhs.data.dtype:
        raise TypeError(
            "CSR sparse-sparse matmul requires matching value dtypes, "
            f"got {lhs.data.dtype} and {rhs.data.dtype}."
        )

    lhs_data = to_numpy(lhs.data)
    lhs_indices = to_numpy(lhs.indices)
    lhs_indptr = to_numpy(lhs.indptr)
    rhs_data = to_numpy(rhs.data)
    rhs_indices = to_numpy(rhs.indices)
    rhs_indptr = to_numpy(rhs.indptr)

    index_dtype = np.promote_types(lhs_indices.dtype, rhs_indices.dtype)
    out_data = []
    out_indices = []
    out_indptr = np.zeros(lhs.shape[0] + 1, dtype=index_dtype)

    for row in range(lhs.shape[0]):
        accum = {}
        for lhs_pos in range(int(lhs_indptr[row]), int(lhs_indptr[row + 1])):
            rhs_row = int(lhs_indices[lhs_pos])
            lhs_value = lhs_data[lhs_pos]
            for rhs_pos in range(
                int(rhs_indptr[rhs_row]), int(rhs_indptr[rhs_row + 1])
            ):
                col = int(rhs_indices[rhs_pos])
                accum[col] = accum.get(col, 0) + lhs_value * rhs_data[rhs_pos]

        for col in sorted(accum):
            value = accum[col]
            if value != 0:
                out_indices.append(col)
                out_data.append(value)
        out_indptr[row + 1] = len(out_data)

    if out_data:
        data_arr = np.asarray(out_data, dtype=lhs_data.dtype)
        indices_arr = np.asarray(out_indices, dtype=index_dtype)
    else:
        data_arr = np.empty((0,), dtype=lhs_data.dtype)
        indices_arr = np.empty((0,), dtype=index_dtype)

    return (
        to_mx(data_arr, dtype=lhs.data.dtype),
        to_mx(
            indices_arr,
            dtype=lhs.indices.dtype if lhs.index_dtype == rhs.index_dtype else mx.int64,
        ),
        to_mx(
            out_indptr,
            dtype=lhs.indptr.dtype if lhs.index_dtype == rhs.index_dtype else mx.int64,
        ),
    )


def csr_matmul_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    rhs_np = to_numpy(rhs)
    out = np.zeros(
        (shape[1], rhs_np.shape[1]),
        dtype=np.result_type(data_np.dtype, rhs_np.dtype),
    )
    for row in range(shape[0]):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        for p in range(start, end):
            out[indices_np[p]] += data_np[p] * rhs_np[row]
    return to_mx(out, dtype=data.dtype)


def csr_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)

    row = np.empty(data_np.shape[0], dtype=indices_np.dtype)
    for r in range(shape[0]):
        row[indptr_np[r] : indptr_np[r + 1]] = r

    return coo_to_csr(
        to_mx(data_np, dtype=data.dtype),
        to_mx(indices_np.astype(indices_np.dtype, copy=False), dtype=indices.dtype),
        to_mx(row, dtype=indices.dtype),
        (shape[1], shape[0]),
    )


def csr_to_csc(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)

    row = np.empty(data_np.shape[0], dtype=indices_np.dtype)
    for r in range(shape[0]):
        row[indptr_np[r] : indptr_np[r + 1]] = r

    return coo_to_csc(
        to_mx(data_np, dtype=data.dtype),
        to_mx(row, dtype=indices.dtype),
        to_mx(indices_np.astype(indices_np.dtype, copy=False), dtype=indices.dtype),
        shape,
    )


def csc_to_csr(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)

    col = np.empty(data_np.shape[0], dtype=indices_np.dtype)
    for c in range(shape[1]):
        col[indptr_np[c] : indptr_np[c + 1]] = c

    return coo_to_csr(
        to_mx(data_np, dtype=data.dtype),
        to_mx(indices_np.astype(indices_np.dtype, copy=False), dtype=indices.dtype),
        to_mx(col, dtype=indices.dtype),
        shape,
    )


def csr_conjugate(data: mx.array) -> mx.array:
    return mx.conjugate(data)


def sort_csr_indices(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)

    out_data = data_np.copy()
    out_indices = indices_np.copy()
    for row in range(indptr_np.size - 1):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        order = np.argsort(out_indices[start:end], kind="stable")
        out_data[start:end] = out_data[start:end][order]
        out_indices[start:end] = out_indices[start:end][order]
    return (
        to_mx(out_data, dtype=data.dtype),
        to_mx(out_indices, dtype=indices.dtype),
        indptr,
    )


def sort_csc_indices(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)

    out_data = data_np.copy()
    out_indices = indices_np.copy()
    for col in range(indptr_np.size - 1):
        start = int(indptr_np[col])
        end = int(indptr_np[col + 1])
        order = np.argsort(out_indices[start:end], kind="stable")
        out_data[start:end] = out_data[start:end][order]
        out_indices[start:end] = out_indices[start:end][order]
    return (
        to_mx(out_data, dtype=data.dtype),
        to_mx(out_indices, dtype=indices.dtype),
        indptr,
    )


def sum_csr_duplicates(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)

    out_data = []
    out_indices = []
    out_indptr = np.zeros_like(indptr_np)

    for row in range(indptr_np.size - 1):
        start = int(indptr_np[row])
        end = int(indptr_np[row + 1])
        row_indices = indices_np[start:end]
        row_data = data_np[start:end]
        if row_indices.size:
            unique, first = np.unique(row_indices, return_index=True)
            order = np.argsort(first)
            for col in unique[order]:
                mask = row_indices == col
                out_indices.append(col)
                out_data.append(row_data[mask].sum())
        out_indptr[row + 1] = len(out_data)

    if out_data:
        data_arr = np.asarray(out_data, dtype=data_np.dtype)
        indices_arr = np.asarray(out_indices, dtype=indices_np.dtype)
    else:
        data_arr = np.empty((0,), dtype=data_np.dtype)
        indices_arr = np.empty((0,), dtype=indices_np.dtype)

    return (
        to_mx(data_arr, dtype=data.dtype),
        to_mx(indices_arr, dtype=indices.dtype),
        to_mx(out_indptr, dtype=indptr.dtype),
    )


def sum_csc_duplicates(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    data_np = to_numpy(data)
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)

    out_data = []
    out_indices = []
    out_indptr = np.zeros_like(indptr_np)

    for col in range(indptr_np.size - 1):
        start = int(indptr_np[col])
        end = int(indptr_np[col + 1])
        col_indices = indices_np[start:end]
        col_data = data_np[start:end]
        if col_indices.size:
            unique, first = np.unique(col_indices, return_index=True)
            order = np.argsort(first)
            for row in unique[order]:
                mask = col_indices == row
                out_indices.append(row)
                out_data.append(col_data[mask].sum())
        out_indptr[col + 1] = len(out_data)

    if out_data:
        data_arr = np.asarray(out_data, dtype=data_np.dtype)
        indices_arr = np.asarray(out_indices, dtype=indices_np.dtype)
    else:
        data_arr = np.empty((0,), dtype=data_np.dtype)
        indices_arr = np.empty((0,), dtype=indices_np.dtype)

    return (
        to_mx(data_arr, dtype=data.dtype),
        to_mx(indices_arr, dtype=indices.dtype),
        to_mx(out_indptr, dtype=indptr.dtype),
    )


def fromdense(
    dense: mx.array,
    *,
    index_dtype,
    threshold: float,
):
    dense_np = to_numpy(dense)
    if index_dtype == mx.int32:
        index_np_dtype = np.int32
    elif index_dtype == mx.int64:
        index_np_dtype = np.int64
    else:
        raise TypeError(f"index_dtype must be mx.int32 or mx.int64, got {index_dtype}.")

    if threshold == 0:
        mask = dense_np != 0
    else:
        mask = np.abs(dense_np) > threshold
    row, col = np.nonzero(mask)
    data = dense_np[row, col]

    indptr = np.zeros(dense_np.shape[0] + 1, dtype=index_np_dtype)
    if row.size:
        counts = np.bincount(row.astype(np.int64), minlength=dense_np.shape[0])
        indptr[1:] = np.cumsum(counts, dtype=index_np_dtype)

    return (
        to_mx(data, dtype=dense.dtype),
        to_mx(col.astype(index_np_dtype, copy=False), dtype=index_dtype),
        to_mx(indptr, dtype=index_dtype),
    )
