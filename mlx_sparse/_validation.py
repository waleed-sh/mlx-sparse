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

from collections.abc import Sequence

import mlx.core as mx

from mlx_sparse._host import to_numpy
from mlx_sparse._typing import INDEX_DTYPES, VALUE_DTYPES, Shape2D, ValidationMode


def normalize_shape(shape: Sequence[int]) -> Shape2D:
    if len(shape) != 2:
        raise ValueError(f"sparse arrays must be rank-2, got shape={tuple(shape)!r}.")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"sparse dimensions must be non-negative, got {shape!r}.")
    return n_rows, n_cols


def normalize_validation_mode(validate: ValidationMode) -> str:
    if validate is True:
        return "full"
    if validate is False:
        return "none"
    if validate not in {"metadata", "full"}:
        raise ValueError(
            "validate must be one of False, True, 'metadata', or 'full', "
            f"got {validate!r}."
        )
    return validate


def ensure_mx_array(value, *, dtype=None) -> mx.array:
    if isinstance(value, mx.array):
        if dtype is not None and value.dtype != dtype:
            return value.astype(dtype)
        return value
    if dtype is None:
        return mx.array(value)
    return mx.array(value, dtype=dtype)


def check_rank1(name: str, array: mx.array) -> None:
    if array.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape={array.shape}.")


def check_index_dtype(name: str, array: mx.array) -> None:
    if array.dtype not in INDEX_DTYPES:
        allowed = ", ".join(str(dt).split(".")[-1] for dt in INDEX_DTYPES)
        raise TypeError(f"{name} dtype must be one of {allowed}, got {array.dtype}.")


def check_value_dtype(name: str, array: mx.array) -> None:
    if array.dtype not in VALUE_DTYPES:
        allowed = ", ".join(str(dt).split(".")[-1] for dt in VALUE_DTYPES)
        raise TypeError(f"{name} dtype must be one of {allowed}, got {array.dtype}.")


def validate_csr_metadata(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> None:
    check_rank1("CSRArray.data", data)
    check_rank1("CSRArray.indices", indices)
    check_rank1("CSRArray.indptr", indptr)
    check_value_dtype("CSRArray.data", data)
    check_index_dtype("CSRArray.indices", indices)
    check_index_dtype("CSRArray.indptr", indptr)

    if indices.dtype != indptr.dtype:
        raise TypeError(
            "CSRArray indices and indptr must have the same dtype, "
            f"got {indices.dtype} and {indptr.dtype}."
        )
    if data.shape[0] != indices.shape[0]:
        raise ValueError(
            "CSRArray data and indices must have the same length, "
            f"got {data.shape[0]} and {indices.shape[0]}."
        )
    expected_indptr = shape[0] + 1
    if indptr.shape[0] != expected_indptr:
        raise ValueError(
            "CSRArray indptr must have shape (n_rows + 1,), "
            f"got {indptr.shape} for n_rows={shape[0]}."
        )


def validate_csr_values(
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    nnz: int,
) -> None:
    indices_np = to_numpy(indices)
    indptr_np = to_numpy(indptr)
    if indptr_np[0] != 0:
        raise ValueError(f"CSRArray indptr[0] must be 0, got {indptr_np[0]}.")
    if indptr_np[-1] != nnz:
        raise ValueError(
            f"CSRArray indptr[-1] must equal nnz={nnz}, got {indptr_np[-1]}."
        )
    if (indptr_np[1:] < indptr_np[:-1]).any():
        raise ValueError("CSRArray indptr must be monotonically nondecreasing.")
    if indices_np.size:
        min_index = int(indices_np.min())
        max_index = int(indices_np.max())
        if min_index < 0 or max_index >= shape[1]:
            raise ValueError(
                "CSRArray indices must be in bounds for n_cols="
                f"{shape[1]}, got min={min_index}, max={max_index}."
            )


def validate_coo_metadata(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
) -> None:
    check_rank1("COOArray.data", data)
    check_rank1("COOArray.row", row)
    check_rank1("COOArray.col", col)
    check_value_dtype("COOArray.data", data)
    check_index_dtype("COOArray.row", row)
    check_index_dtype("COOArray.col", col)

    if row.dtype != col.dtype:
        raise TypeError(
            f"COOArray row and col dtypes must match, got {row.dtype} and {col.dtype}."
        )
    if data.shape[0] != row.shape[0] or data.shape[0] != col.shape[0]:
        raise ValueError(
            "COOArray data, row, and col must have the same length, "
            f"got {data.shape[0]}, {row.shape[0]}, and {col.shape[0]}."
        )


def validate_coo_values(row: mx.array, col: mx.array, shape: Shape2D) -> None:
    row_np = to_numpy(row)
    col_np = to_numpy(col)
    if row_np.size:
        min_row = int(row_np.min())
        max_row = int(row_np.max())
        if min_row < 0 or max_row >= shape[0]:
            raise ValueError(
                "COOArray row coordinates must be in bounds for n_rows="
                f"{shape[0]}, got min={min_row}, max={max_row}."
            )
    if col_np.size:
        min_col = int(col_np.min())
        max_col = int(col_np.max())
        if min_col < 0 or max_col >= shape[1]:
            raise ValueError(
                "COOArray col coordinates must be in bounds for n_cols="
                f"{shape[1]}, got min={min_col}, max={max_col}."
            )


def validate_csr_matvec_inputs(data, indices, indptr, x, shape: Shape2D) -> None:
    validate_csr_metadata(data, indices, indptr, shape)
    if x.ndim != 1:
        raise ValueError(f"csr_matvec expects a rank-1 RHS, got shape={x.shape}.")
    if x.shape[0] != shape[1]:
        raise ValueError(
            f"csr_matvec RHS has length {x.shape[0]}, but sparse n_cols={shape[1]}."
        )
    check_value_dtype("csr_matvec data", data)
    check_value_dtype("csr_matvec RHS", x)
    if data.dtype != x.dtype:
        raise TypeError(
            "csr_matvec requires sparse data and RHS to have the same dtype, "
            f"got {data.dtype} and {x.dtype}."
        )


def validate_csr_matmul_inputs(data, indices, indptr, rhs, shape: Shape2D) -> None:
    validate_csr_metadata(data, indices, indptr, shape)
    if rhs.ndim != 2:
        raise ValueError(f"csr_matmul expects a rank-2 RHS, got shape={rhs.shape}.")
    if rhs.shape[0] != shape[1]:
        raise ValueError(
            f"csr_matmul RHS has leading dimension {rhs.shape[0]}, "
            f"but sparse n_cols={shape[1]}."
        )
    check_value_dtype("csr_matmul data", data)
    check_value_dtype("csr_matmul RHS", rhs)
    if data.dtype != rhs.dtype:
        raise TypeError(
            "csr_matmul requires sparse data and RHS to have the same dtype, "
            f"got {data.dtype} and {rhs.dtype}."
        )
