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
from dataclasses import dataclass
from operator import index as operator_index

import mlx.core as mx
import numpy as np

import mlx_sparse._native as _native
from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._host import to_mx, to_numpy
from mlx_sparse._typing import INDEX_DTYPES, VALUE_DTYPES, Shape2D
from mlx_sparse._validation import ensure_mx_array, normalize_shape


def _numpy_index_dtype(index_dtype):
    if index_dtype == mx.int32:
        return np.int32
    if index_dtype == mx.int64:
        return np.int64
    raise TypeError(f"index_dtype must be mx.int32 or mx.int64, got {index_dtype}.")


def _normalize_value_dtype(dtype):
    if dtype is None:
        return mx.float32
    if dtype not in VALUE_DTYPES:
        raise TypeError(
            "dtype must be one of mx.float32, mx.float16, mx.bfloat16, "
            f"or mx.complex64, got {dtype}."
        )
    return dtype


def _numpy_value_dtype(dtype):
    dtype = _normalize_value_dtype(dtype)
    if dtype == mx.float32:
        return np.float32
    if dtype == mx.float16:
        return np.float16
    if dtype == mx.complex64:
        return np.complex64
    if dtype == mx.bfloat16:
        # NumPy has no portable bfloat16 dtype. Build from float32 host values
        # and cast to bfloat16 when creating the MLX array.
        return np.float32
    raise TypeError(
        "dtype must be one of mx.float32, mx.float16, mx.bfloat16, "
        f"or mx.complex64, got {dtype}."
    )


def _infer_value_dtype_from_numpy(array: np.ndarray):
    if np.iscomplexobj(array):
        return mx.complex64
    if array.dtype == np.float16:
        return mx.float16
    return mx.float32


def _infer_diagonal_array_dtype(diag):
    if isinstance(diag, mx.array):
        return _infer_dense_constructor_dtype(diag)
    diag_np = np.asarray(diag)
    if np.iscomplexobj(diag_np):
        return mx.complex64
    if diag_np.dtype == np.float16:
        return mx.float16
    return mx.float32


def _infer_diagonal_dtype(diagonal_arrays: Sequence[object]):
    inferred = None
    for diag in diagonal_arrays:
        inferred = _promote_constructor_dtype(
            inferred, _infer_diagonal_array_dtype(diag)
        )
    return inferred if inferred is not None else mx.float32


def _diagonal_size(diag) -> int:
    return int(diag.size if isinstance(diag, mx.array) else np.asarray(diag).size)


def _diagonal_to_mx(diag, *, dtype):
    if isinstance(diag, mx.array):
        return diag.astype(dtype) if diag.dtype != dtype else diag
    return to_mx(np.asarray(diag), dtype=dtype)


def _normalize_index_dtype(index_dtype):
    if index_dtype not in INDEX_DTYPES:
        raise TypeError(f"index_dtype must be mx.int32 or mx.int64, got {index_dtype}.")
    return index_dtype


_SUPPORTED_STRUCTURAL_FORMATS = {"coo", "csr", "csc"}
_UNSUPPORTED_SCIPY_FORMATS = {"bsr", "dia", "dok", "lil"}


@dataclass(frozen=True)
class _RawBlock:
    value: object
    shape: Shape2D
    dtype: object
    is_sparse: bool


def _normalize_sparse_format(function_name: str, format, *, default: str) -> str:
    if format is None:
        return default
    if not isinstance(format, str):
        raise TypeError(f"{function_name} format must be a string or None.")
    normalized = format.lower()
    if normalized in _SUPPORTED_STRUCTURAL_FORMATS:
        return normalized
    if normalized in _UNSUPPORTED_SCIPY_FORMATS:
        raise NotImplementedError(
            f"{function_name} format={format!r} is not implemented in mlx-sparse; "
            "supported formats are 'coo', 'csr', and 'csc'."
        )
    raise ValueError(
        f"{function_name} format must be one of 'coo', 'csr', or 'csc', "
        f"got {format!r}."
    )


def _promote_constructor_dtype(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    if lhs == mx.complex64 or rhs == mx.complex64:
        return mx.complex64
    if lhs == mx.float32 or rhs == mx.float32:
        return mx.float32
    if lhs == rhs:
        return lhs
    return mx.float32


def _infer_dense_constructor_dtype(dense: mx.array):
    return dense.dtype if dense.dtype in VALUE_DTYPES else mx.float32


def _constructor_dtype(raw_blocks: Sequence[_RawBlock], dtype):
    if dtype is not None:
        return _normalize_value_dtype(dtype)
    inferred = None
    for block in raw_blocks:
        inferred = _promote_constructor_dtype(inferred, block.dtype)
    return inferred if inferred is not None else mx.float32


def _constructor_index_dtype(raw_blocks: Sequence[_RawBlock]):
    for block in raw_blocks:
        if (
            block.is_sparse
            and getattr(block.value, "index_dtype", mx.int32) == mx.int64
        ):
            return mx.int64
    return mx.int32


def _is_sparse_array(value) -> bool:
    return isinstance(value, (COOArray, CSRArray, CSCArray))


def _as_raw_block(function_name: str, value) -> _RawBlock:
    if _is_sparse_array(value):
        return _RawBlock(
            value=value,
            shape=value.shape,
            dtype=value.data.dtype,
            is_sparse=True,
        )

    dense = ensure_mx_array(value)
    if dense.ndim != 2:
        raise ValueError(
            f"{function_name} blocks must be sparse arrays or dense rank-2 arrays, "
            f"got shape={dense.shape}."
        )
    return _RawBlock(
        value=dense,
        shape=normalize_shape(dense.shape),
        dtype=_infer_dense_constructor_dtype(dense),
        is_sparse=False,
    )


def _cast_sparse_value(array, dtype):
    if array.data.dtype == dtype:
        return array
    if isinstance(array, COOArray):
        return COOArray(
            data=array.data.astype(dtype),
            row=array.row,
            col=array.col,
            shape=array.shape,
            has_canonical_format=array.has_canonical_format,
        )
    if isinstance(array, CSRArray):
        return CSRArray(
            data=array.data.astype(dtype),
            indices=array.indices,
            indptr=array.indptr,
            shape=array.shape,
            sorted_indices=array.sorted_indices,
            has_canonical_format=array.has_canonical_format,
        )
    if isinstance(array, CSCArray):
        return CSCArray(
            data=array.data.astype(dtype),
            indices=array.indices,
            indptr=array.indptr,
            shape=array.shape,
            sorted_indices=array.sorted_indices,
            has_canonical_format=array.has_canonical_format,
        )
    raise TypeError(f"Expected sparse array, got {type(array).__name__}.")


def _cast_coo_index(array: COOArray, index_dtype):
    if array.row.dtype == index_dtype and array.col.dtype == index_dtype:
        return array
    return COOArray(
        data=array.data,
        row=array.row.astype(index_dtype),
        col=array.col.astype(index_dtype),
        shape=array.shape,
        has_canonical_format=array.has_canonical_format,
    )


def _raw_block_to_coo(block: _RawBlock, *, dtype, index_dtype) -> COOArray:
    if block.is_sparse:
        sparse = _cast_sparse_value(block.value, dtype)
        if isinstance(sparse, COOArray):
            coo = sparse
        elif isinstance(sparse, CSRArray):
            coo = sparse.tocoo(canonical=None)
        elif isinstance(sparse, CSCArray):
            coo = sparse.tocoo(canonical=False)
        else:
            raise TypeError(f"Expected sparse array, got {type(sparse).__name__}.")
        return _cast_coo_index(coo, index_dtype)

    csr = fromdense(block.value, dtype=dtype, index_dtype=index_dtype)
    return csr.tocoo(canonical=True)


def _empty_coo(shape: Shape2D, *, dtype, index_dtype) -> COOArray:
    return COOArray(
        data=mx.zeros((0,), dtype=dtype),
        row=mx.zeros((0,), dtype=index_dtype),
        col=mx.zeros((0,), dtype=index_dtype),
        shape=shape,
        has_canonical_format=True,
    )


def _coo_to_format(coo: COOArray, format: str):
    if format == "coo":
        return coo
    if format == "csr":
        return coo.tocsr(canonical=True)
    if format == "csc":
        return coo.tocsc(canonical=True)
    raise ValueError(f"unsupported sparse format {format!r}.")


def _assemble_offset_blocks(
    blocks: Sequence[_RawBlock],
    row_offsets: Sequence[int],
    col_offsets: Sequence[int],
    shape: Shape2D,
    *,
    function_name: str,
    format,
    dtype,
) -> COOArray | CSRArray | CSCArray:
    out_format = _normalize_sparse_format(function_name, format, default="coo")
    shape = normalize_shape(shape)
    if len(blocks) != len(row_offsets) or len(blocks) != len(col_offsets):
        raise ValueError(
            f"{function_name} internal block and offset counts do not match."
        )

    value_dtype = _constructor_dtype(blocks, dtype)
    index_dtype = _constructor_index_dtype(blocks)
    coo_blocks: list[COOArray] = []
    kept_row_offsets: list[int] = []
    kept_col_offsets: list[int] = []
    all_canonical = True
    for block, row_offset, col_offset in zip(
        blocks, row_offsets, col_offsets, strict=True
    ):
        if row_offset < 0 or col_offset < 0:
            raise ValueError(f"{function_name} block offsets must be non-negative.")
        coo = _raw_block_to_coo(block, dtype=value_dtype, index_dtype=index_dtype)
        all_canonical = all_canonical and bool(coo.has_canonical_format)
        if coo.nnz == 0:
            continue
        coo_blocks.append(coo)
        kept_row_offsets.append(int(row_offset))
        kept_col_offsets.append(int(col_offset))

    if not coo_blocks:
        return _coo_to_format(
            _empty_coo(shape, dtype=value_dtype, index_dtype=index_dtype),
            out_format,
        )

    data, row, col = _native.coo_block(
        coo_blocks,
        kept_row_offsets,
        kept_col_offsets,
        shape,
    )
    return _coo_to_format(
        COOArray(
            data=data,
            row=row,
            col=col,
            shape=shape,
            has_canonical_format=all_canonical,
        ),
        out_format,
    )


def _normalize_block_grid(blocks) -> list[list[object]]:
    if isinstance(blocks, (COOArray, CSRArray, CSCArray, mx.array)):
        raise ValueError("block_array blocks must be a 2-D grid, got rank-2 input.")
    try:
        rows = [list(row) for row in blocks]
    except TypeError as exc:
        raise ValueError("block_array blocks must be a 2-D grid.") from exc
    if not rows:
        raise ValueError("block_array blocks must be 2-D and non-empty.")
    n_cols = len(rows[0])
    if n_cols == 0:
        raise ValueError("block_array block rows must be non-empty.")
    for row in rows:
        if len(row) != n_cols:
            raise ValueError("block_array requires a rectangular block grid.")
    return rows


def _validate_block_grid(
    blocks,
) -> tuple[list[_RawBlock], list[int], list[int], Shape2D]:
    grid = _normalize_block_grid(blocks)
    n_block_rows = len(grid)
    n_block_cols = len(grid[0])
    row_heights: list[int | None] = [None] * n_block_rows
    col_widths: list[int | None] = [None] * n_block_cols
    raw_by_position: list[tuple[_RawBlock, int, int]] = []

    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value is None:
                continue
            raw = _as_raw_block("block_array", value)
            height, width = raw.shape
            if row_heights[i] is None:
                row_heights[i] = height
            elif row_heights[i] != height:
                raise ValueError(
                    f"block_array row {i} has incompatible block heights "
                    f"{row_heights[i]} and {height}."
                )
            if col_widths[j] is None:
                col_widths[j] = width
            elif col_widths[j] != width:
                raise ValueError(
                    f"block_array column {j} has incompatible block widths "
                    f"{col_widths[j]} and {width}."
                )
            raw_by_position.append((raw, i, j))

    resolved_row_heights = [0 if height is None else height for height in row_heights]
    resolved_col_widths = [0 if width is None else width for width in col_widths]
    row_starts = [0]
    for height in resolved_row_heights:
        row_starts.append(row_starts[-1] + height)
    col_starts = [0]
    for width in resolved_col_widths:
        col_starts.append(col_starts[-1] + width)

    raw_blocks = [raw for raw, _, _ in raw_by_position]
    row_offsets = [row_starts[i] for _, i, _ in raw_by_position]
    col_offsets = [col_starts[j] for _, _, j in raw_by_position]
    return raw_blocks, row_offsets, col_offsets, (row_starts[-1], col_starts[-1])


def _csr_from_sorted_triplets(
    data,
    row: np.ndarray,
    col: np.ndarray,
    shape: Shape2D,
    *,
    dtype,
    index_dtype,
) -> CSRArray:
    index_np_dtype = _numpy_index_dtype(index_dtype)
    indptr = np.zeros(shape[0] + 1, dtype=index_np_dtype)
    if row.size:
        counts = np.bincount(row.astype(np.int64), minlength=shape[0])
        indptr[1:] = np.cumsum(counts, dtype=index_np_dtype)

    return CSRArray(
        data=_diagonal_to_mx(data, dtype=dtype),
        indices=to_mx(col.astype(index_np_dtype, copy=False), dtype=index_dtype),
        indptr=to_mx(indptr, dtype=index_dtype),
        shape=shape,
        sorted_indices=True,
        has_canonical_format=True,
    )


def eye(
    n: int,
    m: int | None = None,
    *,
    k: int = 0,
    dtype=mx.float32,
    index_dtype=mx.int32,
) -> CSRArray:
    """Return a sparse identity-like CSR matrix with ones on a specified diagonal.

    Produces the same result as :func:`numpy.eye` with ``k=k``, but returns a
    :class:`~mlx_sparse.CSRArray` instead of a dense array. The matrix has at
    most ``min(n, m)`` stored values. Rows (or columns) that the diagonal does
    not pass through are empty rows in the CSR representation.

    Args:
        n: Number of rows.
        m: Number of columns. Defaults to ``n``, producing a square matrix.
        k: Diagonal offset. ``0`` selects the main diagonal. Positive values
            shift the diagonal above the main diagonal (superdiagonal). Negative
            values shift it below (subdiagonal).
        dtype: Value dtype for the stored ones. Must be one of ``mx.float32``,
            ``mx.float16``, ``mx.bfloat16``, or ``mx.complex64``. Defaults to
            ``mx.float32``.
        index_dtype: Integer dtype for ``indices`` and ``indptr``. Must be
            ``mx.int32`` or ``mx.int64``. Defaults to ``mx.int32``.

    Returns:
        A canonical :class:`~mlx_sparse.CSRArray` with ``has_canonical_format=True``
        and ``sorted_indices=True``.

    Raises:
        TypeError: If ``dtype`` or ``index_dtype`` is not a supported value.

    Example::

        import mlx_sparse as ms
        import mlx.core as mx

        # 4x4 identity matrix
        I = ms.eye(4)
        mx.eval(I.data)
        # CSRArray(shape=(4, 4), nnz=4, ...)

        # 3x5 matrix with ones on the first superdiagonal
        A = ms.eye(3, 5, k=1)
        # Non-zeros at (0,1), (1,2), (2,3)
    """
    n = int(n)
    m = n if m is None else int(m)
    shape = normalize_shape((n, m))
    dtype = _normalize_value_dtype(dtype)
    index_dtype = _normalize_index_dtype(index_dtype)
    index_np_dtype = _numpy_index_dtype(index_dtype)

    row_start = max(0, -int(k))
    col_start = max(0, int(k))
    nnz = max(0, min(shape[0] - row_start, shape[1] - col_start))
    row = row_start + np.arange(nnz, dtype=index_np_dtype)
    col = col_start + np.arange(nnz, dtype=index_np_dtype)
    data = np.ones(nnz, dtype=np.complex64 if dtype == mx.complex64 else np.float32)

    return _csr_from_sorted_triplets(
        data,
        row,
        col,
        shape,
        dtype=dtype,
        index_dtype=index_dtype,
    )


def identity(
    n: int,
    dtype=None,
    format=None,
    *,
    index_dtype=mx.int32,
):
    """Return a sparse square identity matrix.

    ``identity(n)`` is a SciPy-compatible square alias for
    :func:`mlx_sparse.eye`. The default output format is CSR, matching
    ``eye(n)``. Pass ``format="coo"`` or ``format="csc"`` to request another
    supported sparse format. Unsupported SciPy formats such as ``"dia"`` and
    ``"bsr"`` are rejected explicitly.

    Args:
        n: Number of rows and columns. Must be non-negative.
        dtype: Stored value dtype. ``None`` defaults to ``mx.float32``.
        format: Output format, one of ``None``, ``"csr"``, ``"coo"``, or
            ``"csc"``. ``None`` returns CSR.
        index_dtype: Integer dtype for sparse indices, ``mx.int32`` or
            ``mx.int64``.

    Returns:
        A sparse square identity matrix in the requested format.
    """
    out_format = _normalize_sparse_format("identity", format, default="csr")
    shape = normalize_shape((int(n), int(n)))
    dtype = _normalize_value_dtype(mx.float32 if dtype is None else dtype)
    index_dtype = _normalize_index_dtype(index_dtype)
    data = mx.ones((shape[0],), dtype=dtype)
    indices = mx.arange(shape[0], dtype=index_dtype)
    indptr = mx.arange(shape[0] + 1, dtype=index_dtype)
    csr = CSRArray(
        data=data,
        indices=indices,
        indptr=indptr,
        shape=shape,
        sorted_indices=True,
        has_canonical_format=True,
    )
    if out_format == "csr":
        return csr
    if out_format == "coo":
        return csr.tocoo(canonical=True)
    return csr.tocsc(canonical=True)


def _as_diagonal_sequence(diagonals) -> list[object]:
    if isinstance(diagonals, mx.array):
        if diagonals.ndim == 0:
            return [mx.reshape(diagonals, (1,))]
        if diagonals.ndim == 1:
            return [diagonals]
        if diagonals.ndim == 2:
            return [diagonals[i] for i in range(diagonals.shape[0])]
    if np.isscalar(diagonals):
        return [np.asarray([diagonals])]
    if isinstance(diagonals, np.ndarray):
        if diagonals.ndim == 0:
            return [diagonals.reshape(1)]
        if diagonals.ndim == 1:
            return [diagonals]
        if diagonals.ndim == 2:
            return [row for row in diagonals]
    if isinstance(diagonals, Sequence):
        if not diagonals:
            return []
        first = diagonals[0]
        if np.isscalar(first) or isinstance(first, mx.array) and first.ndim == 0:
            if any(isinstance(d, mx.array) for d in diagonals):
                parts = []
                for d in diagonals:
                    if isinstance(d, mx.array):
                        if d.ndim != 0:
                            raise ValueError(
                                "diags scalar diagonal sequences cannot mix "
                                "scalar and non-scalar MLX arrays."
                            )
                        parts.append(mx.reshape(d, (1,)))
                    else:
                        parts.append(mx.array([d]))
                return [mx.concatenate(parts, axis=0)]
            return [
                np.asarray(
                    [
                        to_numpy(d).item() if isinstance(d, mx.array) else d
                        for d in diagonals
                    ]
                )
            ]
        return [d if isinstance(d, mx.array) else np.asarray(d) for d in diagonals]
    return [np.asarray(diagonals)]


def diags(
    diagonals,
    offsets=0,
    *,
    shape: Sequence[int] | None = None,
    dtype=None,
    index_dtype=mx.int32,
) -> CSRArray:
    """Construct a CSR matrix from one or more diagonals.

    Mirrors the behaviour of :func:`scipy.sparse.diags` but returns a
    :class:`~mlx_sparse.CSRArray`. Each diagonal is placed at the position
    specified by the corresponding offset. Diagonals are assembled into a COO
    triple and sorted before the CSR row-pointer array is built, so the result
    is always in canonical form. When diagonal values are MLX arrays, the fixed
    diagonal topology is assembled without converting those values to NumPy, so
    ``mx.jvp`` and ``mx.vjp`` propagate sparse-value tangents and cotangents.

    Args:
        diagonals: The diagonal values. Accepted forms:

            - A single 1-D array-like (or scalar) placed at ``offsets``.
            - A 2-D array whose rows are individual diagonals.
            - A list of 1-D array-likes, one per entry in ``offsets``.

            Each diagonal's length must not exceed the number of elements that
            the diagonal at the corresponding offset can hold given ``shape``.

        offsets: Diagonal offset(s). ``0`` is the main diagonal. Positive
            integers are superdiagonals. Negative integers are subdiagonals.
            When ``diagonals`` is a list, ``offsets`` must be a matching list
            of integers. Repeated offsets are not allowed.
        shape: Output matrix shape as ``(n_rows, n_cols)``. When omitted, the
            minimum square shape that fits all diagonals is inferred
            automatically.
        dtype: Value dtype. When ``None`` (default), the dtype is inferred from
            the diagonal arrays: ``complex64`` if any diagonal is complex,
            ``float16`` if any diagonal has dtype ``float16``, otherwise
            ``float32``.
        index_dtype: Integer dtype for ``indices`` and ``indptr``. Must be
            ``mx.int32`` or ``mx.int64``. Defaults to ``mx.int32``.

    Returns:
        A canonical :class:`~mlx_sparse.CSRArray` with ``has_canonical_format=True``
        and ``sorted_indices=True``.

    Raises:
        TypeError: If ``dtype`` or ``index_dtype`` is not supported.
        ValueError: If the number of diagonals and offsets differ, if offsets
            are repeated, or if a diagonal is longer than its allocated space.

    Example::

        import numpy as np
        import mlx_sparse as ms
        import mlx.core as mx

        # Tridiagonal matrix: main diagonal 2, off-diagonals -1
        A = ms.diags(
            [np.full(4, -1.0), np.full(5, 2.0), np.full(4, -1.0)],
            offsets=[-1, 0, 1],
        )
        # 5x5, nnz=13

        # Single diagonal at offset 2
        B = ms.diags([1.0, 2.0, 3.0], offsets=2, shape=(5, 5))
    """
    diagonal_arrays = _as_diagonal_sequence(diagonals)
    if np.isscalar(offsets):
        offsets_array = np.asarray([int(offsets)], dtype=np.int64)
    else:
        offsets_array = np.asarray(list(offsets), dtype=np.int64)

    if len(diagonal_arrays) != offsets_array.size:
        raise ValueError(
            "diags requires the same number of diagonals and offsets, "
            f"got {len(diagonal_arrays)} and {offsets_array.size}."
        )
    if len(set(offsets_array.tolist())) != offsets_array.size:
        raise ValueError("diags does not allow repeated offsets.")

    dtype = _infer_diagonal_dtype(diagonal_arrays) if dtype is None else dtype
    dtype = _normalize_value_dtype(dtype)
    index_dtype = _normalize_index_dtype(index_dtype)
    index_np_dtype = _numpy_index_dtype(index_dtype)

    if shape is None:
        dim = 0
        for diag, offset in zip(diagonal_arrays, offsets_array, strict=True):
            dim = max(dim, _diagonal_size(diag) + abs(int(offset)))
        shape_2d = (dim, dim)
    else:
        shape_2d = normalize_shape(shape)

    data_parts = []
    row_parts = []
    col_parts = []
    for diag, offset in zip(diagonal_arrays, offsets_array, strict=True):
        offset = int(offset)
        row_start = max(0, -offset)
        col_start = max(0, offset)
        capacity = max(0, min(shape_2d[0] - row_start, shape_2d[1] - col_start))
        diag_size = _diagonal_size(diag)
        if diag_size > capacity:
            raise ValueError(
                f"diagonal at offset {offset} has length {diag_size}, "
                f"but shape {shape_2d} can hold at most {capacity} values."
            )
        nnz = int(diag_size)
        if nnz == 0:
            continue
        positions = np.arange(nnz, dtype=index_np_dtype)
        row_parts.append(row_start + positions)
        col_parts.append(col_start + positions)
        data_parts.append(_diagonal_to_mx(diag, dtype=dtype))

    if data_parts:
        row = np.concatenate(row_parts).astype(index_np_dtype, copy=False)
        col = np.concatenate(col_parts).astype(index_np_dtype, copy=False)
        if len(data_parts) == 1:
            data = data_parts[0]
        else:
            data = mx.concatenate(data_parts, axis=0)
        order = np.lexsort((col, row))
        row = row[order]
        col = col[order]
        data = mx.take(data, to_mx(order.astype(index_np_dtype), dtype=index_dtype))
    else:
        row = np.empty((0,), dtype=index_np_dtype)
        col = np.empty((0,), dtype=index_np_dtype)
        data = mx.zeros((0,), dtype=dtype)

    return _csr_from_sorted_triplets(
        data,
        row,
        col,
        shape_2d,
        dtype=dtype,
        index_dtype=index_dtype,
    )


def fromdense(
    array,
    *,
    threshold: float = 0.0,
    dtype=None,
    index_dtype=mx.int32,
) -> CSRArray:
    """Construct a canonical CSR matrix from a rank-2 dense MLX array.

    Identifies the non-zero (or above-threshold) entries of a dense matrix and
    packages them into a :class:`~mlx_sparse.CSRArray`. The native path stages
    this as count, allocate, then fill work so Metal builds can perform the
    dense scan and CSR writes on device while still returning compact buffers.
    Because the output sparse topology depends on numerical values and the
    threshold, ``fromdense`` is intentionally not differentiable; use fixed-
    topology constructors when sparse values need gradients.

    The value dtype is preserved from the input array. Index dtype defaults to
    ``int32`` and can be overridden for matrices with more than ~2 billion
    non-zeros (not typical on Apple Silicon).

    Args:
        array: A rank-2 array-like. Converted to ``mlx.core.array`` if not
            already. Dtype must be one of ``float32``, ``float16``,
            ``bfloat16``, or ``complex64``.
        threshold: Entries with absolute value less than or equal to
            ``threshold`` are treated as structural zeros and excluded from
            the output. The default ``0.0`` keeps every numerically non-zero
            entry. Must be non-negative.
        dtype: Optional value dtype to cast to before extracting non-zeros.
            When ``None``, the input dtype chosen by MLX is preserved.
        index_dtype: Integer dtype for ``indices`` and ``indptr``. Must be
            ``mx.int32`` or ``mx.int64``. Defaults to ``mx.int32``.

    Returns:
        A canonical :class:`~mlx_sparse.CSRArray` with ``has_canonical_format=True``
        and ``sorted_indices=True``.

    Raises:
        TypeError: If the input dtype is not a supported value dtype.
        ValueError: If the input is not rank-2, or if ``threshold`` is
            negative.

    Example::

        import mlx.core as mx
        import numpy as np
        import mlx_sparse as ms

        dense = mx.array(np.array([
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ], dtype=np.float32))

        csr = ms.fromdense(dense)
        # CSRArray(shape=(3, 3), nnz=4, dtype=float32, ...)

        # Drop near-zero entries below 0.1
        csr_thresholded = ms.fromdense(dense, threshold=0.5)
    """
    dtype = None if dtype is None else _normalize_value_dtype(dtype)
    dense = ensure_mx_array(array, dtype=dtype)
    if dense.ndim != 2:
        raise ValueError(f"fromdense expects a rank-2 array, got shape={dense.shape}.")
    if dense.dtype not in VALUE_DTYPES:
        raise TypeError(
            "fromdense input dtype must be float32, float16, bfloat16, "
            f"or complex64, got {dense.dtype}."
        )
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}.")

    index_dtype = _normalize_index_dtype(index_dtype)
    data, indices, indptr = _native.csr_fromdense(
        dense,
        index_dtype=index_dtype,
        threshold=float(threshold),
    )
    return CSRArray(
        data,
        indices,
        indptr,
        shape=(int(dense.shape[0]), int(dense.shape[1])),
        sorted_indices=True,
        has_canonical_format=True,
    )


def from_dense(
    array,
    *,
    threshold: float = 0.0,
    dtype=None,
    index_dtype=mx.int32,
) -> CSRArray:
    """Alias for :func:`fromdense` with a PEP 8 compatible name."""
    return fromdense(
        array,
        threshold=threshold,
        dtype=dtype,
        index_dtype=index_dtype,
    )


def from_scipy(
    matrix,
    *,
    format: str = "csr",
    dtype=None,
    index_dtype=mx.int32,
    canonical: bool = True,
):
    """Convert a SciPy sparse matrix or sparse array to mlx-sparse.

    Any SciPy sparse format is accepted. ``format="csr"`` returns a
    :class:`~mlx_sparse.CSRArray`, ``format="csc"`` returns a
    :class:`~mlx_sparse.CSCArray`, and ``format="coo"`` returns a
    :class:`~mlx_sparse.COOArray`. The conversion preserves supported
    ``float32``, ``float16``, and ``complex64`` values. Other real floating
    dtypes, including SciPy's default ``float64``, are cast to ``float32``
    unless ``dtype`` is provided.

    Args:
        matrix: A ``scipy.sparse`` matrix or array.
        format: Output sparse format: ``"csr"`` (default), ``"csc"``, or
            ``"coo"``.
        dtype: Optional MLX value dtype. Must be one of ``mx.float32``,
            ``mx.float16``, ``mx.bfloat16``, or ``mx.complex64``.
        index_dtype: Integer dtype for sparse indices. Must be ``mx.int32`` or
            ``mx.int64``.
        canonical: If ``True`` (default), sum duplicates and sort indices
            before exporting buffers.

    Returns:
        A ``CSRArray``, ``CSCArray``, or ``COOArray``.

    Raises:
        TypeError: If SciPy is not installed, ``matrix`` is not sparse, or a
            dtype is unsupported.
        ValueError: If ``format`` is not ``"csr"``, ``"csc"``, or ``"coo"``.
    """
    try:
        import scipy.sparse as sp
    except ImportError as exc:
        raise TypeError("from_scipy requires scipy to be installed.") from exc

    if not sp.issparse(matrix):
        raise TypeError(
            "from_scipy expects a scipy.sparse matrix or array, "
            f"got {type(matrix).__name__}."
        )

    out_format = format.lower()
    if out_format not in {"csr", "csc", "coo"}:
        raise ValueError("format must be 'csr', 'csc', or 'coo'.")

    index_dtype = _normalize_index_dtype(index_dtype)
    index_np_dtype = _numpy_index_dtype(index_dtype)

    if out_format == "csc":
        csc = matrix.tocsc(copy=True)
        if canonical:
            csc.sum_duplicates()
            csc.sort_indices()

        value_dtype = (
            _infer_value_dtype_from_numpy(np.asarray(csc.data))
            if dtype is None
            else _normalize_value_dtype(dtype)
        )
        value_np_dtype = _numpy_value_dtype(value_dtype)
        shape = normalize_shape(csc.shape)

        return CSCArray(
            data=to_mx(np.asarray(csc.data, dtype=value_np_dtype), dtype=value_dtype),
            indices=to_mx(
                np.asarray(csc.indices, dtype=index_np_dtype), dtype=index_dtype
            ),
            indptr=to_mx(
                np.asarray(csc.indptr, dtype=index_np_dtype), dtype=index_dtype
            ),
            shape=shape,
            sorted_indices=bool(canonical),
            has_canonical_format=bool(canonical),
        )

    if canonical or out_format == "csr":
        csr = matrix.tocsr(copy=True)
        if canonical:
            csr.sum_duplicates()
            csr.sort_indices()
    else:
        csr = matrix.tocsr(copy=False)

    value_dtype = (
        _infer_value_dtype_from_numpy(np.asarray(csr.data))
        if dtype is None
        else _normalize_value_dtype(dtype)
    )
    value_np_dtype = _numpy_value_dtype(value_dtype)
    shape = normalize_shape(csr.shape)

    if out_format == "csr":
        return CSRArray(
            data=to_mx(np.asarray(csr.data, dtype=value_np_dtype), dtype=value_dtype),
            indices=to_mx(
                np.asarray(csr.indices, dtype=index_np_dtype), dtype=index_dtype
            ),
            indptr=to_mx(
                np.asarray(csr.indptr, dtype=index_np_dtype), dtype=index_dtype
            ),
            shape=shape,
            sorted_indices=bool(canonical),
            has_canonical_format=bool(canonical),
        )

    coo = csr.tocoo(copy=False) if canonical else matrix.tocoo(copy=True)
    from mlx_sparse._coo import COOArray

    return COOArray(
        data=to_mx(np.asarray(coo.data, dtype=value_np_dtype), dtype=value_dtype),
        row=to_mx(np.asarray(coo.row, dtype=index_np_dtype), dtype=index_dtype),
        col=to_mx(np.asarray(coo.col, dtype=index_np_dtype), dtype=index_dtype),
        shape=shape,
        has_canonical_format=bool(canonical),
    )


def asarray(
    x,
    *,
    threshold: float = 0.0,
    dtype=None,
    index_dtype=mx.int32,
) -> CSRArray | CSCArray:
    """Convert common sparse or dense inputs to a sparse array.

    Existing :class:`~mlx_sparse.CSRArray` and :class:`~mlx_sparse.CSCArray`
    instances are returned unchanged
    unless ``dtype`` requests a value cast. :class:`~mlx_sparse.COOArray`
    instances are converted with ``tocsr(canonical=True)``. SciPy sparse
    matrices/arrays route through :func:`from_scipy`, dense MLX, NumPy, and
    Python array-likes route through :func:`fromdense`.

    Args:
        x: Existing mlx-sparse array, SciPy sparse array, dense MLX array,
            NumPy array, or Python rank-2 array-like.
        threshold: Dense-only structural-zero threshold.
        dtype: Optional target value dtype.
        index_dtype: Target index dtype for newly constructed sparse arrays.

    Returns:
        Existing ``CSRArray`` or ``CSCArray`` inputs are preserved. Other
        inputs return a canonical ``CSRArray``.
    """
    from mlx_sparse._coo import COOArray

    dtype = None if dtype is None else _normalize_value_dtype(dtype)
    if isinstance(x, CSCArray):
        if dtype is None or x.data.dtype == dtype:
            return x
        return CSCArray(
            data=x.data.astype(dtype),
            indices=x.indices,
            indptr=x.indptr,
            shape=x.shape,
            sorted_indices=x.sorted_indices,
            has_canonical_format=x.has_canonical_format,
        )
    if isinstance(x, CSRArray):
        if dtype is None or x.data.dtype == dtype:
            return x
        return CSRArray(
            data=x.data.astype(dtype),
            indices=x.indices,
            indptr=x.indptr,
            shape=x.shape,
            sorted_indices=x.sorted_indices,
            has_canonical_format=x.has_canonical_format,
        )
    if isinstance(x, COOArray):
        csr = x.tocsr(canonical=True)
        if dtype is None or csr.data.dtype == dtype:
            return csr
        return CSRArray(
            data=csr.data.astype(dtype),
            indices=csr.indices,
            indptr=csr.indptr,
            shape=csr.shape,
            sorted_indices=csr.sorted_indices,
            has_canonical_format=csr.has_canonical_format,
        )

    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None
    if sp is not None and sp.issparse(x):
        return from_scipy(
            x,
            format="csr",
            dtype=dtype,
            index_dtype=index_dtype,
            canonical=True,
        )

    return fromdense(
        x,
        threshold=threshold,
        dtype=dtype,
        index_dtype=index_dtype,
    )


def block_array(blocks, *, format=None, dtype=None):
    """Build a sparse array from a rectangular grid of sparse or dense blocks.

    ``block_array`` mirrors the SciPy structural constructor while returning
    mlx-sparse arrays. Each non-``None`` entry must be a COO, CSR, CSC, or
    dense rank-2 array. ``None`` entries represent all-zero blocks whose shape
    is inferred from the other blocks in the same block row and block column;
    all-``None`` rows or columns are assigned size zero.

    The block grid is validated before assembly: every block row must have a
    consistent height, every block column must have a consistent width, the
    grid must be rectangular, and only ``"coo"``, ``"csr"``, and ``"csc"``
    output formats are supported. Dense blocks are converted with the native
    :func:`fromdense` path. Sparse blocks are converted to COO through native
    format conversion, then one native coordinate-offset primitive copies
    values and offsets coordinates. No stored entries are iterated in Python.

    ``format=None`` defaults to COO because block assembly is a construction
    operation. CSR and CSC requests canonicalize through the existing native
    compressed conversion path, summing duplicate coordinates.

    Args:
        blocks: Rectangular 2-D grid of sparse arrays, dense rank-2 arrays, or
            ``None`` entries.
        format: Output format, one of ``None``, ``"coo"``, ``"csr"``, or
            ``"csc"``.
        dtype: Optional output value dtype. When omitted, value dtypes are
            promoted across non-``None`` blocks under mlx-sparse's sparse value
            dtype policy.

    Returns:
        A :class:`~mlx_sparse.COOArray`, :class:`~mlx_sparse.CSRArray`, or
        :class:`~mlx_sparse.CSCArray`.
    """
    raw_blocks, row_offsets, col_offsets, shape = _validate_block_grid(blocks)
    return _assemble_offset_blocks(
        raw_blocks,
        row_offsets,
        col_offsets,
        shape,
        function_name="block_array",
        format=format,
        dtype=dtype,
    )


def bmat(blocks, format=None, dtype=None):
    """Compatibility alias for :func:`block_array`.

    Unlike SciPy's historical matrix-returning ``bmat`` behavior, mlx-sparse
    always returns sparse array containers (COO, CSR, or CSC). See
    :func:`block_array` for validation, dtype promotion, and native assembly
    details.
    """
    return block_array(blocks, format=format, dtype=dtype)


def block_diag(mats, format=None, dtype=None):
    """Build a block diagonal sparse array from a sequence of matrices.

    Each input must be a sparse COO/CSR/CSC array or a dense rank-2 array.
    Blocks are placed on the main block diagonal using native coordinate-offset
    assembly; off-diagonal zero regions are implicit and never materialized.
    ``None`` is not accepted because it has no shape in this constructor.

    ``format=None`` defaults to COO, matching SciPy's construction-oriented
    default. CSR and CSC requests canonicalize through native conversion.

    Args:
        mats: Non-empty sequence of sparse or dense rank-2 matrices.
        format: Output format, one of ``None``, ``"coo"``, ``"csr"``, or
            ``"csc"``.
        dtype: Optional output value dtype.

    Returns:
        A sparse block diagonal array.
    """
    try:
        mats_list = list(mats)
    except TypeError as exc:
        raise TypeError("block_diag mats must be an iterable of matrices.") from exc
    if not mats_list:
        raise ValueError("block_diag requires at least one matrix.")
    if any(mat is None for mat in mats_list):
        raise TypeError("block_diag does not accept None blocks; use block_array.")

    raw_blocks = [_as_raw_block("block_diag", mat) for mat in mats_list]
    row_offsets: list[int] = []
    col_offsets: list[int] = []
    row_cursor = 0
    col_cursor = 0
    for block in raw_blocks:
        row_offsets.append(row_cursor)
        col_offsets.append(col_cursor)
        row_cursor += block.shape[0]
        col_cursor += block.shape[1]

    return _assemble_offset_blocks(
        raw_blocks,
        row_offsets,
        col_offsets,
        (row_cursor, col_cursor),
        function_name="block_diag",
        format=format,
        dtype=dtype,
    )


def vstack(blocks, format=None, dtype=None):
    """Stack sparse or dense rank-2 blocks vertically.

    All blocks must have the same number of columns. The implementation uses
    the same native coordinate-offset assembly as :func:`block_array` with a
    single block column. ``None`` entries are rejected because their row height
    cannot be inferred in a one-dimensional stack.

    Args:
        blocks: Non-empty sequence of sparse or dense rank-2 matrices.
        format: Output format, one of ``None``, ``"coo"``, ``"csr"``, or
            ``"csc"``. ``None`` defaults to COO.
        dtype: Optional output value dtype.

    Returns:
        A sparse array containing the vertical stack.
    """
    try:
        blocks_list = list(blocks)
    except TypeError as exc:
        raise TypeError("vstack blocks must be an iterable of matrices.") from exc
    if not blocks_list:
        raise ValueError("vstack requires at least one block.")
    if any(block is None for block in blocks_list):
        raise TypeError("vstack does not accept None blocks; use block_array.")

    raw_blocks = [_as_raw_block("vstack", block) for block in blocks_list]
    n_cols = raw_blocks[0].shape[1]
    row_offsets: list[int] = []
    row_cursor = 0
    for i, block in enumerate(raw_blocks):
        if block.shape[1] != n_cols:
            raise ValueError(
                f"vstack block {i} has {block.shape[1]} columns, expected {n_cols}."
            )
        row_offsets.append(row_cursor)
        row_cursor += block.shape[0]

    return _assemble_offset_blocks(
        raw_blocks,
        row_offsets,
        [0] * len(raw_blocks),
        (row_cursor, n_cols),
        function_name="vstack",
        format=format,
        dtype=dtype,
    )


def hstack(blocks, format=None, dtype=None):
    """Stack sparse or dense rank-2 blocks horizontally.

    All blocks must have the same number of rows. The implementation uses
    native coordinate-offset assembly with a single block row. ``None`` entries
    are rejected because their column width cannot be inferred in a
    one-dimensional stack.

    Args:
        blocks: Non-empty sequence of sparse or dense rank-2 matrices.
        format: Output format, one of ``None``, ``"coo"``, ``"csr"``, or
            ``"csc"``. ``None`` defaults to COO.
        dtype: Optional output value dtype.

    Returns:
        A sparse array containing the horizontal stack.
    """
    try:
        blocks_list = list(blocks)
    except TypeError as exc:
        raise TypeError("hstack blocks must be an iterable of matrices.") from exc
    if not blocks_list:
        raise ValueError("hstack requires at least one block.")
    if any(block is None for block in blocks_list):
        raise TypeError("hstack does not accept None blocks; use block_array.")

    raw_blocks = [_as_raw_block("hstack", block) for block in blocks_list]
    n_rows = raw_blocks[0].shape[0]
    col_offsets: list[int] = []
    col_cursor = 0
    for i, block in enumerate(raw_blocks):
        if block.shape[0] != n_rows:
            raise ValueError(
                f"hstack block {i} has {block.shape[0]} rows, expected {n_rows}."
            )
        col_offsets.append(col_cursor)
        col_cursor += block.shape[1]

    return _assemble_offset_blocks(
        raw_blocks,
        [0] * len(raw_blocks),
        col_offsets,
        (n_rows, col_cursor),
        function_name="hstack",
        format=format,
        dtype=dtype,
    )


def _as_sparse_for_triangular(name: str, value):
    if _is_sparse_array(value):
        return value
    dense = ensure_mx_array(value)
    if dense.ndim != 2:
        raise ValueError(f"{name} expects a rank-2 array, got shape={dense.shape}.")
    return fromdense(dense, dtype=_infer_dense_constructor_dtype(dense))


def _triangular_to_format(array, format: str):
    if format == "coo":
        if isinstance(array, COOArray):
            return array
        return array.tocoo(canonical=None)
    if format == "csr":
        if isinstance(array, CSRArray):
            return array
        if isinstance(array, COOArray):
            return array.tocsr(canonical=True)
        return array.tocsr(canonical=True)
    if format == "csc":
        if isinstance(array, CSCArray):
            return array
        if isinstance(array, COOArray):
            return array.tocsc(canonical=True)
        return array.tocsc(canonical=True)
    raise ValueError(f"unsupported sparse format {format!r}.")


def _triangular(A, *, k=0, format=None, upper: bool):
    name = "triu" if upper else "tril"
    out_format = _normalize_sparse_format(name, format, default="coo")
    diagonal = operator_index(k)
    array = _as_sparse_for_triangular(name, A)

    if isinstance(array, COOArray):
        data, row, col = _native.coo_triangular(
            array,
            k=diagonal,
            upper=upper,
        )
        out = COOArray(
            data=data,
            row=row,
            col=col,
            shape=array.shape,
            has_canonical_format=array.has_canonical_format,
        )
    elif isinstance(array, CSRArray):
        data, indices, indptr = _native.csr_triangular(
            array,
            k=diagonal,
            upper=upper,
        )
        out = CSRArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=array.shape,
            sorted_indices=array.sorted_indices,
            has_canonical_format=array.has_canonical_format,
        )
    elif isinstance(array, CSCArray):
        data, indices, indptr = _native.csc_triangular(
            array,
            k=diagonal,
            upper=upper,
        )
        out = CSCArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=array.shape,
            sorted_indices=array.sorted_indices,
            has_canonical_format=array.has_canonical_format,
        )
    else:
        raise TypeError(f"{name} expects a sparse or dense rank-2 array.")

    return _triangular_to_format(out, out_format)


def tril(A, k=0, format=None):
    """Return the lower triangular portion of a sparse or dense matrix.

    Elements with ``column - row <= k`` are retained. ``k=0`` keeps the main
    diagonal, positive ``k`` includes superdiagonals, and negative ``k`` moves
    the cutoff below the main diagonal. COO, CSR, CSC, and dense rank-2 inputs
    are accepted. Dense inputs are first extracted with native
    :func:`fromdense`; sparse inputs are filtered with native staged count/fill
    kernels for their storage format.

    ``format=None`` defaults to COO, matching SciPy's triangular extraction
    default. CSR and CSC requests are returned in the requested sparse format.
    Triangular extraction compacts sparse structure based on coordinates and is
    intentionally forward-only under MLX autodiff in v0.0.6b0.
    """
    return _triangular(A, k=k, format=format, upper=False)


def triu(A, k=0, format=None):
    """Return the upper triangular portion of a sparse or dense matrix.

    Elements with ``column - row >= k`` are retained. See :func:`tril` for
    input handling, native compaction details, format semantics, and autodiff
    limitations.
    """
    return _triangular(A, k=k, format=format, upper=True)


def from_numpy(
    array,
    *,
    threshold: float = 0.0,
    dtype=None,
    index_dtype=mx.int32,
) -> CSRArray:
    """Convert a rank-2 NumPy array to a canonical CSRArray."""
    return fromdense(
        array,
        threshold=threshold,
        dtype=dtype,
        index_dtype=index_dtype,
    )
