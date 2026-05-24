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
import numpy as np

import mlx_sparse._native as _native
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


def _infer_diagonal_dtype(diagonal_arrays: Sequence[np.ndarray]):
    if any(np.iscomplexobj(diag) for diag in diagonal_arrays):
        return mx.complex64
    if any(diag.dtype == np.float16 for diag in diagonal_arrays):
        return mx.float16
    return mx.float32


def _normalize_index_dtype(index_dtype):
    if index_dtype not in INDEX_DTYPES:
        raise TypeError(f"index_dtype must be mx.int32 or mx.int64, got {index_dtype}.")
    return index_dtype


def _csr_from_sorted_triplets(
    data: np.ndarray,
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
        data=to_mx(data, dtype=dtype),
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


def _as_diagonal_sequence(diagonals) -> list[np.ndarray]:
    if isinstance(diagonals, mx.array):
        if diagonals.ndim == 0:
            return [np.asarray([to_numpy(diagonals).item()])]
        if diagonals.ndim == 1:
            return [to_numpy(diagonals)]
        if diagonals.ndim == 2:
            return [row for row in to_numpy(diagonals)]
    if np.isscalar(diagonals):
        return [np.asarray([diagonals])]
    if isinstance(diagonals, Sequence):
        if not diagonals:
            return []
        first = diagonals[0]
        if np.isscalar(first) or isinstance(first, mx.array) and first.ndim == 0:
            return [
                np.asarray(
                    [
                        to_numpy(d).item() if isinstance(d, mx.array) else d
                        for d in diagonals
                    ]
                )
            ]
        return [
            to_numpy(d) if isinstance(d, mx.array) else np.asarray(d) for d in diagonals
        ]
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
    is always in canonical form.

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
            dim = max(dim, int(diag.size) + abs(int(offset)))
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
        if diag.size > capacity:
            raise ValueError(
                f"diagonal at offset {offset} has length {diag.size}, "
                f"but shape {shape_2d} can hold at most {capacity} values."
            )
        nnz = int(diag.size)
        if nnz == 0:
            continue
        positions = np.arange(nnz, dtype=index_np_dtype)
        row_parts.append(row_start + positions)
        col_parts.append(col_start + positions)
        data_parts.append(np.asarray(diag))

    if data_parts:
        row = np.concatenate(row_parts).astype(index_np_dtype, copy=False)
        col = np.concatenate(col_parts).astype(index_np_dtype, copy=False)
        data = np.concatenate(data_parts)
        order = np.lexsort((col, row))
        row = row[order]
        col = col[order]
        data = data[order]
    else:
        row = np.empty((0,), dtype=index_np_dtype)
        col = np.empty((0,), dtype=index_np_dtype)
        data = np.empty((0,), dtype=np.float32)

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
