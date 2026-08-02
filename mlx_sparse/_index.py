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

"""Row and column selection on compressed rows, expressed with MLX operations.

Selecting rows out of a CSR matrix is a gather, not a rebuild: the entries of a
row are contiguous, so taking whole rows moves runs of ``data`` and ``indices``
without touching a value or reordering anything within a row. That is the reason
the declared flags survive the operation -- a selection of sorted rows is still
sorted, and a selection of canonical rows is still canonical -- and it is why
this is written as array operations rather than as a kernel.

Both routines here read one count back to the host, because the number of stored
entries in the result depends on which rows were asked for and how full they
are. That is the same reason the dynamic-output helpers synchronize, and it is
what makes the bounds check free: the operation already cannot avoid a host
round trip, so refusing an out-of-range index costs nothing on top of it.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_sparse._typing import Shape2D


def row_ids_from_indptr(indptr: mx.array, nnz: int) -> mx.array:
    """Expand a pointer array to the row that owns each stored entry.

    ``indptr=[0, 2, 2, 3]`` describes rows of length 2, 0 and 1, so the answer
    is ``[0, 0, 2]``. Marking each row boundary with a one and taking a running
    sum gives it without a host round trip. Boundaries that sit at ``nnz``
    belong to trailing empty rows and mark nothing, hence the zero weight.
    """
    if nnz == 0:
        return mx.zeros((0,), dtype=mx.int32)
    boundaries = indptr[1:-1]
    weight = (boundaries < nnz).astype(mx.int32)
    starts = mx.zeros((nnz,), dtype=mx.int32)
    starts = starts.at[mx.minimum(boundaries, nnz - 1)].add(weight)
    return mx.cumsum(starts)


def normalize_row_selection(key, n_rows: int) -> np.ndarray:
    """Turn a row key into a validated array of row ids.

    Accepts a slice, a sequence, a NumPy or MLX integer array, or a boolean
    mask over the rows. Negative ids count from the end, as they do everywhere
    else in Python; anything still outside the shape after that is a caller
    error and raises, which is what SciPy and NumPy both do.
    """
    if isinstance(key, slice):
        return np.arange(*key.indices(n_rows), dtype=np.int64)

    if isinstance(key, mx.array):
        selection = np.asarray(key)
    else:
        selection = np.asarray(key)

    if selection.dtype == np.bool_:
        if selection.ndim != 1 or selection.shape[0] != n_rows:
            raise IndexError(
                "CSRArray boolean row mask must have one entry per row, got "
                f"shape {selection.shape} for n_rows={n_rows}."
            )
        return np.nonzero(selection)[0].astype(np.int64)

    if selection.ndim == 0:
        raise IndexError(
            "CSRArray does not support indexing with a bare integer, because "
            "the result would be a one-dimensional sparse array and this "
            "package has no such container. Use A[[i]] for the 1 x n_cols "
            "matrix of row i."
        )
    if selection.ndim != 1:
        raise IndexError(
            f"CSRArray row selection must be one-dimensional, got shape "
            f"{selection.shape}."
        )
    if selection.size == 0:
        # An empty selection carries no values to have a dtype about, and
        # ``np.asarray([])`` is float64, so judging it by dtype would refuse
        # the ordinary ``A[[]]``.
        return np.empty((0,), dtype=np.int64)
    if not np.issubdtype(selection.dtype, np.integer):
        raise IndexError(
            f"CSRArray row selection must be integers or a boolean mask, got "
            f"dtype {selection.dtype}."
        )

    selection = selection.astype(np.int64, copy=True)
    wrapped = np.where(selection < 0, selection + n_rows, selection)
    if wrapped.size and (wrapped.min() < 0 or wrapped.max() >= n_rows):
        offender = selection[(wrapped < 0) | (wrapped >= n_rows)][0]
        raise IndexError(
            f"CSRArray row index {offender} is out of range for n_rows=" f"{n_rows}."
        )
    return wrapped


def select_rows(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rows: np.ndarray,
    shape: Shape2D,
):
    """Gather whole rows, in the order asked for, repeats included.

    ``out_indptr`` is the running sum of the chosen rows' lengths. Expanding it
    back to one entry per output row gives, for each entry, which output row it
    belongs to; subtracting that row's start from the entry's own position gives
    its offset within the row; and adding the offset to the row's start in the
    source is the position to read. One gather of ``data`` and ``indices`` then
    finishes it.
    """
    index_dtype = indptr.dtype
    chosen = mx.array(rows.astype(np.int32 if index_dtype == mx.int32 else np.int64))
    row_lengths = (indptr[1:] - indptr[:-1])[chosen]
    out_indptr = mx.concatenate(
        [mx.zeros((1,), dtype=index_dtype), mx.cumsum(row_lengths)]
    )
    # The size of the result depends on how full the chosen rows are, so it has
    # to be known here rather than deferred.
    out_nnz = int(out_indptr[-1])
    out_shape = (int(rows.size), shape[1])
    if out_nnz == 0:
        return (
            mx.zeros((0,), dtype=data.dtype),
            mx.zeros((0,), dtype=indices.dtype),
            mx.zeros((len(rows) + 1,), dtype=index_dtype),
            out_shape,
        )

    out_rows = row_ids_from_indptr(out_indptr, out_nnz)
    offsets = mx.arange(out_nnz, dtype=index_dtype) - out_indptr[out_rows]
    source = indptr[chosen][out_rows] + offsets
    return (
        mx.take(data, source),
        mx.take(indices, source),
        out_indptr,
        out_shape,
    )


def select_column_range(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    column_slice: slice,
    shape: Shape2D,
):
    """Keep the entries whose column falls in the slice, and renumber them.

    Nothing moves between rows, so the new pointer array is the old one read
    through a running count of what was kept. The surviving entries are brought
    to the front by a stable sort of the discard flag, which keeps them in their
    original order -- so a matrix with sorted indices stays sorted -- and works
    for every value dtype, including the ones MLX will not scatter.
    """
    index_dtype = indptr.dtype
    start, stop, step = column_slice.indices(shape[1])
    if step < 0:
        # A reversed slice renumbers the columns backwards, so the entries of a
        # row would come out in descending order and the sortedness this
        # operation otherwise preserves would have to be rebuilt per row. That
        # is a different operation than a gather, so it is refused rather than
        # answered approximately.
        raise IndexError(
            "CSRArray column slicing does not support a negative step, because "
            "it would reverse the column order within every row. Slice with a "
            "positive step and reverse the result yourself if that is wanted."
        )
    n_kept_columns = len(range(start, stop, step))

    offset = indices - start
    keep = (indices >= start) & (indices < stop)
    if step != 1:
        keep = keep & (offset % step == 0)
    renumbered = offset // step if step != 1 else offset

    kept_running = mx.concatenate(
        [mx.zeros((1,), dtype=index_dtype), mx.cumsum(keep.astype(index_dtype))]
    )
    out_indptr = kept_running[indptr]
    out_nnz = int(out_indptr[-1])
    out_shape = (shape[0], n_kept_columns)
    if out_nnz == 0:
        return (
            mx.zeros((0,), dtype=data.dtype),
            mx.zeros((0,), dtype=indices.dtype),
            mx.zeros((shape[0] + 1,), dtype=index_dtype),
            out_shape,
        )

    order = mx.argsort(mx.logical_not(keep).astype(mx.int32))
    source = order[:out_nnz]
    return (
        mx.take(data, source),
        mx.take(renumbered, source).astype(indices.dtype),
        out_indptr,
        out_shape,
    )
