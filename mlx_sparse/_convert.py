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

"""COO to compressed conversions expressed with MLX array operations.

The Metal kernels for these conversions rank every entry against every other
entry, which is quadratic in ``nnz``. MLX already ships a device sort, a device
scatter and a device scan, so the same conversion can be written as a sort of
the row-major keys followed by two gathers and a histogram. That is
``O(nnz log nnz)``, runs entirely on the GPU, and needs no host round trip.

The CPU backend keeps its own counting sort, which is faster than a general
sort at every size measured, so these routines are used for the GPU stream
only.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_sparse._typing import Shape2D

# The linear key row * n_cols + col has to stay exact in int64. The largest
# value is reserved as the key given to entries outside the declared shape, so
# that a stable sort carries them past every entry that stays.
_MAX_KEY = 2**63 - 1
_OUT_OF_RANGE_KEY = _MAX_KEY

# Value dtypes the Metal scatter does not accept, so mx.take cannot build its
# vjp for them.
_SCATTERLESS_VALUE_DTYPES = (mx.complex64,)


def can_use_array_ops(data: mx.array, shape: Shape2D) -> bool:
    """Return whether the array-op conversions apply to this call.

    They are used on the GPU, where the alternative is quadratic, for shapes
    whose largest linear index is exactly representable in ``int64``, and for
    value dtypes the Metal scatter supports. The scatter appears in the
    gather's own gradient, so a dtype it rejects would convert but fail to
    differentiate; those calls keep the kernel path.
    """
    if mx.default_device().type != mx.DeviceType.gpu:
        return False
    if data.dtype in _SCATTERLESS_VALUE_DTYPES:
        return False
    n_rows, n_cols = shape
    if n_rows <= 0 or n_cols <= 0:
        return False
    # Strictly below, because the top key belongs to the out-of-range entries.
    return (n_rows - 1) * n_cols + (n_cols - 1) < _MAX_KEY


def entries_in_range(row: mx.array, col: mx.array, shape: Shape2D) -> mx.array:
    """Return the mask of entries that address a position inside ``shape``.

    Coordinates are ordinary buffer contents and nothing on the device path
    checks them, so this is computed and applied rather than assumed. It is a
    handful of elementwise comparisons and stays on the device, which the
    documented ``validate="full"`` check cannot: that one reads the coordinates
    back to the host, so it costs a synchronization and cannot run inside
    ``mx.compile``.
    """
    n_rows, n_cols = shape
    return (row >= 0) & (row < n_rows) & (col >= 0) & (col < n_cols)


def _segment_pointers(
    segment_of_entry: mx.array, n_segments: int, index_dtype, keep: mx.array
):
    """Build a compressed pointer array by counting entries per segment.

    ``segment_of_entry`` holds the row of every COO entry for CSR, or the
    column for CSC. Counting scatters into an ``int32`` accumulator because the
    Metal scatter does not take ``int64`` updates; a per-segment count cannot
    exceed ``nnz``, so the width is only a constraint on the running total,
    which is accumulated in the index dtype instead.

    Entries outside the declared shape count zero, at segment zero: the scatter
    needs an index it can address either way, and adding zero there leaves the
    counts alone.
    """
    zero = mx.zeros((1,), dtype=index_dtype)
    if n_segments == 0 or segment_of_entry.size == 0:
        return mx.zeros((n_segments + 1,), dtype=index_dtype)
    counts = mx.zeros((n_segments,), dtype=mx.int32)
    counts = counts.at[mx.where(keep, segment_of_entry, 0)].add(
        mx.where(keep, 1, 0).astype(mx.int32)
    )
    return mx.concatenate([zero, mx.cumsum(counts.astype(index_dtype))])


def coo_sort_permutation(
    major: mx.array, minor: mx.array, n_minor: int, keep: mx.array
) -> mx.array:
    """Return the permutation that orders COO entries by ``(major, minor)``.

    ``mx.argsort`` is stable, so entries that share a coordinate keep their
    input order, which is the tie-break the CPU conversion uses as well.

    Entries outside the declared shape take the largest key, so they sort past
    every entry that stays, in input order. That leaves them occupying the slots
    the pointer array does not reach, which is where the conversion wants them:
    every slot is written, and none of them is referenced.
    """
    safe_major = mx.where(keep, major, 0).astype(mx.int64)
    safe_minor = mx.where(keep, minor, 0).astype(mx.int64)
    key = mx.where(keep, safe_major * n_minor + safe_minor, _OUT_OF_RANGE_KEY)
    return mx.argsort(key)


def _compressed(
    data: mx.array,
    major: mx.array,
    minor: mx.array,
    n_major: int,
    n_minor: int,
    keep: mx.array,
    order: mx.array | None,
):
    if order is None:
        order = coo_sort_permutation(major, minor, n_minor, keep)
    # Zeroing before the gather is what fills the unreferenced tail: the entries
    # that were dropped are the ones the sort put there.
    #
    # The values are masked by multiplication rather than by mx.where, which
    # does not carry a tangent through to its value argument: differentiating
    # `where(mask, data, 0)` with respect to data yields the boolean mask.
    # Multiplying gives the same result and the derivative the conversion wants,
    # which is zero for an entry the conversion drops.
    kept_data = data * keep.astype(data.dtype)
    # The coordinates carry no tangent, so mx.where is safe on them.
    kept_minor = mx.where(keep, minor, mx.zeros((), dtype=minor.dtype))
    return (
        mx.take(kept_data, order),
        mx.take(kept_minor, order),
        _segment_pointers(major, n_major, minor.dtype, keep),
    )


def coo_to_csr(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
    order: mx.array | None = None,
    keep: mx.array | None = None,
):
    """Convert COO buffers to CSR buffers using MLX array operations."""
    if keep is None:
        keep = entries_in_range(row, col, shape)
    return _compressed(data, row, col, shape[0], shape[1], keep, order)


def coo_to_csc(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
    order: mx.array | None = None,
    keep: mx.array | None = None,
):
    """Convert COO buffers to CSC buffers using MLX array operations."""
    if keep is None:
        keep = entries_in_range(row, col, shape)
    return _compressed(data, col, row, shape[1], shape[0], keep, order)
