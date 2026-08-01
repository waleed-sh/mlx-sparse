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

# The linear key row * n_cols + col has to stay exact in int64.
_MAX_KEY = 2**63 - 1

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
    return (n_rows - 1) * n_cols + (n_cols - 1) <= _MAX_KEY


def _segment_pointers(segment_of_entry: mx.array, n_segments: int, index_dtype):
    """Build a compressed pointer array by counting entries per segment.

    ``segment_of_entry`` holds the row of every COO entry for CSR, or the
    column for CSC. Counting scatters into an ``int32`` accumulator because the
    Metal scatter does not take ``int64`` updates; a per-segment count cannot
    exceed ``nnz``, so the width is only a constraint on the running total,
    which is accumulated in the index dtype instead.
    """
    zero = mx.zeros((1,), dtype=index_dtype)
    if n_segments == 0 or segment_of_entry.size == 0:
        return mx.zeros((n_segments + 1,), dtype=index_dtype)
    counts = mx.zeros((n_segments,), dtype=mx.int32)
    counts = counts.at[segment_of_entry].add(
        mx.ones(segment_of_entry.shape, dtype=mx.int32)
    )
    return mx.concatenate([zero, mx.cumsum(counts.astype(index_dtype))])


def coo_sort_permutation(major: mx.array, minor: mx.array, n_minor: int) -> mx.array:
    """Return the permutation that orders COO entries by ``(major, minor)``.

    ``mx.argsort`` is stable, so entries that share a coordinate keep their
    input order, which is the tie-break the CPU conversion uses as well.
    """
    key = major.astype(mx.int64) * n_minor + minor.astype(mx.int64)
    return mx.argsort(key)


def coo_to_csr(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
    order: mx.array | None = None,
):
    """Convert COO buffers to CSR buffers using MLX array operations."""
    if order is None:
        order = coo_sort_permutation(row, col, shape[1])
    return (
        mx.take(data, order),
        mx.take(col, order),
        _segment_pointers(row, shape[0], col.dtype),
    )


def coo_to_csc(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
    order: mx.array | None = None,
):
    """Convert COO buffers to CSC buffers using MLX array operations."""
    if order is None:
        order = coo_sort_permutation(col, row, shape[0])
    return (
        mx.take(data, order),
        mx.take(row, order),
        _segment_pointers(col, shape[1], row.dtype),
    )
