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

from dataclasses import dataclass
from numbers import Number

import mlx.core as mx

import mlx_sparse._native as _native
from mlx_sparse._csr import CSRArray
from mlx_sparse._typing import Shape2D, ValidationMode
from mlx_sparse._validation import (
    ensure_mx_array,
    normalize_shape,
    normalize_validation_mode,
    validate_coo_metadata,
    validate_coo_values,
)


@dataclass(frozen=True, slots=True)
class COOArray:
    """A 2D sparse matrix in Coordinate (COO) format.

    COOArray stores a sparse matrix as three parallel arrays:

    - **data**: non-zero values, shape ``(nnz,)``.
    - **row**: row coordinate for each value, shape ``(nnz,)``.
    - **col**: column coordinate for each value, shape ``(nnz,)``.

    COO is the primary *construction* format. It allows duplicate
    ``(row, col)`` entries and does not require sorted coordinates, making it
    straightforward to assemble matrices from element lists, graph adjacency
    lists, finite-element stencils, or Hamiltonians.

    COO supports native sparse-dense products directly. For heavily repeated
    row-oriented workloads, CSR may still be preferable after construction
    because its compressed row layout avoids coordinate scatter.

    **Format invariants** (checked by ``validate="metadata"`` by default):

    - All three arrays must be rank-1 with the same length.
    - ``row`` and ``col`` share the same integer dtype (``int32`` or
      ``int64``).
    - ``data`` dtype is one of ``float32``, ``float16``, ``bfloat16``, or
      ``complex64``.

    **Additional value-level invariants** (``validate="full"`` only):

    - ``0 <= row[i] < n_rows`` for all entries.
    - ``0 <= col[i] < n_cols`` for all entries.

    Args:
        data: Non-zero values, shape ``(nnz,)``.
        row: Row coordinates, shape ``(nnz,)``.
        col: Column coordinates, shape ``(nnz,)``.
        shape: Matrix dimensions as ``(n_rows, n_cols)``.
        has_canonical_format: Hint that coordinates are sorted and duplicate-
            free. Defaults to ``False``.

    Example::

        import mlx.core as mx
        import mlx_sparse as ms

        data = mx.array([2.0, -1.0, 4.0], dtype=mx.float32)
        row = mx.array([0, 0, 1], dtype=mx.int32)
        col = mx.array([0, 2, 1], dtype=mx.int32)

        # 2×3 matrix:  [[2, 0, -1],
        #               [0, 4,  0]]
        coo = ms.coo_array((data, (row, col)), shape=(2, 3))
        csr = coo.tocsr(canonical=True)
    """

    data: mx.array
    row: mx.array
    col: mx.array
    shape: Shape2D
    has_canonical_format: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", normalize_shape(self.shape))

    @property
    def nnz(self) -> int:
        """Number of stored values (including any duplicates)."""
        return int(self.data.shape[0])

    @property
    def dtype(self):
        """Value dtype of the stored non-zeros (e.g. ``mlx.core.float32``)."""
        return self.data.dtype

    @property
    def index_dtype(self):
        """Integer dtype used for ``row`` and ``col``."""
        return self.row.dtype

    @property
    def ndim(self) -> int:
        """Always 2. Sparse arrays in this package are rank-2."""
        return 2

    def __repr__(self) -> str:
        return (
            "COOArray("
            f"shape={self.shape}, nnz={self.nnz}, dtype={self.dtype}, "
            f"index_dtype={self.index_dtype}, "
            f"has_canonical_format={self.has_canonical_format})"
        )

    def tocsr(self, *, canonical: bool = False) -> CSRArray:
        """Convert to :class:`CSRArray`.

        Sorts entries by row then column and builds a ``(n_rows + 1,)`` row
        pointer array. Duplicate ``(row, col)`` entries are preserved in the
        raw output. Pass ``canonical=True`` to sum them.

        Args:
            canonical: If ``True``, call :meth:`~CSRArray.canonicalize` on the
                result to sort indices and sum duplicates. Default ``False``.

        Returns:
            A :class:`CSRArray` with ``sorted_indices=True``. If
            ``canonical=True``, also ``has_canonical_format=True``.
        """
        data, indices, indptr = _native.coo_tocsr(
            self.data, self.row, self.col, self.shape
        )
        csr = CSRArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=self.shape,
            sorted_indices=True,
            has_canonical_format=False,
        )
        if canonical:
            return csr.canonicalize()
        return csr

    def tocsc(self, *, canonical: bool = False):
        """Convert to :class:`~mlx_sparse.CSCArray`."""
        from mlx_sparse._csc import CSCArray

        data, indices, indptr = _native.coo_tocsc(
            self.data, self.row, self.col, self.shape
        )
        csc = CSCArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=self.shape,
            sorted_indices=True,
            has_canonical_format=False,
        )
        if canonical:
            return csc.canonicalize()
        return csc

    def todense(self) -> mx.array:
        """Materialize as a dense MLX array.

        Internally converts to CSR and then calls
        :meth:`~CSRArray.todense`. Duplicate entries are summed.

        Returns:
            Dense array of shape ``(n_rows, n_cols)`` with the same dtype as
            ``self.data``.
        """
        return self.tocsr(canonical=False).todense()

    def row_sums(self) -> mx.array:
        """Return the sum of stored values in each COO row."""
        from mlx_sparse._ops import coo_row_sums

        return coo_row_sums(self)

    def col_sums(self) -> mx.array:
        """Return the sum of stored values in each COO column."""
        from mlx_sparse._ops import coo_col_sums

        return coo_col_sums(self)

    def column_sums(self) -> mx.array:
        """Alias for :meth:`col_sums`."""
        return self.col_sums()

    def row_norms(self) -> mx.array:
        """Return the dense-semantics L2 norm of each COO row as ``float32``."""
        from mlx_sparse._ops import coo_row_norms

        return coo_row_norms(self)

    def col_norms(self) -> mx.array:
        """Return the dense-semantics L2 norm of each COO column as ``float32``."""
        from mlx_sparse._ops import coo_col_norms

        return coo_col_norms(self)

    def column_norms(self) -> mx.array:
        """Alias for :meth:`col_norms`."""
        return self.col_norms()

    def diagonal(self) -> mx.array:
        """Return the summed main diagonal."""
        from mlx_sparse._ops import coo_diagonal

        return coo_diagonal(self)

    def trace(self) -> mx.array:
        """Return the summed main diagonal as a scalar."""
        from mlx_sparse._ops import coo_trace

        return coo_trace(self)

    def sum(self, axis=None) -> mx.array:
        """Sum sparse values over all entries, rows, or columns.

        ``axis=None`` returns a scalar, ``axis=1`` returns row sums, and
        ``axis=0`` returns column sums.
        """
        if axis is None:
            return mx.sum(self.row_sums())
        if axis in (1, -1):
            return self.row_sums()
        if axis in (0, -2):
            return self.col_sums()
        raise ValueError(f"COOArray.sum axis must be None, 0, or 1; got {axis!r}.")

    def __matmul__(self, rhs):
        """Matrix multiplication via the ``@`` operator."""
        from mlx_sparse._csc import CSCArray
        from mlx_sparse._ops import coo_matmat, coo_matmul, coo_matvec

        if isinstance(rhs, COOArray):
            return coo_matmat(self, rhs)
        if isinstance(rhs, (CSRArray, CSCArray)):
            raise NotImplementedError(
                "Mixed-format COO sparse-sparse matmul is not implemented. "
                "Convert explicitly if another format is acceptable for your workload."
            )

        rhs = ensure_mx_array(rhs)
        if rhs.ndim == 1:
            return coo_matvec(self, rhs)
        if rhs.ndim >= 2:
            return coo_matmul(self, rhs)
        raise ValueError(f"COO matmul expects rank-1 or higher RHS, got {rhs.shape}.")

    def __rmul__(self, other):
        """Multiply the current CSCArray by a number using the ``*`` operator.

        This returns a new CSCArray with the data multiplied by the number, and
        therefore does not in-place mutate the current CSCArray.

        Args:
            other: A valid number (complex or not).

        Returns:
            A new CSCArray with the data multiplied by the number.

        Raises:
            TypeError: If ``other`` is not an actual number.
        """
        if not isinstance(other, Number):
            raise TypeError(f"Expected a number, got {type(other)!r}")

        return COOArray(
            data=other * self.data,
            row=self.row,
            col=self.col,
            shape=self.shape,
            has_canonical_format=self.has_canonical_format,
        )


def coo_array(
    arg,
    shape,
    *,
    validate: ValidationMode = "metadata",
    canonical: bool | None = None,
) -> COOArray:
    """Construct a :class:`COOArray` from coordinate arrays.

    Accepts either a ``(data, (row, col))`` pair or an existing
    ``COOArray``. All array inputs are converted to ``mlx.core.array`` if they
    are not already.

    Args:
        arg: A ``(data, (row, col))`` pair where

            - *data*: non-zero values, shape ``(nnz,)``, dtype
              ``float32 | float16 | bfloat16 | complex64``.
            - *row*: row coordinates, shape ``(nnz,)``, dtype
              ``int32 | int64``.
            - *col*: column coordinates, shape ``(nnz,)``, same integer
              dtype as *row*.

            Alternatively, an existing :class:`COOArray` (returned unchanged if
            ``shape`` matches).

        shape: Matrix dimensions as a length-2 sequence ``(n_rows, n_cols)``.
        validate: Validation level, one of:

            - ``"metadata"`` *(default)*: checks ranks, lengths, and dtypes.
            - ``"full"`` / ``True``: also verifies coordinate bounds. May
              synchronize to host.
            - ``False`` / ``"none"``: skips all checks.

        canonical: Set to ``True`` to assert the coordinates are sorted and
            duplicate-free. Default ``None`` (not asserted).

    Returns:
        A :class:`COOArray` with the given buffers and shape.

    Raises:
        TypeError: If ``arg`` cannot be unpacked as ``(data, (row, col))``, or
            if dtype constraints are violated.
        ValueError: If shape or length constraints are violated.

    Example::

        import mlx.core as mx
        import mlx_sparse as ms

        data = mx.array([1.0, 2.0, 3.0, 2.0], dtype=mx.float32)
        row = mx.array([0, 0, 1, 0], dtype=mx.int32)
        col = mx.array([0, 1, 0, 0], dtype=mx.int32)

        # Two entries at (0, 0). Summed when converting to CSR.
        coo = ms.coo_array((data, (row, col)), shape=(2, 2))
        csr = coo.tocsr(canonical=True)  # sums duplicates
    """
    mode = normalize_validation_mode(validate)
    shape = normalize_shape(shape)

    if isinstance(arg, COOArray):
        array = arg
        if array.shape != shape:
            raise ValueError(
                f"COOArray shape mismatch: got {array.shape}, expected {shape}."
            )
        return array

    try:
        data, coords = arg
        row, col = coords
    except Exception as exc:
        raise TypeError("coo_array expects (data, (row, col)) or a COOArray.") from exc

    data = ensure_mx_array(data)
    row = ensure_mx_array(row)
    col = ensure_mx_array(col)

    if mode != "none":
        validate_coo_metadata(data, row, col, shape)
    if mode == "full":
        validate_coo_values(row, col, shape)

    return COOArray(
        data=data,
        row=row,
        col=col,
        shape=shape,
        has_canonical_format=bool(canonical),
    )
