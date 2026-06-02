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

import mlx.core as mx

import mlx_sparse._native as _native
from mlx_sparse._csr import CSRArray
from mlx_sparse._typing import Shape2D, ValidationMode
from mlx_sparse._validation import (
    ensure_mx_array,
    normalize_shape,
    normalize_validation_mode,
    sanitize_scalar,
    validate_csc_metadata,
    validate_csc_values,
)


@dataclass(frozen=True, slots=True)
class CSCArray:
    """A 2D sparse matrix in Compressed Sparse Column (CSC) format.

    CSCArray stores a 2D sparse matrix using three MLX arrays:

    - **data**: non-zero values, shape ``(nnz,)``.
    - **indices**: row index of each stored value, shape ``(nnz,)``.
    - **indptr**: column pointer array, shape ``(n_cols + 1,)``.

    Column ``j`` spans ``data[indptr[j] : indptr[j+1]]`` with corresponding
    row indices ``indices[indptr[j] : indptr[j+1]]``. Duplicate row entries
    within a column are permitted unless the matrix is in canonical form.

    CSC is the column-compressed dual of CSR. It is the natural layout for
    operations that consume one full column at a time, such as transpose
    matvec, column-oriented canonicalization, and future direct factorization
    kernels.

    Args:
        data: Non-zero values, shape ``(nnz,)``.
        indices: Row indices, shape ``(nnz,)``.
        indptr: Column pointer array, shape ``(n_cols + 1,)``.
        shape: Matrix dimensions as ``(n_rows, n_cols)``.
        sorted_indices: Hint that row indices within each column are sorted
            ascending. Defaults to ``False``.
        has_canonical_format: Hint that the matrix has sorted row indices and
            no duplicate row index in any column. Implies
            ``sorted_indices=True``. Defaults to ``False``.
    """

    data: mx.array
    indices: mx.array
    indptr: mx.array
    shape: Shape2D
    sorted_indices: bool = False
    has_canonical_format: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", normalize_shape(self.shape))

    @property
    def nnz(self) -> int:
        """Number of stored values (including any duplicates)."""
        return int(self.data.shape[0])

    @property
    def dtype(self):
        """Value dtype of the stored non-zeros."""
        return self.data.dtype

    @property
    def index_dtype(self):
        """Integer dtype used for ``indices`` and ``indptr``."""
        return self.indices.dtype

    @property
    def ndim(self) -> int:
        """Always 2. Sparse arrays in this package are rank-2."""
        return 2

    def __repr__(self) -> str:
        return (
            "CSCArray("
            f"shape={self.shape}, nnz={self.nnz}, dtype={self.dtype}, "
            f"index_dtype={self.index_dtype}, "
            f"sorted_indices={self.sorted_indices}, "
            f"has_canonical_format={self.has_canonical_format})"
        )

    def todense(self) -> mx.array:
        """Materialize the sparse matrix as a dense MLX array."""
        return _native.csc_todense(self.data, self.indices, self.indptr, self.shape)

    def row_sums(self) -> mx.array:
        """Return the sum of stored values in each CSC row."""
        from mlx_sparse._ops import csc_row_sums

        return csc_row_sums(self)

    def col_sums(self) -> mx.array:
        """Return the sum of stored values in each CSC column."""
        from mlx_sparse._ops import csc_col_sums

        return csc_col_sums(self)

    def column_sums(self) -> mx.array:
        """Alias for :meth:`col_sums`."""
        return self.col_sums()

    def row_norms(self) -> mx.array:
        """Return the dense-semantics L2 norm of each CSC row as ``float32``."""
        from mlx_sparse._ops import csc_row_norms

        return csc_row_norms(self)

    def col_norms(self) -> mx.array:
        """Return the dense-semantics L2 norm of each CSC column as ``float32``."""
        from mlx_sparse._ops import csc_col_norms

        return csc_col_norms(self)

    def column_norms(self) -> mx.array:
        """Alias for :meth:`col_norms`."""
        return self.col_norms()

    def diagonal(self) -> mx.array:
        """Return the summed main diagonal."""
        from mlx_sparse._ops import csc_diagonal

        return csc_diagonal(self)

    def trace(self) -> mx.array:
        """Return the summed main diagonal as a scalar."""
        from mlx_sparse._ops import csc_trace

        return csc_trace(self)

    def sum(self, axis=None) -> mx.array:
        """Sum sparse values over all entries, rows, or columns.

        ``axis=None`` returns a scalar, ``axis=1`` returns row sums, and
        ``axis=0`` returns column sums.
        """
        if axis is None:
            return mx.sum(self.col_sums())
        if axis in (1, -1):
            return self.row_sums()
        if axis in (0, -2):
            return self.col_sums()
        raise ValueError(f"CSCArray.sum axis must be None, 0, or 1; got {axis!r}.")

    def tocsr(self, *, canonical: bool | None = None) -> CSRArray:
        """Convert to :class:`~mlx_sparse.CSRArray`.

        Args:
            canonical: If ``True``, canonicalize the returned CSR matrix. If
                ``False`` or ``None`` (default), return the structural
                conversion as produced by the native count/prefix/fill path.
                The structural path preserves values but does not promise
                sorted output metadata on every backend.
        """
        data, indices, indptr = _native.csc_tocsr(
            self.data,
            self.indices,
            self.indptr,
            self.shape,
        )
        out = CSRArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=self.shape,
            sorted_indices=False,
            has_canonical_format=False,
        )
        if canonical is True:
            return out.canonicalize()
        if canonical is False:
            return CSRArray(
                data=out.data,
                indices=out.indices,
                indptr=out.indptr,
                shape=out.shape,
                sorted_indices=out.sorted_indices,
                has_canonical_format=False,
            )
        return out

    def sort_indices(self) -> "CSCArray":
        """Return a new CSCArray with row indices sorted within each column."""
        if self.sorted_indices:
            return self
        data, indices, indptr = _native.csc_sort_indices(
            self.data,
            self.indices,
            self.indptr,
        )
        return CSCArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=self.shape,
            sorted_indices=True,
            has_canonical_format=False,
        )

    def sum_duplicates(self) -> "CSCArray":
        """Sum duplicate row entries within each column."""
        sorted_self = self.sort_indices()
        data, indices, indptr = _native.csc_sum_duplicates(
            sorted_self.data,
            sorted_self.indices,
            sorted_self.indptr,
        )
        return CSCArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=self.shape,
            sorted_indices=True,
            has_canonical_format=True,
        )

    def canonicalize(self) -> "CSCArray":
        """Return canonical form: sorted row indices, no duplicates."""
        if self.has_canonical_format:
            return self
        return self.sum_duplicates()

    def transpose(self) -> CSRArray:
        """Transpose the sparse matrix, returning a zero-copy CSRArray."""
        return CSRArray(
            data=self.data,
            indices=self.indices,
            indptr=self.indptr,
            shape=(self.shape[1], self.shape[0]),
            sorted_indices=self.sorted_indices,
            has_canonical_format=self.has_canonical_format,
        )

    @property
    def T(self) -> CSRArray:
        """Transposed matrix. Alias for :meth:`transpose`."""
        return self.transpose()

    def conj(self) -> "CSCArray":
        """Complex-conjugate the stored values."""
        return CSCArray(
            data=mx.conjugate(self.data),
            indices=self.indices,
            indptr=self.indptr,
            shape=self.shape,
            sorted_indices=self.sorted_indices,
            has_canonical_format=self.has_canonical_format,
        )

    def conjugate(self) -> "CSCArray":
        """Alias for :meth:`conj`."""
        return self.conj()

    @property
    def H(self) -> CSRArray:
        """Hermitian (conjugate) transpose. Equivalent to ``conj().T``."""
        return self.conj().transpose()

    def __matmul__(self, rhs):
        """Matrix multiplication via the ``@`` operator."""
        from mlx_sparse._coo import COOArray
        from mlx_sparse._ops import csc_matmat, csc_matmul, csc_matvec

        if isinstance(rhs, CSCArray):
            return csc_matmat(self, rhs)
        if isinstance(rhs, CSRArray | COOArray):
            raise NotImplementedError(
                "Mixed-format CSC sparse-sparse matmul is not implemented. "
                "Convert explicitly if another format is acceptable for your workload."
            )

        rhs = ensure_mx_array(rhs)
        if rhs.ndim == 1:
            return csc_matvec(self, rhs)
        if rhs.ndim >= 2:
            return csc_matmul(self, rhs)
        raise ValueError(f"CSC matmul expects rank-1 or higher RHS, got {rhs.shape}.")

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
        other = sanitize_scalar(other)

        return CSCArray(
            data=other * self.data,
            indices=self.indices,
            indptr=self.indptr,
            shape=self.shape,
            sorted_indices=self.sorted_indices,
            has_canonical_format=self.has_canonical_format,
        )

    def __mul__(self, other):
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
        return self.__rmul__(other)


def csc_array(
    arg,
    shape,
    *,
    validate: ValidationMode = "metadata",
    sorted_indices: bool = False,
    canonical: bool | None = None,
) -> CSCArray:
    """Construct a :class:`CSCArray` from explicit CSC buffers."""
    mode = normalize_validation_mode(validate)
    shape = normalize_shape(shape)

    if isinstance(arg, CSCArray):
        array = arg
        if array.shape != shape:
            raise ValueError(
                f"CSCArray shape mismatch: got {array.shape}, expected {shape}."
            )
        return array

    try:
        data, indices, indptr = arg
    except Exception as exc:
        raise TypeError(
            "csc_array expects (data, indices, indptr) or a CSCArray instance."
        ) from exc

    data = ensure_mx_array(data)
    indices = ensure_mx_array(indices)
    indptr = ensure_mx_array(indptr)

    if mode != "none":
        validate_csc_metadata(data, indices, indptr, shape)
    if mode == "full":
        validate_csc_values(indices, indptr, shape, data.shape[0])

    has_canonical_format = bool(canonical) if canonical is not None else False
    if has_canonical_format:
        sorted_indices = True

    return CSCArray(
        data=data,
        indices=indices,
        indptr=indptr,
        shape=shape,
        sorted_indices=sorted_indices,
        has_canonical_format=has_canonical_format,
    )
