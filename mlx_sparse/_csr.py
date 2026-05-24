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
from mlx_sparse._typing import Shape2D, ValidationMode
from mlx_sparse._validation import (
    ensure_mx_array,
    normalize_shape,
    normalize_validation_mode,
    validate_csr_metadata,
    validate_csr_values,
)


@dataclass(frozen=True, slots=True)
class CSRArray:
    """A 2D sparse matrix in Compressed Sparse Row (CSR) format.

    CSRArray stores a 2D sparse matrix using three MLX arrays:

    - **data**: non-zero values, shape ``(nnz,)``.
    - **indices**: column index of each stored value, shape ``(nnz,)``.
    - **indptr**: row pointer array, shape ``(n_rows + 1,)``.

    Row ``i`` spans ``data[indptr[i] : indptr[i+1]]`` with corresponding
    column indices ``indices[indptr[i] : indptr[i+1]]``. Duplicate column
    entries are permitted unless the matrix is in canonical form.

    **Format invariants** (checked by ``validate="metadata"`` by default):

    - All three arrays must be rank-1.
    - ``data.shape[0] == indices.shape[0]`` (the ``nnz`` count).
    - ``indptr.shape[0] == n_rows + 1``.
    - ``indices`` and ``indptr`` share the same integer dtype (``int32`` or
      ``int64``).
    - ``data`` dtype is one of ``float32``, ``float16``, ``bfloat16``, or
      ``complex64``.

    **Additional value-level invariants** (``validate="full"`` only):

    - ``indptr[0] == 0``, ``indptr[-1] == nnz``.
    - ``indptr`` is monotonically nondecreasing.
    - ``0 <= indices[j] < n_cols`` for all stored values.

    ``CSRArray`` is immutable (frozen dataclass). Structural operations return
    new instances. The ``sorted_indices`` and ``has_canonical_format`` flags are
    metadata hints. Set them only when the input is already known to satisfy
    those properties. Use ``canonicalize()`` to sort and sum duplicates.

    Args:
        data: Non-zero values, shape ``(nnz,)``.
        indices: Column indices, shape ``(nnz,)``.
        indptr: Row pointer array, shape ``(n_rows + 1,)``.
        shape: Matrix dimensions as ``(n_rows, n_cols)``.
        sorted_indices: Hint that column indices within each row are sorted
            ascending. Defaults to ``False``.
        has_canonical_format: Hint that the matrix has sorted column indices
            and no duplicate column index in any row. Implies
            ``sorted_indices=True``. Defaults to ``False``.

    Example::

        import mlx.core as mx
        import mlx_sparse as ms

        data = mx.array([2.0, -1.0, 4.0, 5.0], dtype=mx.float32)
        indices = mx.array([0, 2, 1, 3], dtype=mx.int32)
        indptr = mx.array([0, 2, 2, 4], dtype=mx.int32)
        A = ms.csr_array((data, indices, indptr), shape=(3, 4))
        x = mx.array([1.0, 0.0, 1.0, 1.0], dtype=mx.float32)
        y = A @ x  # shape (3,)
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
        """Value dtype of the stored non-zeros (e.g. ``mlx.core.float32``)."""
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
            "CSRArray("
            f"shape={self.shape}, nnz={self.nnz}, dtype={self.dtype}, "
            f"index_dtype={self.index_dtype}, "
            f"sorted_indices={self.sorted_indices}, "
            f"has_canonical_format={self.has_canonical_format})"
        )

    def todense(self) -> mx.array:
        """Materialize the sparse matrix as a dense MLX array.

        Duplicate column entries in the same row are summed, matching the
        semantics of ``canonicalize().todense()``.

        Returns:
            Dense array of shape ``(n_rows, n_cols)`` and the same dtype as
            ``self.data``.
        """
        return _native.csr_todense(self.data, self.indices, self.indptr, self.shape)

    def row_sums(self) -> mx.array:
        """Return the sum of stored values in each CSR row."""
        return _native.csr_row_sums(self.data, self.indices, self.indptr, self.shape)

    def col_sums(self) -> mx.array:
        """Return the sum of stored values in each CSR column."""
        return _native.csr_col_sums(self.data, self.indices, self.indptr, self.shape)

    def column_sums(self) -> mx.array:
        """Alias for :meth:`col_sums`."""
        return self.col_sums()

    def row_norms(self) -> mx.array:
        """Return the L2 norm of each CSR row as ``float32``."""
        array = self if self.has_canonical_format else self.canonicalize()
        return _native.csr_row_norms(
            array.data,
            array.indices,
            array.indptr,
            array.shape,
        )

    def diagonal(self) -> mx.array:
        """Return the summed main diagonal."""
        return _native.csr_diagonal(self.data, self.indices, self.indptr, self.shape)

    def trace(self) -> mx.array:
        """Return the summed main diagonal as a scalar."""
        return _native.csr_trace(self.data, self.indices, self.indptr, self.shape)

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
        raise ValueError(f"CSRArray.sum axis must be None, 0, or 1; got {axis!r}.")

    def sort_indices(self) -> "CSRArray":
        """Return a new CSRArray with column indices sorted within each row.

        If ``self.sorted_indices`` is already ``True``, returns ``self``
        unchanged (no copy). Otherwise dispatches the native sort primitive.

        Returns:
            A new ``CSRArray`` with ``sorted_indices=True`` and
            ``has_canonical_format=False`` (duplicates may still be present).
        """
        if self.sorted_indices:
            return self
        data, indices, indptr = _native.csr_sort_indices(
            self.data,
            self.indices,
            self.indptr,
        )
        return CSRArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=self.shape,
            sorted_indices=True,
            has_canonical_format=False,
        )

    def sum_duplicates(self) -> "CSRArray":
        """Sum duplicate column entries within each row.

        Sorts indices first (via ``sort_indices``), then accumulates entries
        that share the same column index. The resulting ``nnz`` may be smaller
        than the original.

        Returns:
            A new ``CSRArray`` with ``sorted_indices=True`` and
            ``has_canonical_format=True``.
        """
        sorted_self = self.sort_indices()
        data, indices, indptr = _native.csr_sum_duplicates(
            sorted_self.data,
            sorted_self.indices,
            sorted_self.indptr,
        )
        return CSRArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=self.shape,
            sorted_indices=True,
            has_canonical_format=True,
        )

    def canonicalize(self) -> "CSRArray":
        """Return the canonical form: sorted indices, no duplicates.

        If ``self.has_canonical_format`` is already ``True``, returns ``self``
        with no work done. Otherwise calls ``sum_duplicates()``.

        Returns:
            A ``CSRArray`` with ``has_canonical_format=True``.
        """
        if self.has_canonical_format:
            return self
        return self.sum_duplicates()

    def transpose(self) -> "CSRArray":
        """Transpose the sparse matrix, returning a new CSRArray.

        The result has ``shape=(n_cols, n_rows)`` and ``sorted_indices=True``.
        If the source has ``has_canonical_format=True``, the result also
        inherits that flag.

        Returns:
            A new ``CSRArray`` with shape ``(n_cols, n_rows)``.
        """
        data, indices, indptr = _native.csr_transpose(
            self.data,
            self.indices,
            self.indptr,
            self.shape,
        )
        return CSRArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=(self.shape[1], self.shape[0]),
            sorted_indices=True,
            has_canonical_format=self.has_canonical_format,
        )

    def tocsc(self, *, canonical: bool | None = None):
        """Convert to :class:`~mlx_sparse.CSCArray`.

        Args:
            canonical: If ``True``, canonicalize the returned CSC matrix. If
                ``False`` or ``None`` (default), return the structural
                conversion as produced by the native count/prefix/fill path.
                The structural path preserves values but does not promise
                sorted output metadata on every backend.
        """
        from mlx_sparse._csc import CSCArray

        data, indices, indptr = _native.csr_tocsc(
            self.data,
            self.indices,
            self.indptr,
            self.shape,
        )
        out = CSCArray(
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
            return CSCArray(
                data=out.data,
                indices=out.indices,
                indptr=out.indptr,
                shape=out.shape,
                sorted_indices=out.sorted_indices,
                has_canonical_format=False,
            )
        return out

    @property
    def T(self) -> "CSRArray":
        """Transposed matrix. Alias for :meth:`transpose`."""
        return self.transpose()

    def conj(self) -> "CSRArray":
        """Complex-conjugate the stored values.

        Structure (indices, indptr, shape) is shared. For real dtypes this is
        a no-op at the value level but still returns a new ``CSRArray``
        pointing to the conjugated data array.

        Returns:
            A new ``CSRArray`` with conjugated ``data``.
        """
        return CSRArray(
            data=mx.conjugate(self.data),
            indices=self.indices,
            indptr=self.indptr,
            shape=self.shape,
            sorted_indices=self.sorted_indices,
            has_canonical_format=self.has_canonical_format,
        )

    def conjugate(self) -> "CSRArray":
        """Alias for :meth:`conj`."""
        return self.conj()

    @property
    def H(self) -> "CSRArray":
        """Hermitian (conjugate) transpose. Equivalent to ``conj().T``."""
        return self.conj().transpose()

    def vdot(self, other) -> mx.array:
        """Sparse Frobenius inner product with another sparse array.

        Both operands are canonicalized CSR matrices and the matching-column
        merge is executed by the native sparse kernel. Dense materialization is
        never used.
        """
        rhs = other
        if isinstance(rhs, CSRArray):
            rhs_csr = rhs.canonicalize()
        else:
            from mlx_sparse._coo import COOArray

            if not isinstance(rhs, COOArray):
                raise TypeError(
                    f"CSRArray.vdot expects CSRArray or COOArray, got {type(rhs).__name__}."
                )
            rhs_csr = rhs.tocsr(canonical=True)
        if rhs_csr.shape != self.shape:
            raise ValueError(
                f"vdot shape mismatch: got {self.shape} and {rhs_csr.shape}."
            )
        lhs = self.canonicalize()
        if lhs.data.dtype in {mx.float16, mx.bfloat16}:
            lhs = CSRArray(
                data=lhs.data.astype(mx.float32),
                indices=lhs.indices,
                indptr=lhs.indptr,
                shape=lhs.shape,
                sorted_indices=lhs.sorted_indices,
                has_canonical_format=lhs.has_canonical_format,
            )
        if rhs_csr.data.dtype in {mx.float16, mx.bfloat16}:
            rhs_csr = CSRArray(
                data=rhs_csr.data.astype(mx.float32),
                indices=rhs_csr.indices,
                indptr=rhs_csr.indptr,
                shape=rhs_csr.shape,
                sorted_indices=rhs_csr.sorted_indices,
                has_canonical_format=rhs_csr.has_canonical_format,
            )
        if lhs.data.dtype != rhs_csr.data.dtype:
            raise TypeError(
                "CSRArray.vdot operands must have the same dtype after low-precision promotion."
            )
        if lhs.data.dtype not in {mx.float32, mx.complex64}:
            raise TypeError("CSRArray.vdot supports float32 and complex64 data.")
        return _native.csr_vdot(
            lhs.data,
            lhs.indices,
            lhs.indptr,
            rhs_csr.data,
            rhs_csr.indices,
            rhs_csr.indptr,
            lhs.shape,
        )

    def dot(self, other) -> mx.array:
        """Sparse Frobenius dot product with another sparse array.

        Unlike :meth:`vdot`, complex operands are not conjugated.
        """
        rhs = other
        if isinstance(rhs, CSRArray):
            rhs_csr = rhs.canonicalize()
        else:
            from mlx_sparse._coo import COOArray

            if not isinstance(rhs, COOArray):
                raise TypeError(
                    f"CSRArray.dot expects CSRArray or COOArray, got {type(rhs).__name__}."
                )
            rhs_csr = rhs.tocsr(canonical=True)
        if rhs_csr.shape != self.shape:
            raise ValueError(
                f"dot shape mismatch: got {self.shape} and {rhs_csr.shape}."
            )
        lhs = self.canonicalize()
        if lhs.data.dtype in {mx.float16, mx.bfloat16}:
            lhs = CSRArray(
                data=lhs.data.astype(mx.float32),
                indices=lhs.indices,
                indptr=lhs.indptr,
                shape=lhs.shape,
                sorted_indices=lhs.sorted_indices,
                has_canonical_format=lhs.has_canonical_format,
            )
        if rhs_csr.data.dtype in {mx.float16, mx.bfloat16}:
            rhs_csr = CSRArray(
                data=rhs_csr.data.astype(mx.float32),
                indices=rhs_csr.indices,
                indptr=rhs_csr.indptr,
                shape=rhs_csr.shape,
                sorted_indices=rhs_csr.sorted_indices,
                has_canonical_format=rhs_csr.has_canonical_format,
            )
        if lhs.data.dtype != rhs_csr.data.dtype:
            raise TypeError(
                "CSRArray.dot operands must have the same dtype after low-precision promotion."
            )
        if lhs.data.dtype not in {mx.float32, mx.complex64}:
            raise TypeError("CSRArray.dot supports float32 and complex64 data.")
        return _native.csr_dot(
            lhs.data,
            lhs.indices,
            lhs.indptr,
            rhs_csr.data,
            rhs_csr.indices,
            rhs_csr.indptr,
            lhs.shape,
        )

    def __matmul__(self, rhs):
        """Matrix multiplication via the ``@`` operator.

        Dispatches to :func:`~mlx_sparse.csr_matmat` for CSR operands,
        :func:`~mlx_sparse.csr_matvec` for a rank-1 dense ``rhs``, or
        :func:`~mlx_sparse.csr_matmul` for rank-2 and batched dense operands.
        Dense inputs are converted to MLX arrays if needed.

        Args:
            rhs: CSR sparse matrix, dense vector of shape ``(n_cols,)``, dense
                matrix of shape ``(n_cols, k)``, or batched dense matrix with
                sparse dimension at ``rhs.shape[-2]``.

        Returns:
            A CSRArray for CSR RHS, otherwise a dense MLX array.

        Raises:
            ValueError: If dense ``rhs.ndim`` is not at least 1.
            TypeError: If ``rhs`` dtype does not match ``self.data`` dtype.
        """
        if isinstance(rhs, CSRArray):
            from mlx_sparse._ops import csr_matmat

            return csr_matmat(self, rhs)

        from mlx_sparse._coo import COOArray

        if isinstance(rhs, COOArray):
            from mlx_sparse._ops import csr_matmat

            return csr_matmat(self, rhs.tocsr(canonical=True))

        rhs = ensure_mx_array(rhs)
        if rhs.ndim == 1:
            from mlx_sparse._ops import csr_matvec

            return csr_matvec(self, rhs)
        if rhs.ndim >= 2:
            from mlx_sparse._ops import csr_matmul

            return csr_matmul(self, rhs)
        raise ValueError(f"CSR matmul expects rank-1 or higher RHS, got {rhs.shape}.")


def csr_array(
    arg,
    shape,
    *,
    validate: ValidationMode = "metadata",
    sorted_indices: bool = False,
    canonical: bool | None = None,
) -> CSRArray:
    """Construct a :class:`CSRArray` from explicit CSR buffers.

    Accepts either a ``(data, indices, indptr)`` triple or an existing
    ``CSRArray``. All array inputs are converted to ``mlx.core.array`` if they
    are not already.

    Args:
        arg: A length-3 iterable ``(data, indices, indptr)`` where

            - *data*: non-zero values, shape ``(nnz,)``, dtype
              ``float32 | float16 | bfloat16 | complex64``.
            - *indices*: column indices, shape ``(nnz,)``, dtype
              ``int32 | int64``.
            - *indptr*: row pointers, shape ``(n_rows + 1,)``, same integer
              dtype as *indices*.

            Alternatively, an existing :class:`CSRArray` instance (returned
            unchanged if ``shape`` matches).

        shape: Matrix dimensions as a length-2 sequence ``(n_rows, n_cols)``.
        validate: Validation level, one of:

            - ``"metadata"`` *(default)*: checks ranks, lengths, and dtypes
              without reading array values. Safe to call on device arrays.
            - ``"full"`` / ``True``: also verifies ``indptr`` monotonicity
              and column index bounds. May synchronize to host.
            - ``False`` / ``"none"``: skips all checks.

        sorted_indices: Set to ``True`` to assert that column indices within
            each row are already sorted ascending. Default ``False``.
        canonical: Set to ``True`` to assert canonical format (sorted indices,
            no duplicate columns). Implies ``sorted_indices=True``. Default
            ``None`` (not asserted).

    Returns:
        A :class:`CSRArray` with the given buffers and shape.

    Raises:
        TypeError: If ``arg`` is not a 3-tuple or a ``CSRArray``, or if dtype
            constraints are violated.
        ValueError: If shape or length constraints are violated.

    Example::

        import mlx.core as mx
        import mlx_sparse as ms

        data = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        indices = mx.array([0, 1, 0], dtype=mx.int32)
        indptr = mx.array([0, 2, 3], dtype=mx.int32)

        A = ms.csr_array((data, indices, indptr), shape=(2, 2))
        # Full validation from host arrays:
        A_checked = ms.csr_array(
            (data, indices, indptr), shape=(2, 2), validate="full"
        )
    """
    mode = normalize_validation_mode(validate)
    shape = normalize_shape(shape)

    if isinstance(arg, CSRArray):
        array = arg
        if array.shape != shape:
            raise ValueError(
                f"CSRArray shape mismatch: got {array.shape}, expected {shape}."
            )
        return array

    try:
        data, indices, indptr = arg
    except Exception as exc:
        raise TypeError(
            "csr_array expects (data, indices, indptr) or a CSRArray instance."
        ) from exc

    data = ensure_mx_array(data)
    indices = ensure_mx_array(indices)
    indptr = ensure_mx_array(indptr)

    if mode != "none":
        validate_csr_metadata(data, indices, indptr, shape)
    if mode == "full":
        validate_csr_values(indices, indptr, shape, data.shape[0])

    has_canonical_format = bool(canonical) if canonical is not None else False
    if has_canonical_format:
        sorted_indices = True

    return CSRArray(
        data=data,
        indices=indices,
        indptr=indptr,
        shape=shape,
        sorted_indices=sorted_indices,
        has_canonical_format=has_canonical_format,
    )
