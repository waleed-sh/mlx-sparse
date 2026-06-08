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

from functools import reduce
from operator import mul

import mlx.core as mx

import mlx_sparse._native as _native
from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._typing import VALUE_DTYPES
from mlx_sparse._validation import (
    ensure_mx_array,
    sanitize_scalar,
    validate_coo_matmul_inputs,
    validate_coo_matvec_inputs,
    validate_csc_matmul_inputs,
    validate_csc_matvec_inputs,
    validate_csc_matvec_transpose_inputs,
    validate_csr_matmul_inputs,
    validate_csr_matvec_inputs,
    validate_csr_metadata,
)

_SUPPORTED_STRUCTURAL_FORMATS = {"coo", "csr", "csc"}
_UNSUPPORTED_SCIPY_FORMATS = {"bsr", "dia", "dok", "lil"}
_MAX_MLX_DIM = 2**31 - 1


def _prod(values) -> int:
    return int(reduce(mul, values, 1))


def identity_like(x: mx.array) -> mx.array:
    """Return a native MLX copy of ``x``.

    This function exists as an extension smoke test. It passes ``x`` through
    the native ``_ext`` module (if available) and returns an identical MLX
    array. For production code, prefer ``mlx.core`` operations directly.

    Args:
        x: Any MLX array.

    Returns:
        An MLX array with the same shape, dtype, and values as ``x``.
    """
    return _native.identity_like(ensure_mx_array(x))


def todense(array) -> mx.array:
    """Materialize a sparse array as a dense MLX array.

    Convenience wrapper that calls ``array.todense()`` on any sparse container.
    Duplicate entries are summed, consistent with ``canonicalize().todense()``.

    Args:
        array: A :class:`~mlx_sparse.COOArray`, :class:`~mlx_sparse.CSRArray`,
            or :class:`~mlx_sparse.CSCArray` instance.

    Returns:
        Dense array of shape ``(n_rows, n_cols)`` with the same dtype as
        ``array.data``.

    Raises:
        TypeError: If ``array`` does not have a ``todense`` method.

    Example::

        import mlx_sparse as ms

        dense = ms.todense(my_csr)
    """
    if hasattr(array, "todense"):
        return array.todense()
    raise TypeError(f"todense expects an mlx-sparse array, got {type(array).__name__}.")


def _is_sparse_array(value) -> bool:
    return isinstance(value, (COOArray, CSRArray, CSCArray))


def _sparse_name(value) -> str:
    return type(value).__name__


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


def _promote_kron_dtype(lhs_dtype, rhs_dtype):
    if lhs_dtype not in VALUE_DTYPES or rhs_dtype not in VALUE_DTYPES:
        raise TypeError(
            "kron operands must have supported sparse value dtypes after dense "
            "normalization."
        )
    if lhs_dtype == mx.complex64 or rhs_dtype == mx.complex64:
        return mx.complex64
    if lhs_dtype == mx.float32 or rhs_dtype == mx.float32:
        return mx.float32
    if lhs_dtype == rhs_dtype:
        return lhs_dtype
    return mx.float32


def _supported_dense_value_dtype(dtype):
    if dtype in VALUE_DTYPES:
        return dtype
    return mx.float32


def _astype_sparse_value(array, dtype):
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


def _as_sparse_rank2(name: str, value):
    if _is_sparse_array(value):
        return value

    dense = ensure_mx_array(value)
    if dense.ndim != 2:
        raise ValueError(
            f"{name} must be a sparse array or dense rank-2 array, "
            f"got shape={dense.shape}."
        )
    dtype = _supported_dense_value_dtype(dense.dtype)
    from mlx_sparse._construct import fromdense

    return fromdense(dense, dtype=dtype)


def _to_coo_for_kron(name: str, value) -> COOArray:
    if isinstance(value, COOArray):
        return value
    if isinstance(value, CSRArray):
        return value.tocoo(canonical=None)
    if isinstance(value, CSCArray):
        return value.tocoo(canonical=False)
    raise TypeError(f"{name} expects a COOArray, CSRArray, or CSCArray.")


def _checked_product_for_kron(lhs: int, rhs: int, name: str) -> int:
    if lhs < 0 or rhs < 0:
        raise ValueError(f"{name} dimensions must be non-negative.")
    if lhs and rhs > _MAX_MLX_DIM // lhs:
        raise OverflowError(f"{name} exceeds MLX rank-2 shape limits.")
    return lhs * rhs


def _check_kron_overflow(lhs, rhs) -> tuple[int, int]:
    out_rows = _checked_product_for_kron(lhs.shape[0], rhs.shape[0], "kron rows")
    out_cols = _checked_product_for_kron(lhs.shape[1], rhs.shape[1], "kron columns")
    _checked_product_for_kron(lhs.nnz, rhs.nnz, "kron nnz")
    return out_rows, out_cols


def _coo_kron_raw(lhs: COOArray, rhs: COOArray) -> COOArray:
    out_shape = _check_kron_overflow(lhs, rhs)
    data, row, col = _native.coo_kron(lhs, rhs)
    return COOArray(
        data=data,
        row=row,
        col=col,
        shape=out_shape,
        has_canonical_format=(
            bool(lhs.has_canonical_format) and bool(rhs.has_canonical_format)
        ),
    )


def _coo_to_requested_format(coo: COOArray, format: str):
    if format == "coo":
        return coo
    if format == "csr":
        return coo.tocsr(canonical=True)
    if format == "csc":
        return coo.tocsc(canonical=True)
    raise ValueError(f"unsupported sparse format {format!r}.")


def _as_canonical_csr(name: str, value) -> CSRArray:
    if isinstance(value, CSRArray):
        return value.canonicalize()
    if isinstance(value, COOArray):
        return value.tocsr(canonical=True)
    if isinstance(value, CSCArray):
        return value.tocsr(canonical=True)
    raise TypeError(f"{name} expects a sparse array, got {_sparse_name(value)}.")


def kron(A, B, format=None):
    """Return the sparse Kronecker product of two rank-2 operands.

    ``kron(A, B)`` builds the matrix whose stored entries follow
    ``row = row_A * B.shape[0] + row_B``,
    ``col = col_A * B.shape[1] + col_B``, and
    ``data = data_A * data_B``. COO, CSR, CSC, and dense rank-2 MLX-compatible
    inputs are accepted; dense inputs are converted with the native
    :func:`mlx_sparse.fromdense` path before assembly, never with Python loops
    over entries.

    ``format`` may be ``"coo"``, ``"csr"``, ``"csc"``, or ``None``. The
    default is COO, matching the construction-oriented SciPy API. COO output is
    the direct native fixed-topology product and preserves duplicate structural
    entries if either input contains duplicates. CSR and CSC output canonicalize
    through native compressed conversion, summing duplicate products and
    returning duplicate-free compressed structures. Unsupported SciPy formats
    such as ``"bsr"``, ``"dia"``, ``"dok"``, and ``"lil"`` are rejected
    explicitly.

    Value dtype promotion follows the package's sparse value constraints:
    ``complex64`` wins over real dtypes, any ``float32`` operand yields
    ``float32``, equal low-precision operands keep their dtype, and mixed
    ``float16``/``bfloat16`` promotes to ``float32``. Dense integer or boolean
    operands are converted to ``float32`` because mlx-sparse sparse containers
    do not store integer or boolean value buffers in this release.

    Sparse-value JVP/VJP is implemented for the native COO data product when
    the input structures are fixed. Gradients through integer coordinates,
    dense-to-sparse extraction, and duplicate-summing canonicalization are not
    part of the differentiable contract.

    Args:
        A: Left COO, CSR, CSC, or dense rank-2 operand.
        B: Right COO, CSR, CSC, or dense rank-2 operand.
        format: Output format, one of ``None``, ``"coo"``, ``"csr"``, or
            ``"csc"``. ``None`` defaults to ``"coo"``.

    Returns:
        A :class:`~mlx_sparse.COOArray`, :class:`~mlx_sparse.CSRArray`, or
        :class:`~mlx_sparse.CSCArray` with shape
        ``(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])``.

    Raises:
        ValueError: If an operand is not rank-2, the requested format is
            unknown, or output dimensions exceed MLX limits.
        TypeError: If ``format`` is not a string or ``None``.
        NotImplementedError: If a known unsupported SciPy sparse format is
            requested.
    """
    out_format = _normalize_sparse_format("kron", format, default="coo")
    lhs = _as_sparse_rank2("kron A", A)
    rhs = _as_sparse_rank2("kron B", B)
    dtype = _promote_kron_dtype(lhs.data.dtype, rhs.data.dtype)
    lhs = _astype_sparse_value(lhs, dtype)
    rhs = _astype_sparse_value(rhs, dtype)
    lhs_coo = _to_coo_for_kron("kron A", lhs)
    rhs_coo = _to_coo_for_kron("kron B", rhs)
    return _coo_to_requested_format(_coo_kron_raw(lhs_coo, rhs_coo), out_format)


def kronsum(A, B, format=None):
    """Return the Kronecker sum of two square sparse or dense matrices.

    The Kronecker sum is defined as ``kron(I_n, A) + kron(B, I_m)`` for
    ``A.shape == (m, m)`` and ``B.shape == (n, n)``. Inputs may be COO, CSR,
    CSC, or dense rank-2 arrays. Dense inputs are extracted with native
    :func:`mlx_sparse.fromdense`; the two Kronecker products are assembled with
    native COO kernels and the sum is merged with native sparse addition.

    ``format`` may be ``"coo"``, ``"csr"``, ``"csc"``, or ``None``. The
    default is COO. The intermediate sum is canonical CSR, so returned CSR and
    CSC outputs are canonical; returned COO is produced by native CSR-to-COO
    expansion and is also canonical.

    Args:
        A: Left square COO, CSR, CSC, or dense rank-2 operand.
        B: Right square COO, CSR, CSC, or dense rank-2 operand.
        format: Output format, one of ``None``, ``"coo"``, ``"csr"``, or
            ``"csc"``. ``None`` defaults to ``"coo"``.

    Returns:
        A sparse array with shape ``(A.shape[0] * B.shape[0],
        A.shape[1] * B.shape[1])``.

    Raises:
        ValueError: If either operand is not square or if output shape/nnz
            limits are exceeded.
        TypeError: If ``format`` is not a string or ``None``.
        NotImplementedError: If a known unsupported SciPy sparse format is
            requested.
    """
    out_format = _normalize_sparse_format("kronsum", format, default="coo")
    lhs = _as_sparse_rank2("kronsum A", A)
    rhs = _as_sparse_rank2("kronsum B", B)
    if lhs.shape[0] != lhs.shape[1]:
        raise ValueError(f"kronsum A must be square, got shape={lhs.shape}.")
    if rhs.shape[0] != rhs.shape[1]:
        raise ValueError(f"kronsum B must be square, got shape={rhs.shape}.")

    dtype = _promote_kron_dtype(lhs.data.dtype, rhs.data.dtype)
    lhs = _astype_sparse_value(lhs, dtype)
    rhs = _astype_sparse_value(rhs, dtype)

    from mlx_sparse._construct import eye

    index_dtype = (
        mx.int64
        if (
            getattr(lhs, "index_dtype", mx.int32) == mx.int64
            or getattr(rhs, "index_dtype", mx.int32) == mx.int64
        )
        else mx.int32
    )
    lhs_identity = eye(rhs.shape[0], dtype=dtype, index_dtype=index_dtype)
    rhs_identity = eye(lhs.shape[0], dtype=dtype, index_dtype=index_dtype)
    left = kron(lhs_identity, lhs, format="csr")
    right = kron(rhs, rhs_identity, format="csr")
    summed = add(left, right)
    if out_format == "csr":
        return summed
    if out_format == "csc":
        return summed.tocsc(canonical=True)
    return summed.tocoo(canonical=True)


def _is_zero_python_scalar(value) -> bool:
    sanitize_scalar(value)
    return bool(value == 0)


def _handle_sparse_scalar_addition(
    sparse, scalar, *, subtract: bool, scalar_left: bool
):
    try:
        is_zero = _is_zero_python_scalar(scalar)
    except TypeError as exc:
        raise TypeError(
            "Sparse addition only supports sparse-sparse operands. "
            "Sparse+dense addition is intentionally unsupported because it "
            "would produce a dense result; call sparse.todense() explicitly."
        ) from exc

    if is_zero:
        if subtract and scalar_left:
            return (-1) * sparse
        return sparse

    if subtract and scalar_left:
        raise NotImplementedError(
            "Subtracting a sparse array from a nonzero scalar would produce a "
            "dense matrix. Call sparse.todense() and subtract explicitly."
        )
    raise NotImplementedError(
        "Adding or subtracting a nonzero scalar from a sparse array would "
        "produce a dense matrix. Call sparse.todense() explicitly."
    )


def _convert_csr_add_output(result: CSRArray, lhs, rhs):
    if isinstance(lhs, CSCArray) and isinstance(rhs, CSCArray):
        return result.tocsc(canonical=True)
    return result


def add(A, B):
    """Add two sparse arrays without densifying.

    Computes ``A + B`` for rank-2 mlx-sparse arrays with equal shape and
    matching value dtype. The production path canonicalizes both operands with
    native sort/sum kernels, merges their CSR structures in native C++ or Metal,
    sums duplicate coordinates, and removes exact zero cancellations from the
    result.

    CSR inputs return a canonical :class:`~mlx_sparse.CSRArray`. Homogeneous
    CSC inputs return a canonical :class:`~mlx_sparse.CSCArray` via native
    CSR/CSC conversion. COO and mixed-format inputs return canonical CSR output
    so no dense matrix is created.

    Sparse+dense addition is intentionally out of scope for this release:
    adding a sparse matrix to a dense matrix returns a dense matrix
    mathematically, and this API does not hide that cost. Add or subtract a
    Python scalar only when the scalar is exactly zero; nonzero scalar addition
    is rejected for the same reason.

    The output structure depends on the input structures and on exact numerical
    cancellation, so public sparse addition is treated as a dynamic-topology
    operation. Gradients through integer structure are unsupported, and no
    fixed-topology sparse-value autodiff contract is claimed for this dynamic
    operation.

    Args:
        A: Left sparse operand, or the scalar ``0`` for ``0 + B``.
        B: Right sparse operand, or the scalar ``0`` for ``A + 0``.

    Returns:
        A canonical sparse array. The result is CSR except for homogeneous CSC
        inputs, which return CSC.

    Raises:
        TypeError: If operands are dense, shapes differ, or value dtypes differ.
        NotImplementedError: If nonzero scalar addition would densify.
    """
    if _is_sparse_array(A) and _is_sparse_array(B):
        if A.shape != B.shape:
            raise ValueError(f"sparse add shape mismatch: got {A.shape} and {B.shape}.")
        if A.data.dtype != B.data.dtype:
            raise TypeError(
                "Sparse add requires matching value dtypes, "
                f"got {A.data.dtype} and {B.data.dtype}."
            )
        lhs = _as_canonical_csr("add lhs", A)
        rhs = _as_canonical_csr("add rhs", B)
        data, indices, indptr = _native.csr_add(lhs, rhs, subtract=False)
        result = CSRArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=A.shape,
            sorted_indices=True,
            has_canonical_format=True,
        )
        return _convert_csr_add_output(result, A, B)

    if _is_sparse_array(A):
        return _handle_sparse_scalar_addition(A, B, subtract=False, scalar_left=False)
    if _is_sparse_array(B):
        return _handle_sparse_scalar_addition(B, A, subtract=False, scalar_left=True)
    raise TypeError(
        "add expects at least one mlx-sparse COOArray, CSRArray, or CSCArray operand."
    )


def subtract(A, B):
    """Subtract two sparse arrays without densifying.

    Computes ``A - B`` for rank-2 mlx-sparse arrays with equal shape and
    matching value dtype. Semantics match :func:`add`: inputs are canonicalized
    natively, the structural union is merged in CSR form, duplicate coordinates
    are summed, and exact zero cancellations are pruned from the canonical
    output. Homogeneous CSC inputs return CSC; all other supported sparse
    combinations return CSR.

    Sparse-dense subtraction and nonzero scalar subtraction are rejected because
    they would produce dense results. The scalar ``0`` is accepted as the
    additive identity: ``A - 0`` returns ``A`` and ``0 - A`` returns ``-A`` as a
    sparse array with the same structure.

    The output topology is dynamic because exact cancellation can remove stored
    entries. Gradients through the public sparse subtraction structure are not
    claimed in this release.

    Args:
        A: Left sparse operand, or scalar ``0`` for ``0 - B``.
        B: Right sparse operand, or scalar ``0`` for ``A - 0``.

    Returns:
        A canonical sparse array. The result is CSR except for homogeneous CSC
        inputs, which return CSC.

    Raises:
        TypeError: If operands are dense, shapes differ, or value dtypes differ.
        NotImplementedError: If nonzero scalar subtraction would densify.
    """
    if _is_sparse_array(A) and _is_sparse_array(B):
        if A.shape != B.shape:
            raise ValueError(
                f"sparse subtract shape mismatch: got {A.shape} and {B.shape}."
            )
        if A.data.dtype != B.data.dtype:
            raise TypeError(
                "Sparse subtract requires matching value dtypes, "
                f"got {A.data.dtype} and {B.data.dtype}."
            )
        lhs = _as_canonical_csr("subtract lhs", A)
        rhs = _as_canonical_csr("subtract rhs", B)
        data, indices, indptr = _native.csr_add(lhs, rhs, subtract=True)
        result = CSRArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=A.shape,
            sorted_indices=True,
            has_canonical_format=True,
        )
        return _convert_csr_add_output(result, A, B)

    if _is_sparse_array(A):
        return _handle_sparse_scalar_addition(A, B, subtract=True, scalar_left=False)
    if _is_sparse_array(B):
        return _handle_sparse_scalar_addition(B, A, subtract=True, scalar_left=True)
    raise TypeError(
        "subtract expects at least one mlx-sparse COOArray, CSRArray, or CSCArray operand."
    )


def _ensure_csr_array(name: str, a) -> CSRArray:
    if not isinstance(a, CSRArray):
        raise TypeError(f"{name} expects CSRArray, got {type(a).__name__}.")
    validate_csr_metadata(a.data, a.indices, a.indptr, a.shape)
    return a


def _ensure_csc_array(name: str, a) -> CSCArray:
    if not isinstance(a, CSCArray):
        raise TypeError(f"{name} expects CSCArray, got {type(a).__name__}.")
    return a


def _ensure_coo_array(name: str, a) -> COOArray:
    if not isinstance(a, COOArray):
        raise TypeError(f"{name} expects COOArray, got {type(a).__name__}.")
    return a


def csr_row_sums(a: CSRArray) -> mx.array:
    """Reduce each row of a CSR matrix to the sum of its stored values."""
    a = _ensure_csr_array("csr_row_sums", a)
    return _native.csr_row_sums(a.data, a.indices, a.indptr, a.shape)


def csr_col_sums(a: CSRArray) -> mx.array:
    """Reduce each column of a CSR matrix to the sum of its stored values."""
    a = _ensure_csr_array("csr_col_sums", a)
    return _native.csr_col_sums(a.data, a.indices, a.indptr, a.shape)


def csr_column_sums(a: CSRArray) -> mx.array:
    """Alias for :func:`csr_col_sums`."""
    return csr_col_sums(a)


def csr_row_norms(a: CSRArray) -> mx.array:
    """Compute the L2 norm of each CSR row."""
    a = _ensure_csr_array("csr_row_norms", a)
    if not a.has_canonical_format:
        a = a.canonicalize()
    return _native.csr_row_norms(a.data, a.indices, a.indptr, a.shape)


def csr_diagonal(a: CSRArray) -> mx.array:
    """Extract the summed diagonal of a CSR matrix."""
    a = _ensure_csr_array("csr_diagonal", a)
    return _native.csr_diagonal(a.data, a.indices, a.indptr, a.shape)


def csr_trace(a: CSRArray) -> mx.array:
    """Compute the trace of a CSR matrix."""
    a = _ensure_csr_array("csr_trace", a)
    return _native.csr_trace(a.data, a.indices, a.indptr, a.shape)


def coo_row_sums(a: COOArray) -> mx.array:
    """Reduce each row of a COO matrix to the sum of its stored values."""
    a = _ensure_coo_array("coo_row_sums", a)
    return _native.coo_row_sums(a.data, a.row, a.col, a.shape)


def coo_col_sums(a: COOArray) -> mx.array:
    """Reduce each column of a COO matrix to the sum of its stored values."""
    a = _ensure_coo_array("coo_col_sums", a)
    return _native.coo_col_sums(a.data, a.row, a.col, a.shape)


def coo_column_sums(a: COOArray) -> mx.array:
    """Alias for :func:`coo_col_sums`."""
    return coo_col_sums(a)


def coo_row_norms(a: COOArray) -> mx.array:
    """Compute the dense-semantics L2 norm of each COO row."""
    a = _ensure_coo_array("coo_row_norms", a)
    if not a.has_canonical_format:
        return a.tocsr(canonical=True).row_norms()
    return _native.coo_row_norms(a.data, a.row, a.col, a.shape, assume_canonical=True)


def coo_col_norms(a: COOArray) -> mx.array:
    """Compute the dense-semantics L2 norm of each COO column."""
    a = _ensure_coo_array("coo_col_norms", a)
    if not a.has_canonical_format:
        return a.tocsc(canonical=True).col_norms()
    return _native.coo_col_norms(a.data, a.row, a.col, a.shape, assume_canonical=True)


def coo_column_norms(a: COOArray) -> mx.array:
    """Alias for :func:`coo_col_norms`."""
    return coo_col_norms(a)


def coo_diagonal(a: COOArray) -> mx.array:
    """Extract the summed diagonal of a COO matrix."""
    a = _ensure_coo_array("coo_diagonal", a)
    return _native.coo_diagonal(a.data, a.row, a.col, a.shape)


def coo_trace(a: COOArray) -> mx.array:
    """Compute the trace of a COO matrix."""
    a = _ensure_coo_array("coo_trace", a)
    return _native.coo_trace(a.data, a.row, a.col, a.shape)


def csc_row_sums(a: CSCArray) -> mx.array:
    """Reduce each row of a CSC matrix to the sum of its stored values."""
    a = _ensure_csc_array("csc_row_sums", a)
    return _native.csc_row_sums(a.data, a.indices, a.indptr, a.shape)


def csc_col_sums(a: CSCArray) -> mx.array:
    """Reduce each column of a CSC matrix to the sum of its stored values."""
    a = _ensure_csc_array("csc_col_sums", a)
    return _native.csc_col_sums(a.data, a.indices, a.indptr, a.shape)


def csc_column_sums(a: CSCArray) -> mx.array:
    """Alias for :func:`csc_col_sums`."""
    return csc_col_sums(a)


def csc_row_norms(a: CSCArray) -> mx.array:
    """Compute the L2 norm of each CSC row."""
    a = _ensure_csc_array("csc_row_norms", a)
    if not a.has_canonical_format:
        a = a.canonicalize()
    return _native.csc_row_norms(
        a.data, a.indices, a.indptr, a.shape, assume_canonical=True
    )


def csc_col_norms(a: CSCArray) -> mx.array:
    """Compute the L2 norm of each CSC column."""
    a = _ensure_csc_array("csc_col_norms", a)
    if not a.has_canonical_format:
        a = a.canonicalize()
    return _native.csc_col_norms(
        a.data, a.indices, a.indptr, a.shape, assume_canonical=True
    )


def csc_column_norms(a: CSCArray) -> mx.array:
    """Alias for :func:`csc_col_norms`."""
    return csc_col_norms(a)


def csc_diagonal(a: CSCArray) -> mx.array:
    """Extract the summed diagonal of a CSC matrix."""
    a = _ensure_csc_array("csc_diagonal", a)
    return _native.csc_diagonal(a.data, a.indices, a.indptr, a.shape)


def csc_trace(a: CSCArray) -> mx.array:
    """Compute the trace of a CSC matrix."""
    a = _ensure_csc_array("csc_trace", a)
    return _native.csc_trace(a.data, a.indices, a.indptr, a.shape)


def csc_matvec(a: CSCArray, x) -> mx.array:
    """Multiply a CSC sparse matrix by a dense vector."""
    a = _ensure_csc_array("csc_matvec", a)
    x = ensure_mx_array(x)
    validate_csc_matvec_inputs(a.data, a.indices, a.indptr, x, a.shape)
    return _native.csc_matvec(a.data, a.indices, a.indptr, x, a.shape)


def coo_matvec(a: COOArray, x) -> mx.array:
    """Multiply a COO sparse matrix by a dense vector."""
    a = _ensure_coo_array("coo_matvec", a)
    x = ensure_mx_array(x)
    validate_coo_matvec_inputs(a.data, a.row, a.col, x, a.shape)
    return _native.coo_matvec(a.data, a.row, a.col, x, a.shape)


def csc_matvec_transpose(a: CSCArray, x) -> mx.array:
    """Multiply the transpose of a CSC sparse matrix by a dense vector."""
    a = _ensure_csc_array("csc_matvec_transpose", a)
    x = ensure_mx_array(x)
    validate_csc_matvec_transpose_inputs(a.data, a.indices, a.indptr, x, a.shape)
    return _native.csc_matvec_transpose(a.data, a.indices, a.indptr, x, a.shape)


def coo_batched_matvec(a: COOArray, rhs) -> mx.array:
    """Multiply a COO sparse matrix by a batch of dense vectors."""
    a = _ensure_coo_array("coo_batched_matvec", a)
    rhs = ensure_mx_array(rhs)
    if rhs.ndim < 2:
        raise ValueError(
            f"coo_batched_matvec expects rank-2 or higher RHS, got {rhs.shape}."
        )
    if rhs.shape[-1] != a.shape[1]:
        raise ValueError(
            f"coo_batched_matvec RHS has vector dimension {rhs.shape[-1]}, "
            f"but sparse n_cols={a.shape[1]}."
        )
    if a.data.dtype != rhs.dtype:
        raise TypeError(
            "coo_batched_matvec requires sparse data and RHS to have the same dtype, "
            f"got {a.data.dtype} and {rhs.dtype}."
        )
    batch_shape = tuple(int(dim) for dim in rhs.shape[:-1])
    batch_size = _prod(batch_shape)
    rhs_flat = mx.reshape(rhs, (batch_size, a.shape[1]))
    out_flat = _native.coo_batched_matvec(a.data, a.row, a.col, rhs_flat, a.shape)
    return mx.reshape(out_flat, (*batch_shape, a.shape[0]))


def csc_batched_matvec(a: CSCArray, rhs) -> mx.array:
    """Multiply a CSC sparse matrix by a batch of dense vectors."""
    a = _ensure_csc_array("csc_batched_matvec", a)
    rhs = ensure_mx_array(rhs)
    if rhs.ndim < 2:
        raise ValueError(
            f"csc_batched_matvec expects rank-2 or higher RHS, got {rhs.shape}."
        )
    if rhs.shape[-1] != a.shape[1]:
        raise ValueError(
            f"csc_batched_matvec RHS has vector dimension {rhs.shape[-1]}, "
            f"but sparse n_cols={a.shape[1]}."
        )
    if a.data.dtype != rhs.dtype:
        raise TypeError(
            "csc_batched_matvec requires sparse data and RHS to have the same dtype, "
            f"got {a.data.dtype} and {rhs.dtype}."
        )
    batch_shape = tuple(int(dim) for dim in rhs.shape[:-1])
    batch_size = _prod(batch_shape)
    rhs_flat = mx.reshape(rhs, (batch_size, a.shape[1]))
    out_flat = _native.csc_batched_matvec(
        a.data, a.indices, a.indptr, rhs_flat, a.shape
    )
    return mx.reshape(out_flat, (*batch_shape, a.shape[0]))


def csr_matvec(a: CSRArray, x) -> mx.array:
    """Multiply a CSR sparse matrix by a dense vector.

    Computes ``y = A @ x`` where ``A`` is a :class:`~mlx_sparse.CSRArray` and
    ``x`` is a rank-1 dense array. The result is added to the MLX computation
    graph and not evaluated eagerly.

    On Apple Silicon, the Metal backend dispatches a scalar row kernel for
    short rows and a vector-reduction kernel for long rows. CPU and GPU paths
    support ``float32``, ``float16``, ``bfloat16``, and ``complex64`` values
    with ``int32`` or ``int64`` indices.

    Args:
        a: The sparse matrix, shape ``(n_rows, n_cols)``.
        x: Dense vector, shape ``(n_cols,)``. Converted to ``mx.array`` if
            needed. Must have the same dtype as ``a.data``.

    Returns:
        Dense vector of shape ``(n_rows,)`` with the same dtype as ``a.data``.

    Raises:
        TypeError: If ``a`` is not a :class:`~mlx_sparse.CSRArray`, or if the
            dtypes of ``a.data`` and ``x`` do not match.
        ValueError: If shape constraints are violated.

    Example::

        import mlx.core as mx
        import mlx_sparse as ms

        y = a @ x  # preferred via __matmul__
        y = ms.csr_matvec(a, x)  # explicit call
        mx.eval(y)
    """
    if not isinstance(a, CSRArray):
        raise TypeError(f"csr_matvec expects CSRArray, got {type(a).__name__}.")
    x = ensure_mx_array(x)
    validate_csr_matvec_inputs(a.data, a.indices, a.indptr, x, a.shape)
    return _native.csr_matvec(a.data, a.indices, a.indptr, x, a.shape)


def csr_batched_matvec(a: CSRArray, rhs) -> mx.array:
    """Multiply a CSR sparse matrix by a batch of dense vectors.

    Computes ``Y[b] = A @ X[b]`` for ``X`` with shape ``(..., n_cols)`` and
    returns shape ``(..., n_rows)``. The implementation uses native batched
    CPU/Metal kernels after flattening any leading batch dimensions.
    """
    if not isinstance(a, CSRArray):
        raise TypeError(f"csr_batched_matvec expects CSRArray, got {type(a).__name__}.")
    rhs = ensure_mx_array(rhs)
    if rhs.ndim < 2:
        raise ValueError(
            f"csr_batched_matvec expects rank-2 or higher RHS, got {rhs.shape}."
        )
    if rhs.shape[-1] != a.shape[1]:
        raise ValueError(
            f"csr_batched_matvec RHS has vector dimension {rhs.shape[-1]}, "
            f"but sparse n_cols={a.shape[1]}."
        )
    if a.data.dtype != rhs.dtype:
        raise TypeError(
            "csr_batched_matvec requires sparse data and RHS to have the same dtype, "
            f"got {a.data.dtype} and {rhs.dtype}."
        )

    batch_shape = tuple(int(dim) for dim in rhs.shape[:-1])
    batch_size = _prod(batch_shape)
    rhs_flat = mx.reshape(rhs, (batch_size, a.shape[1]))
    out_flat = _native.csr_batched_matvec(
        a.data, a.indices, a.indptr, rhs_flat, a.shape
    )
    return mx.reshape(out_flat, (*batch_shape, a.shape[0]))


def csr_matmat(a: CSRArray, rhs: CSRArray) -> CSRArray:
    """Multiply two CSR sparse matrices and return a canonical CSR matrix.

    Computes ``C = A @ B`` where both ``A`` and ``B`` are
    :class:`~mlx_sparse.CSRArray` instances. The output sparsity pattern is
    not known at graph-build time, so this operation performs a native C++
    structural assembly pass on the host (calling ``mx.eval`` on the input
    arrays internally) and returns a new :class:`~mlx_sparse.CSRArray` with
    canonical format.

    Because the output size is data-dependent, this operation is not
    representable as a fixed-shape MLX primitive. It is suitable for one-shot
    matrix products and matrix-power computations, but is not appropriate
    inside a JIT-compiled function.

    Args:
        a: Left-hand sparse matrix, shape ``(m, k)``.
        rhs: Right-hand sparse matrix, shape ``(k, n)``.

    Returns:
        A canonical :class:`~mlx_sparse.CSRArray` with shape ``(m, n)``,
        ``has_canonical_format=True``, and ``sorted_indices=True``.

    Raises:
        TypeError: If either argument is not a :class:`~mlx_sparse.CSRArray`.
        ValueError: If the inner dimensions do not match (``a.shape[1] != rhs.shape[0]``).

    Example::

        import mlx_sparse as ms

        # Compute the square of a sparse matrix
        C = A @ A  # dispatches csr_matmat when A is CSRArray
        C = ms.csr_matmat(A, A)  # explicit call

        # Chain sparse matrix products
        D = ms.csr_matmat(ms.csr_matmat(A, B), C)
    """
    if not isinstance(a, CSRArray):
        raise TypeError(f"csr_matmat expects CSRArray lhs, got {type(a).__name__}.")
    if not isinstance(rhs, CSRArray):
        raise TypeError(f"csr_matmat expects CSRArray rhs, got {type(rhs).__name__}.")
    if a.data.dtype != rhs.data.dtype:
        raise TypeError(
            "CSR sparse-sparse matmul requires matching value dtypes, "
            f"got {a.data.dtype} and {rhs.data.dtype}."
        )
    data, indices, indptr = _native.csr_matmat(a, rhs)
    return CSRArray(
        data=data,
        indices=indices,
        indptr=indptr,
        shape=(a.shape[0], rhs.shape[1]),
        sorted_indices=True,
        has_canonical_format=True,
    )


def coo_matmat(a: COOArray, rhs: COOArray) -> COOArray:
    """Multiply two COO sparse matrices and return a canonical COO matrix.

    The native implementation groups both operands by coordinate rows, performs
    a symbolic row pass to size the result, then fills sorted output coordinates
    without routing through CSR.
    """
    a = _ensure_coo_array("coo_matmat", a)
    if not isinstance(rhs, COOArray):
        raise TypeError(f"coo_matmat expects COOArray rhs, got {type(rhs).__name__}.")
    if a.shape[1] != rhs.shape[0]:
        raise ValueError(
            f"COO sparse-sparse matmul dimension mismatch: {a.shape} @ {rhs.shape}."
        )
    if a.data.dtype != rhs.data.dtype:
        raise TypeError(
            "COO sparse-sparse matmul requires matching value dtypes, "
            f"got {a.data.dtype} and {rhs.data.dtype}."
        )
    data, row, col = _native.coo_matmat(a, rhs)
    return COOArray(
        data=data,
        row=row,
        col=col,
        shape=(a.shape[0], rhs.shape[1]),
        has_canonical_format=True,
    )


def csc_matmat(a: CSCArray, rhs: CSCArray) -> CSCArray:
    """Multiply two CSC sparse matrices and return a canonical CSC matrix.

    The native implementation traverses right-hand columns and left-hand
    compressed columns directly, producing sorted row indices per output column.
    It does not convert to CSR internally.
    """
    a = _ensure_csc_array("csc_matmat", a)
    if not isinstance(rhs, CSCArray):
        raise TypeError(f"csc_matmat expects CSCArray rhs, got {type(rhs).__name__}.")
    if a.shape[1] != rhs.shape[0]:
        raise ValueError(
            f"CSC sparse-sparse matmul dimension mismatch: {a.shape} @ {rhs.shape}."
        )
    if a.data.dtype != rhs.data.dtype:
        raise TypeError(
            "CSC sparse-sparse matmul requires matching value dtypes, "
            f"got {a.data.dtype} and {rhs.data.dtype}."
        )
    data, indices, indptr = _native.csc_matmat(a, rhs)
    return CSCArray(
        data=data,
        indices=indices,
        indptr=indptr,
        shape=(a.shape[0], rhs.shape[1]),
        sorted_indices=True,
        has_canonical_format=True,
    )


def _csr_matmul_rank2(a: CSRArray, rhs: mx.array) -> mx.array:
    validate_csr_matmul_inputs(a.data, a.indices, a.indptr, rhs, a.shape)
    return _native.csr_matmul(a.data, a.indices, a.indptr, rhs, a.shape)


def _coo_matmul_rank2(a: COOArray, rhs: mx.array) -> mx.array:
    validate_coo_matmul_inputs(a.data, a.row, a.col, rhs, a.shape)
    return _native.coo_matmul(a.data, a.row, a.col, rhs, a.shape)


def _csc_matmul_rank2(a: CSCArray, rhs: mx.array) -> mx.array:
    validate_csc_matmul_inputs(a.data, a.indices, a.indptr, rhs, a.shape)
    return _native.csc_matmul(a.data, a.indices, a.indptr, rhs, a.shape)


def _csr_matmul_batched(a: CSRArray, rhs: mx.array) -> mx.array:
    if a.data.dtype != rhs.dtype:
        raise TypeError(
            "csr_matmul requires sparse data and RHS to have the same dtype, "
            f"got {a.data.dtype} and {rhs.dtype}."
        )
    batch_shape = tuple(int(dim) for dim in rhs.shape[:-2])
    rhs_cols = int(rhs.shape[-1])
    batch_size = _prod(batch_shape)
    rhs_flat = mx.reshape(rhs, (batch_size, a.shape[1], rhs_cols))
    out_flat = _native.csr_batched_matmul(
        a.data, a.indices, a.indptr, rhs_flat, a.shape
    )
    return mx.reshape(out_flat, (*batch_shape, a.shape[0], rhs_cols))


def _coo_matmul_batched(a: COOArray, rhs: mx.array) -> mx.array:
    if a.data.dtype != rhs.dtype:
        raise TypeError(
            "coo_matmul requires sparse data and RHS to have the same dtype, "
            f"got {a.data.dtype} and {rhs.dtype}."
        )
    batch_shape = tuple(int(dim) for dim in rhs.shape[:-2])
    rhs_cols = int(rhs.shape[-1])
    batch_size = _prod(batch_shape)
    rhs_flat = mx.reshape(rhs, (batch_size, a.shape[1], rhs_cols))
    out_flat = _native.coo_batched_matmul(a.data, a.row, a.col, rhs_flat, a.shape)
    return mx.reshape(out_flat, (*batch_shape, a.shape[0], rhs_cols))


def _csc_matmul_batched(a: CSCArray, rhs: mx.array) -> mx.array:
    if a.data.dtype != rhs.dtype:
        raise TypeError(
            "csc_matmul requires sparse data and RHS to have the same dtype, "
            f"got {a.data.dtype} and {rhs.dtype}."
        )
    batch_shape = tuple(int(dim) for dim in rhs.shape[:-2])
    rhs_cols = int(rhs.shape[-1])
    batch_size = _prod(batch_shape)
    rhs_flat = mx.reshape(rhs, (batch_size, a.shape[1], rhs_cols))
    out_flat = _native.csc_batched_matmul(
        a.data, a.indices, a.indptr, rhs_flat, a.shape
    )
    return mx.reshape(out_flat, (*batch_shape, a.shape[0], rhs_cols))


def coo_batched_matmul(a: COOArray, rhs) -> mx.array:
    """Multiply a COO sparse matrix by a batch of dense matrices."""
    a = _ensure_coo_array("coo_batched_matmul", a)
    rhs = ensure_mx_array(rhs)
    if rhs.ndim < 3:
        raise ValueError(
            f"coo_batched_matmul expects rank-3 or higher RHS, got {rhs.shape}."
        )
    if rhs.shape[-2] != a.shape[1]:
        raise ValueError(
            f"coo_batched_matmul RHS has sparse dimension {rhs.shape[-2]}, "
            f"but sparse n_cols={a.shape[1]}."
        )
    return _coo_matmul_batched(a, rhs)


def csc_batched_matmul(a: CSCArray, rhs) -> mx.array:
    """Multiply a CSC sparse matrix by a batch of dense matrices."""
    a = _ensure_csc_array("csc_batched_matmul", a)
    rhs = ensure_mx_array(rhs)
    if rhs.ndim < 3:
        raise ValueError(
            f"csc_batched_matmul expects rank-3 or higher RHS, got {rhs.shape}."
        )
    if rhs.shape[-2] != a.shape[1]:
        raise ValueError(
            f"csc_batched_matmul RHS has sparse dimension {rhs.shape[-2]}, "
            f"but sparse n_cols={a.shape[1]}."
        )
    return _csc_matmul_batched(a, rhs)


def coo_matmul(a: COOArray, rhs) -> mx.array:
    """Multiply a COO sparse matrix by a dense matrix or batched matrices."""
    a = _ensure_coo_array("coo_matmul", a)
    rhs = ensure_mx_array(rhs)
    if rhs.ndim == 2:
        return _coo_matmul_rank2(a, rhs)
    if rhs.ndim < 2:
        raise ValueError(f"coo_matmul expects rank-2 or higher RHS, got {rhs.shape}.")
    if rhs.shape[-2] != a.shape[1]:
        raise ValueError(
            f"coo_matmul RHS has sparse dimension {rhs.shape[-2]}, "
            f"but sparse n_cols={a.shape[1]}."
        )
    return _coo_matmul_batched(a, rhs)


def csc_matmul(a: CSCArray, rhs) -> mx.array:
    """Multiply a CSC sparse matrix by a dense matrix or batched matrices."""
    a = _ensure_csc_array("csc_matmul", a)
    rhs = ensure_mx_array(rhs)
    if rhs.ndim == 2:
        return _csc_matmul_rank2(a, rhs)
    if rhs.ndim < 2:
        raise ValueError(f"csc_matmul expects rank-2 or higher RHS, got {rhs.shape}.")
    if rhs.shape[-2] != a.shape[1]:
        raise ValueError(
            f"csc_matmul RHS has sparse dimension {rhs.shape[-2]}, "
            f"but sparse n_cols={a.shape[1]}."
        )
    return _csc_matmul_batched(a, rhs)


def csr_batched_matmul(a: CSRArray, rhs) -> mx.array:
    """Multiply a CSR sparse matrix by a batch of dense matrices.

    ``rhs`` must have shape ``(..., n_cols, k)`` and the result has shape
    ``(..., n_rows, k)``. For rank-2 dense matrices, use :func:`csr_matmul`.
    """
    if not isinstance(a, CSRArray):
        raise TypeError(f"csr_batched_matmul expects CSRArray, got {type(a).__name__}.")
    rhs = ensure_mx_array(rhs)
    if rhs.ndim < 3:
        raise ValueError(
            f"csr_batched_matmul expects rank-3 or higher RHS, got {rhs.shape}."
        )
    if rhs.shape[-2] != a.shape[1]:
        raise ValueError(
            f"csr_batched_matmul RHS has sparse dimension {rhs.shape[-2]}, "
            f"but sparse n_cols={a.shape[1]}."
        )
    return _csr_matmul_batched(a, rhs)


def csr_matmul(a: CSRArray, rhs) -> mx.array:
    """Multiply a CSR sparse matrix by a dense matrix.

    Computes ``Y = A @ B`` where ``A`` is a :class:`~mlx_sparse.CSRArray` and
    ``B`` is a rank-2 or batched dense array. The result is added to the MLX
    computation graph and not evaluated eagerly.

    On Apple Silicon, the Metal backend dispatches scalar output-element
    kernels for short rows and vector-reduction kernels for long rows. CPU and
    GPU paths support ``float32``, ``float16``, ``bfloat16``, and ``complex64``
    values with ``int32`` or ``int64`` indices.

    Args:
        a: The sparse matrix, shape ``(n_rows, n_cols)``.
        rhs: Dense matrix, shape ``(n_cols, k)``, or batched dense matrix with
            sparse dimension at ``rhs.shape[-2]``. Converted to ``mx.array`` if
            needed. Must have the same dtype as ``a.data``.

    Returns:
        Dense matrix or batched dense matrix with sparse dimension replaced by
        ``n_rows`` and the same dtype as ``a.data``.

    Raises:
        TypeError: If ``a`` is not a :class:`~mlx_sparse.CSRArray`, or if
            dtype constraints are violated.
        ValueError: If shape constraints are violated.

    Example::

        import mlx.core as mx
        import mlx_sparse as ms

        Y = a @ B  # preferred via __matmul__
        Y = ms.csr_matmul(a, B)  # explicit call
        mx.eval(Y)
    """
    if not isinstance(a, CSRArray):
        raise TypeError(f"csr_matmul expects CSRArray, got {type(a).__name__}.")
    rhs = ensure_mx_array(rhs)
    if rhs.ndim == 2:
        return _csr_matmul_rank2(a, rhs)
    if rhs.ndim < 2:
        raise ValueError(f"csr_matmul expects rank-2 or higher RHS, got {rhs.shape}.")
    if rhs.shape[-2] != a.shape[1]:
        raise ValueError(
            f"csr_matmul RHS has sparse dimension {rhs.shape[-2]}, "
            f"but sparse n_cols={a.shape[1]}."
        )

    return _csr_matmul_batched(a, rhs)
