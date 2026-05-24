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
from mlx_sparse._validation import (
    ensure_mx_array,
    validate_coo_matmul_inputs,
    validate_coo_matvec_inputs,
    validate_csc_matmul_inputs,
    validate_csc_matvec_inputs,
    validate_csc_matvec_transpose_inputs,
    validate_csr_matmul_inputs,
    validate_csr_matvec_inputs,
    validate_csr_metadata,
)


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
