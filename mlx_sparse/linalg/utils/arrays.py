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

"""Array conversion, dtype promotion, and host scalar helpers."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._host import to_numpy
from mlx_sparse._validation import ensure_mx_array

REAL_FLOAT_DTYPES = {mx.float16, mx.bfloat16, mx.float32}


def ensure_array(x, *, dtype=None) -> mx.array:
    """Return ``x`` as an MLX array, optionally casting to ``dtype``.

    Args:
        x: Any object accepted by :func:`mlx.core.array`, or an existing
            ``mlx.core.array``.
        dtype: Optional target dtype. Existing arrays are returned unchanged
            when their dtype already matches.

    Returns:
        ``x`` as an ``mlx.core.array``.
    """

    if isinstance(x, mx.array):
        if dtype is not None and x.dtype != dtype:
            return x.astype(dtype)
        return x
    if dtype is None:
        return mx.array(x)
    return mx.array(x, dtype=dtype)


def host_bool(value: mx.array) -> bool:
    """Synchronize a scalar boolean MLX array and return it as ``bool``."""

    mx.eval(value)
    return bool(np.asarray(to_numpy(value)).item())


def host_norm(values) -> float:
    """Compute a stable host-side Euclidean norm for diagnostic checks."""

    array = np.asarray(values, dtype=np.float64).ravel()
    return float(np.sqrt(np.sum(array * array)))


def finite_scalar(name: str, value) -> float:
    """Convert ``value`` to ``float`` and reject non-finite scalars."""

    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def ensure_float32_array(x, *, context: str) -> mx.array:
    """Return a real dense array promoted to ``float32``.

    Args:
        x: Dense array-like input.
        context: Human-readable caller name used in error messages.

    Returns:
        A rank-preserving ``float32`` MLX array.

    Raises:
        TypeError: If ``x`` is not a real floating array.
    """

    array = ensure_mx_array(x)
    if array.dtype == mx.float32:
        return array
    if array.dtype in {mx.float16, mx.bfloat16}:
        return array.astype(mx.float32)
    raise TypeError(f"{context} currently require real float data.")


def ensure_float32_vector(
    name: str,
    value,
    *,
    require_finite: bool = False,
) -> mx.array:
    """Return a rank-1 real vector promoted to ``float32``.

    Args:
        name: Name used in validation errors.
        value: Vector-like input.
        require_finite: When ``True``, synchronizes and rejects NaN/Inf values.

    Returns:
        A rank-1 ``float32`` MLX array.
    """

    array = ensure_mx_array(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape={array.shape}.")
    if array.dtype != mx.float32:
        if array.dtype not in REAL_FLOAT_DTYPES:
            raise TypeError(f"{name} must have a real floating dtype.")
        array = array.astype(mx.float32)
    if require_finite and not host_bool(mx.all(mx.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def ensure_float32_csr(A: CSRArray, *, context: str) -> CSRArray:
    """Return a CSR matrix with real values promoted to ``float32``."""

    if A.data.dtype == mx.float32:
        return A
    if A.data.dtype in {mx.float16, mx.bfloat16}:
        return CSRArray(
            data=A.data.astype(mx.float32),
            indices=A.indices,
            indptr=A.indptr,
            shape=A.shape,
            sorted_indices=A.sorted_indices,
            has_canonical_format=A.has_canonical_format,
        )
    raise TypeError(f"{context} currently require real float data.")


def ensure_float32_sparse(
    A: CSRArray | CSCArray | COOArray,
    *,
    context: str,
) -> CSRArray | CSCArray | COOArray:
    """Return a sparse matrix with real values promoted to ``float32``."""

    if A.data.dtype == mx.float32:
        return A
    if A.data.dtype not in REAL_FLOAT_DTYPES:
        raise TypeError(f"{context} currently require real float data.")
    if isinstance(A, CSRArray):
        return CSRArray(
            data=A.data.astype(mx.float32),
            indices=A.indices,
            indptr=A.indptr,
            shape=A.shape,
            sorted_indices=A.sorted_indices,
            has_canonical_format=A.has_canonical_format,
        )
    if isinstance(A, CSCArray):
        return CSCArray(
            data=A.data.astype(mx.float32),
            indices=A.indices,
            indptr=A.indptr,
            shape=A.shape,
            sorted_indices=A.sorted_indices,
            has_canonical_format=A.has_canonical_format,
        )
    return COOArray(
        data=A.data.astype(mx.float32),
        row=A.row,
        col=A.col,
        shape=A.shape,
        has_canonical_format=A.has_canonical_format,
    )


def ensure_rank1_or_rank2_rhs(
    rhs,
    *,
    leading_dim: int,
    dtype=mx.float32,
    require_finite: bool = False,
) -> mx.array:
    """Validate a vector or matrix right-hand side.

    Args:
        rhs: Array-like right-hand side.
        leading_dim: Required first dimension.
        dtype: Target dtype for the returned array.
        require_finite: When ``True``, synchronizes and rejects NaN/Inf values.

    Returns:
        A rank-1 or rank-2 MLX array with the requested dtype.
    """

    array = ensure_mx_array(rhs)
    if array.ndim not in (1, 2):
        raise ValueError(
            f"right-hand side must be rank-1 or rank-2, got {array.shape}."
        )
    if array.shape[0] != leading_dim:
        raise ValueError(
            f"right-hand side has leading dimension {array.shape[0]}, "
            f"expected {leading_dim}."
        )
    if array.ndim == 2 and array.shape[1] <= 0:
        raise ValueError("right-hand side must include at least one column.")
    if dtype is not None and array.dtype != dtype:
        if dtype == mx.float32 and array.dtype not in REAL_FLOAT_DTYPES:
            raise TypeError("right-hand side must have a real floating dtype.")
        array = array.astype(dtype)
    if require_finite and not host_bool(mx.all(mx.isfinite(array))):
        raise ValueError("right-hand side must contain only finite values.")
    return array
