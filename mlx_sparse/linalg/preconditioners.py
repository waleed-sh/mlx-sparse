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

"""Native-backed sparse solver preconditioners.

The Python objects in this module are containers and dispatch helpers.
Application and Krylov iteration dispatch to native mlx-sparse primitives rather
than Python solver loops. Constructors may use existing sparse native kernels
and MLX scalar array expressions to build immutable setup data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import mlx.core as mx
import numpy as np

import mlx_sparse._native as _native
from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._host import to_numpy
from mlx_sparse._validation import ensure_mx_array, normalize_shape
from mlx_sparse.linalg._interface import LinearOperator


@runtime_checkable
class Preconditioner(Protocol):
    """Inverse-apply preconditioner protocol."""

    shape: tuple[int, int]
    dtype: object
    kind: str

    def solve(self, x) -> mx.array:
        """Apply the preconditioner solve to ``x``."""

    def __call__(self, x) -> mx.array:
        """Alias for :meth:`solve`."""


def _host_bool(value: mx.array) -> bool:
    mx.eval(value)
    return bool(np.asarray(to_numpy(value)).item())


def _as_square_shape(A_or_shape) -> tuple[int, int]:
    if isinstance(A_or_shape, int):
        shape = (int(A_or_shape), int(A_or_shape))
    elif hasattr(A_or_shape, "shape"):
        shape = A_or_shape.shape
    else:
        shape = A_or_shape
    shape = normalize_shape(shape)
    if shape[0] != shape[1]:
        raise ValueError(f"preconditioners require a square shape, got {shape}.")
    return shape


def _canonical_csr(A) -> CSRArray:
    if isinstance(A, CSRArray):
        return A.canonicalize()
    if isinstance(A, COOArray):
        return A.tocsr(canonical=True)
    if isinstance(A, CSCArray):
        return A.tocsr(canonical=True)
    if isinstance(A, LinearOperator) and A._sparse_array is not None:
        return _canonical_csr(A._sparse_array)
    raise TypeError(
        "jacobi expects CSRArray, COOArray, CSCArray, or a sparse-backed "
        "LinearOperator."
    )


def _float32_vector(name: str, value) -> mx.array:
    array = ensure_mx_array(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape={array.shape}.")
    if array.dtype != mx.float32:
        if array.dtype not in {mx.float16, mx.bfloat16, mx.float32}:
            raise TypeError(f"{name} must have a real floating dtype.")
        array = array.astype(mx.float32)
    return array


def _check_rhs(rhs, shape: tuple[int, int]) -> mx.array:
    array = ensure_mx_array(rhs)
    if array.ndim not in (1, 2):
        raise ValueError(
            f"right-hand side must be rank-1 or rank-2, got {array.shape}."
        )
    if array.shape[0] != shape[0]:
        raise ValueError(
            f"right-hand side has leading dimension {array.shape[0]}, "
            f"expected {shape[0]}."
        )
    if array.ndim == 2 and array.shape[1] <= 0:
        raise ValueError("right-hand side must include at least one column.")
    if array.dtype != mx.float32:
        if array.dtype not in {mx.float16, mx.bfloat16, mx.float32}:
            raise TypeError("right-hand side must have a real floating dtype.")
        array = array.astype(mx.float32)
    return array


@dataclass(frozen=True, slots=True)
class IdentityPreconditioner:
    """No-op preconditioner container."""

    shape: tuple[int, int]
    dtype: object = mx.float32
    kind: str = "identity"
    is_symmetric: bool = True
    is_positive_definite: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _as_square_shape(self.shape))

    @property
    def nnz(self) -> int:
        return self.shape[0]

    def solve(self, x) -> mx.array:
        return _check_rhs(x, self.shape)

    def __call__(self, x) -> mx.array:
        return self.solve(x)


@dataclass(frozen=True, slots=True)
class DiagonalPreconditioner:
    """Native-applied diagonal inverse preconditioner."""

    inverse_diagonal: mx.array
    shape: tuple[int, int]
    kind: str = "diagonal"
    is_symmetric: bool = True
    is_positive_definite: bool = False

    def __post_init__(self) -> None:
        shape = _as_square_shape(self.shape)
        inv_diag = _float32_vector("inverse_diagonal", self.inverse_diagonal)
        if inv_diag.shape[0] != shape[0]:
            raise ValueError(
                f"inverse_diagonal has length {inv_diag.shape[0]}, "
                f"expected {shape[0]}."
            )
        if not _host_bool(mx.all(mx.isfinite(inv_diag))):
            raise ValueError("inverse_diagonal must contain only finite values.")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "inverse_diagonal", inv_diag)

    @property
    def dtype(self):
        return self.inverse_diagonal.dtype

    @property
    def nnz(self) -> int:
        return int(self.inverse_diagonal.shape[0])

    def solve(self, x) -> mx.array:
        rhs = _check_rhs(x, self.shape)
        return _native.diagonal_preconditioner_apply(self.inverse_diagonal, rhs)

    def matvec(self, x) -> mx.array:
        return self.solve(x)

    def __call__(self, x) -> mx.array:
        return self.solve(x)


@dataclass(frozen=True, slots=True)
class JacobiPreconditioner(DiagonalPreconditioner):
    """Jacobi preconditioner with native PCG solver support."""

    kind: str = "jacobi"


def identity(A_or_shape, *, dtype=None) -> IdentityPreconditioner:
    """Create an identity preconditioner."""

    shape = _as_square_shape(A_or_shape)
    return IdentityPreconditioner(
        shape=shape, dtype=mx.float32 if dtype is None else dtype
    )


def diagonal(
    inv_diag_or_diag,
    *,
    inverse: bool = False,
    shape=None,
    dtype=None,
    zero_atol: float = 0.0,
) -> DiagonalPreconditioner:
    """Create a native-applied diagonal inverse preconditioner."""

    values = _float32_vector("diagonal", inv_diag_or_diag)
    if dtype is not None and dtype != mx.float32:
        raise TypeError("diagonal preconditioners currently use float32 values.")
    pc_shape = (
        _as_square_shape((values.shape[0], values.shape[0]))
        if shape is None
        else _as_square_shape(shape)
    )
    if values.shape[0] != pc_shape[0]:
        raise ValueError(
            f"diagonal has length {values.shape[0]}, expected {pc_shape[0]}."
        )
    if not _host_bool(mx.all(mx.isfinite(values))):
        raise ValueError("diagonal must contain only finite values.")
    if inverse:
        inv_diag = values
    else:
        atol = float(zero_atol)
        if atol < 0.0:
            raise ValueError("zero_atol must be non-negative.")
        if _host_bool(mx.any(mx.abs(values) <= atol)):
            raise ValueError("diagonal contains zero or near-zero entries.")
        inv_diag = 1.0 / values
    return DiagonalPreconditioner(inv_diag, pc_shape)


def jacobi(
    A,
    *,
    omega: float = 1.0,
    shift: float = 0.0,
    zero_policy: str = "raise",
    zero_atol: float = 0.0,
) -> JacobiPreconditioner:
    """Create a Jacobi preconditioner from a sparse matrix diagonal."""

    if zero_policy not in {"raise", "unit"}:
        raise ValueError("zero_policy must be 'raise' or 'unit'.")
    csr = _canonical_csr(A)
    if csr.shape[0] != csr.shape[1]:
        raise ValueError(f"jacobi requires a square matrix, got {csr.shape}.")
    diag = _float32_vector("diagonal", csr.diagonal())
    shifted = diag + mx.array(float(shift), dtype=mx.float32)
    if not _host_bool(mx.all(mx.isfinite(shifted))):
        raise ValueError("shifted diagonal must contain only finite values.")
    atol = float(zero_atol)
    if atol < 0.0:
        raise ValueError("zero_atol must be non-negative.")
    near_zero = mx.abs(shifted) <= atol
    if _host_bool(mx.any(near_zero)):
        if zero_policy == "raise":
            raise ValueError(
                "Jacobi shifted diagonal contains zero or near-zero entries."
            )
        shifted = mx.where(near_zero, mx.ones_like(shifted), shifted)
    inv_diag = mx.array(float(omega), dtype=mx.float32) / shifted
    return JacobiPreconditioner(inv_diag, csr.shape)


def aspreconditioner(M, A=None, *, assume_inverse: bool = True) -> Preconditioner:
    """Normalize supported objects to a native-backed preconditioner."""

    if M is None:
        if A is None:
            raise ValueError("A is required when M is None.")
        return identity(A)
    if isinstance(M, (IdentityPreconditioner, DiagonalPreconditioner)):
        if A is not None and M.shape != _as_square_shape(A):
            raise ValueError(f"preconditioner shape {M.shape} does not match A.shape.")
        return M
    if isinstance(M, (CSRArray, COOArray, CSCArray)):
        raise TypeError(
            "sparse matrices are not inverse-apply preconditioners; use "
            "preconditioners.jacobi(A) or preconditioners.diagonal(...)."
        )
    _ = assume_inverse
    raise TypeError(
        "only native-backed identity, diagonal, and Jacobi preconditioners are "
        "supported by the current solver integration."
    )


__all__ = [
    "DiagonalPreconditioner",
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "Preconditioner",
    "aspreconditioner",
    "diagonal",
    "identity",
    "jacobi",
]
