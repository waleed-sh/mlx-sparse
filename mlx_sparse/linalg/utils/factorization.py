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

"""Sparse direct-solver helper routines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

import mlx_sparse._native as _native
from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._host import to_numpy
from mlx_sparse.linalg.utils.arrays import (
    REAL_FLOAT_DTYPES,
    ensure_array,
    ensure_float32_csr,
    ensure_float32_sparse,
    host_norm,
)
from mlx_sparse.linalg.utils.sparse import as_sparse as base_as_sparse
from mlx_sparse.linalg.utils.sparse import canonical_csr

FACTORIZED_METHODS = {"auto", "lu", "cholesky", "ldlt", "qr", "cholesky_ata"}
ACCELERATE_ONLY_METHODS = {"ldlt", "qr", "cholesky_ata"}
ACCELERATE_SPSOLVE_RESIDUAL_RTOL = 1e-3


@dataclass(frozen=True, slots=True)
class NativeFactorizedSolve:
    """Adapter exposing a direct factorization through a reusable solve object."""

    factorization: object
    rhs_size: int

    def solve(self, b) -> mx.array:
        """Solve one or more RHS columns with the wrapped factorization."""

        return solve_columns(self.factorization, b, rhs_size=self.rhs_size)


def as_csr(A) -> CSRArray:
    """Return sparse factorization input as canonical CSR."""

    return canonical_csr(
        A,
        context="sparse factorization",
        dense_guidance="Dense MLX arrays belong in mlx.linalg, not mlx_sparse.linalg.",
    )


def as_sparse(A) -> CSRArray | CSCArray | COOArray:
    """Return sparse factorization input without changing its storage format."""

    return base_as_sparse(
        A,
        context="sparse factorization",
        dense_guidance="Dense MLX arrays belong in mlx.linalg, not mlx_sparse.linalg.",
    )


def float32_csr(A: CSRArray) -> CSRArray:
    """Return sparse direct-factorization CSR input with float32 values."""

    return ensure_float32_csr(A, context="sparse direct factorizations")


def float32_sparse(A: CSRArray | CSCArray | COOArray) -> CSRArray | CSCArray | COOArray:
    """Return sparse direct-factorization input with float32 values."""

    return ensure_float32_sparse(A, context="sparse direct factorizations")


def triangular_solve(
    factor: CSRArray,
    b,
    *,
    lower: bool,
    unit_diagonal: bool,
    diagonal_positions: mx.array | None = None,
    level_schedule: tuple[mx.array, mx.array] | None = None,
):
    """Validate and apply a native CSR triangular solve."""

    rhs = ensure_factor_rhs(b, leading_dim=factor.shape[0])
    return _native.csr_triangular_solve(
        factor.data,
        factor.indices,
        factor.indptr,
        rhs,
        factor.shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
        diagonal_positions=diagonal_positions,
        level_schedule=level_schedule,
    )


def normalize_factorized_method(method: str) -> str:
    """Normalize public aliases for :func:`mlx_sparse.linalg.factorized`."""

    normalized = method.lower().replace("-", "_")
    aliases = {
        "chol": "cholesky",
        "spd": "cholesky",
        "posdef": "cholesky",
        "positive_definite": "cholesky",
        "least_squares": "qr",
        "normal_equations": "cholesky_ata",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in FACTORIZED_METHODS:
        allowed = ", ".join(sorted(FACTORIZED_METHODS))
        raise ValueError(f"factorized method must be one of {allowed}.")
    return normalized


def auto_factorized_method(A: CSRArray | CSCArray | COOArray) -> str:
    """Choose the default direct solve method from the sparse matrix shape."""

    if A.shape[0] == A.shape[1]:
        return "lu"
    return "qr"


def accelerate_method_available(method: str) -> bool:
    """Return whether the current native extension can use Accelerate."""

    if not _native.accelerate_solvers_available():
        return False
    if method == "lu":
        return _native.accelerate_lu_solvers_available()
    return True


def should_use_accelerate(A: CSRArray | CSCArray | COOArray, method: str) -> bool:
    """Return whether Accelerate is available and appropriate for ``A``."""

    return A.data.dtype in REAL_FLOAT_DTYPES and accelerate_method_available(method)


def accelerate_factorize_sparse(A: CSRArray | CSCArray | COOArray, method: str):
    """Create an Accelerate factorization object for float32 sparse input."""

    sparse = float32_sparse(A)
    if isinstance(sparse, CSRArray):
        return sparse, _native.accelerate_factorize_csr_float32(
            sparse.data, sparse.indices, sparse.indptr, sparse.shape, method
        )
    if isinstance(sparse, CSCArray):
        return sparse, _native.accelerate_factorize_csc_float32(
            sparse.data, sparse.indices, sparse.indptr, sparse.shape, method
        )
    return sparse, _native.accelerate_factorize_coo_float32(
        sparse.data, sparse.row, sparse.col, sparse.shape, method
    )


def ensure_factor_rhs(b, *, leading_dim: int) -> mx.array:
    """Validate a vector or matrix RHS for direct sparse solves."""

    rhs = ensure_array(b, dtype=mx.float32)
    if rhs.ndim not in (1, 2):
        raise ValueError(f"right-hand side must be rank-1 or rank-2, got {rhs.shape}.")
    if rhs.shape[0] != leading_dim:
        if rhs.ndim == 1:
            raise ValueError(
                f"right-hand side has incompatible shape {rhs.shape}; "
                f"expected ({leading_dim},)."
            )
        raise ValueError(
            f"right-hand side has incompatible shape {rhs.shape}; "
            f"expected first dimension {leading_dim}."
        )
    if rhs.ndim == 2 and rhs.shape[1] <= 0:
        raise ValueError("right-hand side must include at least one column.")
    return rhs


def solve_columns(solver, b, *, rhs_size: int) -> mx.array:
    """Validate one or more RHS columns and delegate to ``solver.solve``."""

    rhs = ensure_factor_rhs(b, leading_dim=rhs_size)
    return solver.solve(rhs)


def check_accelerate_direct_residual(
    A: CSRArray | CSCArray | COOArray,
    x: mx.array,
    rhs: mx.array,
) -> None:
    """Reject non-finite or high-residual Accelerate direct-solve output."""

    x_np = np.asarray(to_numpy(x), dtype=np.float64)
    if not np.all(np.isfinite(x_np)):
        raise RuntimeError(
            "Accelerate sparse direct solve produced non-finite values; "
            "the matrix may be singular or ill-conditioned."
        )

    residual = A @ x - rhs
    residual_np = np.asarray(to_numpy(residual), dtype=np.float64)
    rhs_np = np.asarray(to_numpy(rhs), dtype=np.float64)
    scale = max(host_norm(rhs_np), 1.0)
    relative_residual = host_norm(residual_np) / scale
    if (
        not np.isfinite(relative_residual)
        or relative_residual > ACCELERATE_SPSOLVE_RESIDUAL_RTOL
    ):
        raise RuntimeError(
            "Accelerate sparse direct solve residual is too large; "
            "the matrix may be singular or ill-conditioned."
        )


def accelerate_singularity_probe(n: int) -> mx.array:
    """Return a deterministic alternating-sign probe RHS for singularity checks."""

    values = np.ones((n,), dtype=np.float32)
    values[1::2] = -1.0
    return mx.array(values, dtype=mx.float32)


def solve_accelerate_spsolve_checked(
    A: CSRArray | CSCArray | COOArray,
    solver,
    rhs: mx.array,
    *,
    singularity_checker: Callable[[CSRArray | CSCArray | COOArray], object],
) -> mx.array:
    """Solve with Accelerate and validate user/probe residuals.

    ``singularity_checker`` is called only when the probe residual fails. The
    native caller passes its fallback factorization routine so this utility can
    preserve existing singular-matrix error behavior without importing the
    public factorization module and creating a circular dependency.
    """

    probe = accelerate_singularity_probe(A.shape[0])
    if rhs.ndim == 1:
        combined_rhs = mx.concatenate([rhs[:, None], probe[:, None]], axis=1)
        combined_x = solver.solve(combined_rhs)
        user_x = combined_x[:, 0]
        probe_x = combined_x[:, 1]
    elif rhs.ndim == 2:
        combined_rhs = mx.concatenate([rhs, probe[:, None]], axis=1)
        combined_x = solver.solve(combined_rhs)
        user_x = combined_x[:, : rhs.shape[1]]
        probe_x = combined_x[:, rhs.shape[1]]
    else:
        raise ValueError(f"right-hand side must be rank-1 or rank-2, got {rhs.shape}.")

    A_float32 = ensure_float32_sparse(A, context="sparse direct factorizations")
    check_accelerate_direct_residual(A_float32, user_x, rhs)
    try:
        check_accelerate_direct_residual(A_float32, probe_x, probe)
    except RuntimeError as exc:
        try:
            singularity_checker(A_float32)
        except RuntimeError:
            raise exc
    return user_x
