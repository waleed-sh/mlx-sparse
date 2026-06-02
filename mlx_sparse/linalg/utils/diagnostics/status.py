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

"""Status-code normalization for iterative sparse solvers."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_sparse._host import to_numpy
from mlx_sparse.linalg.utils.diagnostics.info import SolverInfo


def solver_info_to_int(info) -> int:
    """Synchronize a scalar native ``info`` array and return it as ``int``."""

    mx.eval(info)
    return int(np.asarray(to_numpy(info)).item())


def scalar_float(value) -> float:
    """Synchronize a scalar MLX/Python value and return it as ``float``."""

    if isinstance(value, mx.array):
        mx.eval(value)
        return float(np.asarray(to_numpy(value)).item())
    return float(value)


def scalar_int(value) -> int:
    """Synchronize a scalar MLX/Python value and return it as ``int``."""

    if isinstance(value, mx.array):
        return solver_info_to_int(value)
    return int(value)


def status_reason(status: int) -> tuple[str, str | None, str]:
    """Map native solver status codes to stable diagnostic text."""

    if status == 0:
        return "converged", None, "converged to the requested true-residual tolerance"
    if status > 0:
        return (
            "iteration_limit",
            None,
            "iteration budget exhausted before reaching the requested tolerance",
        )
    if status == -1:
        return (
            "numerical_breakdown",
            "breakdown",
            "solver encountered a near-zero denominator or Krylov breakdown",
        )
    if status == -2:
        return (
            "invalid_preconditioner_or_operator",
            "non_positive_preconditioned_inner_product",
            "solver encountered a non-positive preconditioned inner product",
        )
    if status == -3:
        return (
            "non_finite",
            "non_finite",
            "solver encountered a non-finite scalar, vector, or preconditioner output",
        )
    return (
        "numerical_breakdown",
        f"status_{status}",
        "solver reported an implementation-specific negative breakdown status",
    )


def make_solver_info(
    *,
    solver: str,
    status,
    residual_norm,
    iterations,
    preconditioned_residual_norm=None,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int | None = None,
    restart: int | None = None,
    preconditioner: str | None = None,
) -> SolverInfo:
    """Synchronize native scalar outputs and build :class:`SolverInfo`."""

    status_value = scalar_int(status)
    residual_value = scalar_float(residual_norm)
    iterations_value = scalar_int(iterations)
    preconditioned_value = (
        None
        if preconditioned_residual_norm is None
        else scalar_float(preconditioned_residual_norm)
    )
    reason, breakdown, message = status_reason(status_value)
    return SolverInfo(
        solver=solver,
        status=status_value,
        residual_norm=residual_value,
        iterations=iterations_value,
        convergence_reason=reason,
        breakdown_reason=breakdown,
        message=message,
        preconditioned_residual_norm=preconditioned_value,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        restart=restart,
        preconditioner=preconditioner,
    )
