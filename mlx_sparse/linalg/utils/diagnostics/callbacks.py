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

"""Exit-callback and return-value helpers for iterative sparse solvers."""

from __future__ import annotations

import mlx.core as mx

from mlx_sparse.linalg.utils.diagnostics.info import SolverInfo
from mlx_sparse.linalg.utils.diagnostics.status import make_solver_info, scalar_int


def invoke_callback(
    callback,
    *,
    solver: str,
    callback_type: str | None,
    x: mx.array,
    info: SolverInfo,
) -> None:
    """Invoke an opt-in solver callback after the native loop completes.

    Native CPU/Metal solvers intentionally do not call Python inside each
    Krylov iteration. If a user provides a callback, it is called once after
    the native solve has completed. ``gmres(callback_type="x")`` receives the
    final solution array. ``"pr_norm"`` and ``"legacy"`` receive the final
    reported residual norm, which is the true residual unless a future native
    path exposes a distinct preconditioned residual norm. ``"legacy"`` keeps
    mlx-sparse's normal ``maxiter`` accounting; it does not request a Python
    callback stream from inside native restart cycles.
    """

    if callback is None:
        return
    if not callable(callback):
        raise TypeError(f"{solver} callback must be callable.")
    if callback_type in {"pr_norm", "legacy"}:
        payload = (
            info.preconditioned_residual_norm
            if info.preconditioned_residual_norm is not None
            else info.residual_norm
        )
    else:
        mx.eval(x)
        payload = x
    callback(payload)


def finish_solver_result(
    x: mx.array,
    status,
    residual_norm,
    iterations,
    *,
    solver: str,
    return_info: bool,
    callback=None,
    callback_type: str | None = None,
    preconditioned_residual_norm=None,
    rtol: float | None = None,
    atol: float | None = None,
    maxiter: int | None = None,
    restart: int | None = None,
    preconditioner: str | None = None,
) -> tuple[mx.array, int | SolverInfo]:
    """Return ``(x, int)`` by default or ``(x, SolverInfo)`` on request."""

    if return_info or callback is not None:
        diagnostic = make_solver_info(
            solver=solver,
            status=status,
            residual_norm=residual_norm,
            iterations=iterations,
            preconditioned_residual_norm=preconditioned_residual_norm,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            restart=restart,
            preconditioner=preconditioner,
        )
        invoke_callback(
            callback,
            solver=solver,
            callback_type=callback_type,
            x=x,
            info=diagnostic,
        )
        return x, diagnostic if return_info else diagnostic.status
    return x, scalar_int(status)
