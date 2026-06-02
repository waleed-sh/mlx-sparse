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

"""Structured iterative-solver diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolverInfo:
    """Structured diagnostics for native iterative sparse solvers.

    The public solver default remains ``(x, int)``. When a solver is called
    with ``return_info=True``, the integer is replaced by ``SolverInfo`` so
    callers can inspect the native status, true residual norm, iteration count,
    and a stable textual convergence or breakdown reason without changing the
    default fast path.
    """

    solver: str
    status: int
    residual_norm: float
    iterations: int
    convergence_reason: str
    breakdown_reason: str | None
    message: str
    preconditioned_residual_norm: float | None = None
    rtol: float | None = None
    atol: float | None = None
    maxiter: int | None = None
    restart: int | None = None
    preconditioner: str | None = None

    @property
    def converged(self) -> bool:
        """Whether the solver reported convergence."""

        return self.status == 0

    @property
    def info(self) -> int:
        """SciPy-style integer status code."""

        return self.status

    def __int__(self) -> int:
        """Return :attr:`status` for compatibility with integer checks."""

        return self.status
