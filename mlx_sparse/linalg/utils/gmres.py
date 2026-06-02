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

"""Host fallback routines for GMRES preconditioner support."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import numpy as np

from mlx_sparse._csr import CSRArray
from mlx_sparse._host import to_numpy
from mlx_sparse.linalg import preconditioners


def _host_vector(x: mx.array, *, name: str) -> np.ndarray:
    """Synchronize a rank-1 MLX array to a finite ``float64`` host vector."""

    mx.eval(x)
    out = np.asarray(to_numpy(x), dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape={out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values.")
    return out


def _host_csr(csr: CSRArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synchronize CSR structural buffers for host GMRES iteration."""

    mx.eval(csr.data, csr.indices, csr.indptr)
    data = np.asarray(to_numpy(csr.data), dtype=np.float64)
    indices = np.asarray(to_numpy(csr.indices))
    indptr = np.asarray(to_numpy(csr.indptr))
    return data, indices, indptr


def _csr_matvec(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Compute a host CSR matrix-vector product without densifying ``A``."""

    out = np.empty(indptr.size - 1, dtype=np.float64)
    for row in range(out.size):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        out[row] = data[start:end] @ x[indices[start:end]]
    return out


def _preconditioner_apply(
    pc: preconditioners.Preconditioner,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a finite host vector inverse-apply callable."""

    if isinstance(pc, preconditioners.IdentityPreconditioner):
        return lambda x: x.copy()
    if isinstance(pc, preconditioners.DiagonalPreconditioner):
        inv_diag = _host_vector(pc.inverse_diagonal, name="inverse_diagonal")

        def apply_diagonal(x: np.ndarray) -> np.ndarray:
            return inv_diag * x

        return apply_diagonal

    def apply_callable(x: np.ndarray) -> np.ndarray:
        result = pc.solve(mx.array(x.astype(np.float32, copy=False)))
        out = _host_vector(result, name="preconditioner output")
        if out.shape != x.shape:
            raise ValueError(
                f"preconditioner output shape {out.shape} does not match "
                f"input shape {x.shape}."
            )
        return out

    return apply_callable


def left_preconditioned_gmres_host(
    csr: CSRArray,
    b: mx.array,
    x0: mx.array,
    M: preconditioners.Preconditioner,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
) -> tuple[mx.array, int, float, int]:
    """Solve ``A @ x = b`` with host left-preconditioned restarted GMRES.

    This path exists for Python callables and custom inverse-apply objects.
    It builds Krylov vectors for ``M^{-1} A`` but always checks convergence
    against the true residual ``b - A @ x``.  It intentionally stays serial:
    v0.0.4b1 measured fixed-worker parallelism inside Krylov iteration loops
    and rejected it for production because per-iteration launches and
    reductions dominated the sparse work.

    Args:
        csr: Canonical float32 CSR coefficient matrix.
        b: Rank-1 float32 right-hand side.
        x0: Rank-1 float32 initial guess.
        M: Normalized inverse-apply preconditioner.
        rtol: True-residual relative tolerance.
        atol: True-residual absolute tolerance.
        restart: Restart dimension.
        maxiter: Maximum total matrix-vector products.

    Returns:
        ``(x, info, residual_norm, iterations)``. ``info`` follows the sparse
        solver convention: ``0`` converged, positive iteration budget exhausted,
        negative numerical breakdown or invalid preconditioner output.
    """

    data, indices, indptr = _host_csr(csr)
    rhs = _host_vector(b, name="b")
    x = _host_vector(x0, name="x0").copy()
    apply_preconditioner = _preconditioner_apply(M)

    b_norm = float(np.linalg.norm(rhs))
    tolerance = max(float(atol), float(rtol) * b_norm)
    iterations = 0
    true_residual = rhs - _csr_matvec(data, indices, indptr, x)
    residual_norm = float(np.linalg.norm(true_residual))
    if not np.isfinite(residual_norm):
        return mx.array(x.astype(np.float32)), -3, residual_norm, iterations
    if residual_norm <= tolerance:
        return mx.array(x.astype(np.float32)), 0, residual_norm, iterations
    if maxiter == 0:
        return mx.array(x.astype(np.float32)), 1, residual_norm, iterations

    while iterations < maxiter:
        true_residual = rhs - _csr_matvec(data, indices, indptr, x)
        residual_norm = float(np.linalg.norm(true_residual))
        if not np.isfinite(residual_norm):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iterations
        if residual_norm <= tolerance:
            return mx.array(x.astype(np.float32)), 0, residual_norm, iterations

        try:
            z0 = apply_preconditioner(true_residual)
        except Exception:
            return mx.array(x.astype(np.float32)), -3, residual_norm, iterations
        if z0.shape != true_residual.shape or not np.all(np.isfinite(z0)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iterations
        beta = float(np.linalg.norm(z0))
        if not np.isfinite(beta) or beta <= np.finfo(np.float32).eps:
            return mx.array(x.astype(np.float32)), -1, residual_norm, iterations

        steps = min(restart, maxiter - iterations, csr.shape[0])
        basis = np.zeros((csr.shape[0], steps + 1), dtype=np.float64)
        hessenberg = np.zeros((steps + 1, steps), dtype=np.float64)
        basis[:, 0] = z0 / beta
        used = 0

        for col in range(steps):
            Av = _csr_matvec(data, indices, indptr, basis[:, col])
            try:
                w = apply_preconditioner(Av)
            except Exception:
                return mx.array(x.astype(np.float32)), -3, residual_norm, iterations
            if w.shape != Av.shape or not np.all(np.isfinite(w)):
                return mx.array(x.astype(np.float32)), -3, residual_norm, iterations

            for _ in range(2):
                for row in range(col + 1):
                    coeff = float(basis[:, row] @ w)
                    hessenberg[row, col] += coeff
                    w -= coeff * basis[:, row]

            h_next = float(np.linalg.norm(w))
            if not np.isfinite(h_next):
                return mx.array(x.astype(np.float32)), -3, residual_norm, iterations
            hessenberg[col + 1, col] = h_next
            used = col + 1
            if h_next <= np.finfo(np.float32).eps:
                break
            basis[:, col + 1] = w / h_next

        if used == 0:
            return mx.array(x.astype(np.float32)), -1, residual_norm, iterations

        rhs_small = np.zeros((used + 1,), dtype=np.float64)
        rhs_small[0] = beta
        try:
            y, *_ = np.linalg.lstsq(
                hessenberg[: used + 1, :used], rhs_small, rcond=None
            )
        except np.linalg.LinAlgError:
            return mx.array(x.astype(np.float32)), -1, residual_norm, iterations
        if y.shape != (used,) or not np.all(np.isfinite(y)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iterations

        x += basis[:, :used] @ y
        iterations += used

        if iterations >= maxiter:
            true_residual = rhs - _csr_matvec(data, indices, indptr, x)
            residual_norm = float(np.linalg.norm(true_residual))
            if residual_norm <= tolerance:
                return mx.array(x.astype(np.float32)), 0, residual_norm, iterations
            break

    return (
        mx.array(x.astype(np.float32)),
        maxiter if maxiter > 0 else 1,
        residual_norm,
        iterations,
    )
