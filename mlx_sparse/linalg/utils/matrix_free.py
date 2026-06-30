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

"""Bounded host fallbacks for matrix-free Krylov solvers."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import numpy as np

import mlx_sparse.linalg.preconditioners as preconditioners
from mlx_sparse._host import to_numpy
from mlx_sparse.linalg._interface import LinearOperator
from mlx_sparse.linalg.utils.arrays import ensure_float32_vector


def is_fully_matrix_free(A) -> bool:
    """Return whether ``A`` is a matrix-free ``LinearOperator``.

    Sparse-backed ``LinearOperator`` instances intentionally return ``False``
    so they continue to use the native CSR solver path.
    """

    return isinstance(A, LinearOperator) and A._sparse_array is None


def normalize_matrix_free_inputs(
    A: LinearOperator,
    b,
    x0,
    maxiter: int | None,
) -> tuple[mx.array, mx.array, int]:
    """Validate matrix-free solver inputs and return float32 MLX vectors."""

    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"matrix-free iterative solvers require square A, got {A.shape}."
        )
    rhs = ensure_float32_vector("b", b, require_finite=True)
    if rhs.shape[0] != A.shape[0]:
        raise ValueError(f"b has length {rhs.shape[0]}, expected {A.shape[0]}.")
    if x0 is None:
        guess = mx.zeros((A.shape[1],), dtype=mx.float32)
    else:
        guess = ensure_float32_vector("x0", x0, require_finite=True)
        if guess.shape[0] != A.shape[1]:
            raise ValueError(f"x0 must have shape ({A.shape[1]},), got {guess.shape}.")
    maxiter_value = 10 * A.shape[1] if maxiter is None else int(maxiter)
    if maxiter_value < 0:
        raise ValueError("maxiter must be non-negative.")
    return rhs, guess, maxiter_value


def normalize_preconditioner(
    M,
    *,
    shape: tuple[int, int],
) -> tuple[preconditioners.Preconditioner, str | None]:
    """Return a normalized inverse-apply preconditioner for host fallbacks."""

    if M is None:
        return preconditioners.identity(shape), None
    pc = preconditioners.aspreconditioner(M, shape)
    return pc, pc.kind


def _host_vector(
    x: mx.array, *, name: str, expected_size: int | None = None
) -> np.ndarray:
    """Synchronize a finite rank-1 MLX array to a float64 host vector."""

    mx.eval(x)
    out = np.asarray(to_numpy(x), dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape={out.shape}.")
    if expected_size is not None and out.shape[0] != expected_size:
        raise ValueError(f"{name} has length {out.shape[0]}, expected {expected_size}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values.")
    return out


def _apply_operator(A: LinearOperator, x: np.ndarray) -> np.ndarray:
    """Apply a matrix-free operator and validate the host vector result."""

    result = A.matvec(mx.array(x.astype(np.float32, copy=False)))
    return _host_vector(result, name="LinearOperator output", expected_size=A.shape[0])


def _preconditioner_apply(
    M: preconditioners.Preconditioner,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a finite host inverse-apply callable for one vector RHS."""

    if isinstance(M, preconditioners.IdentityPreconditioner):
        return lambda x: x.copy()
    if isinstance(M, preconditioners.DiagonalPreconditioner):
        inv_diag = _host_vector(
            M.inverse_diagonal,
            name="inverse_diagonal",
            expected_size=M.shape[0],
        )

        def apply_diagonal(x: np.ndarray) -> np.ndarray:
            return inv_diag * x

        return apply_diagonal

    def apply(x: np.ndarray) -> np.ndarray:
        result = M.solve(mx.array(x.astype(np.float32, copy=False)))
        return _host_vector(
            result,
            name="preconditioner output",
            expected_size=M.shape[0],
        )

    return apply


def cg_matrix_free_host(
    A: LinearOperator,
    b: mx.array,
    x0: mx.array,
    M: preconditioners.Preconditioner,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
) -> tuple[mx.array, int, float, int]:
    """Solve ``A @ x = b`` with a host matrix-free PCG fallback.

    This path exists for fully matrix-free ``LinearOperator`` inputs. It is
    intentionally serial and host-synchronizing at each matvec/preconditioner
    application. Sparse-backed operators keep using the native CPU/Metal CSR
    path.
    """

    rhs = _host_vector(b, name="b", expected_size=A.shape[0])
    x = _host_vector(x0, name="x0", expected_size=A.shape[1]).copy()
    apply_preconditioner = _preconditioner_apply(M)
    b_norm = float(np.linalg.norm(rhs))
    tolerance = max(float(atol), float(rtol) * b_norm)

    try:
        r = rhs - _apply_operator(A, x)
    except Exception:
        residual_norm = float("nan")
        return mx.array(x.astype(np.float32)), -3, residual_norm, 0
    residual_norm = float(np.linalg.norm(r))
    if not np.isfinite(residual_norm):
        return mx.array(x.astype(np.float32)), -3, residual_norm, 0
    if residual_norm <= tolerance:
        return mx.array(x.astype(np.float32)), 0, residual_norm, 0
    if maxiter == 0:
        return mx.array(x.astype(np.float32)), 1, residual_norm, 0

    try:
        z = apply_preconditioner(r)
    except Exception:
        return mx.array(x.astype(np.float32)), -3, residual_norm, 0
    if z.shape != r.shape or not np.all(np.isfinite(z)):
        return mx.array(x.astype(np.float32)), -3, residual_norm, 0
    rho = float(r @ z)
    if not np.isfinite(rho) or rho <= 0.0:
        return mx.array(x.astype(np.float32)), -2, residual_norm, 0
    p = z.copy()
    eps = np.finfo(np.float32).eps

    for iteration in range(1, maxiter + 1):
        try:
            Ap = _apply_operator(A, p)
        except Exception:
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1
        if not np.all(np.isfinite(Ap)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1
        denom = float(p @ Ap)
        scale = max(float(np.linalg.norm(p)) * float(np.linalg.norm(Ap)), 1.0)
        if not np.isfinite(denom) or abs(denom) <= eps * scale:
            return mx.array(x.astype(np.float32)), -1, residual_norm, iteration - 1
        alpha = rho / denom
        if not np.isfinite(alpha):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1
        x += alpha * p
        r -= alpha * Ap
        residual_norm = float(np.linalg.norm(r))
        if not np.isfinite(residual_norm):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        if residual_norm <= tolerance:
            return mx.array(x.astype(np.float32)), 0, residual_norm, iteration
        try:
            z = apply_preconditioner(r)
        except Exception:
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        if z.shape != r.shape or not np.all(np.isfinite(z)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        rho_next = float(r @ z)
        if not np.isfinite(rho_next) or rho_next <= 0.0:
            return mx.array(x.astype(np.float32)), -2, residual_norm, iteration
        beta = rho_next / rho
        if not np.isfinite(beta):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        p = z + beta * p
        rho = rho_next

    try:
        true_residual = rhs - _apply_operator(A, x)
        residual_norm = float(np.linalg.norm(true_residual))
    except Exception:
        residual_norm = float("nan")
        return mx.array(x.astype(np.float32)), -3, residual_norm, maxiter
    status = 0 if np.isfinite(residual_norm) and residual_norm <= tolerance else maxiter
    return mx.array(x.astype(np.float32)), status, residual_norm, maxiter


def bicgstab_matrix_free_host(
    A: LinearOperator,
    b: mx.array,
    x0: mx.array,
    M: preconditioners.Preconditioner,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
) -> tuple[mx.array, int, float, int]:
    """Solve ``A @ x = b`` with a host matrix-free BiCGSTAB fallback.

    This fallback is intentionally serial and synchronizes for every operator
    and preconditioner application. Sparse-backed inputs keep using the native
    CSR CPU/Metal BiCGSTAB paths.
    """

    rhs = _host_vector(b, name="b", expected_size=A.shape[0])
    x = _host_vector(x0, name="x0", expected_size=A.shape[1]).copy()
    apply_preconditioner = _preconditioner_apply(M)
    b_norm = float(np.linalg.norm(rhs))
    tolerance = max(float(atol), float(rtol) * b_norm)
    eps = np.finfo(np.float32).eps

    def true_residual() -> np.ndarray:
        return rhs - _apply_operator(A, x)

    def near_zero_dot(value: float, lhs: np.ndarray, rhs_vec: np.ndarray) -> bool:
        scale = float(np.linalg.norm(lhs) * np.linalg.norm(rhs_vec))
        return (not np.isfinite(scale)) or abs(value) <= eps * max(1.0, scale)

    try:
        r = true_residual()
    except Exception:
        return mx.array(x.astype(np.float32)), -3, float("nan"), 0
    residual_norm = float(np.linalg.norm(r))
    if not np.isfinite(residual_norm):
        return mx.array(x.astype(np.float32)), -3, residual_norm, 0
    if residual_norm <= tolerance:
        return mx.array(x.astype(np.float32)), 0, residual_norm, 0
    if maxiter == 0:
        return mx.array(x.astype(np.float32)), 1, residual_norm, 0

    r_hat = r.copy()
    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    p = np.zeros_like(r)
    v = np.zeros_like(r)

    for iteration in range(1, maxiter + 1):
        rho = float(r_hat @ r)
        if not np.isfinite(rho) or near_zero_dot(rho, r_hat, r):
            return mx.array(x.astype(np.float32)), -1, residual_norm, iteration - 1
        if iteration == 1:
            p = r.copy()
        else:
            if abs(omega) <= eps:
                return mx.array(x.astype(np.float32)), -1, residual_norm, iteration - 1
            beta = (rho / rho_prev) * (alpha / omega)
            if not np.isfinite(beta):
                return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1
            p = r + beta * (p - omega * v)
            if not np.all(np.isfinite(p)):
                return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1
        try:
            p_hat = apply_preconditioner(p)
            v = _apply_operator(A, p_hat)
        except Exception:
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1
        if p_hat.shape != p.shape or not np.all(np.isfinite(p_hat)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1
        if v.shape != r.shape or not np.all(np.isfinite(v)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1

        alpha_den = float(r_hat @ v)
        if not np.isfinite(alpha_den) or near_zero_dot(alpha_den, r_hat, v):
            return mx.array(x.astype(np.float32)), -1, residual_norm, iteration - 1
        alpha = rho / alpha_den
        if not np.isfinite(alpha):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1

        s = r - alpha * v
        x = x + alpha * p_hat
        residual_norm = float(np.linalg.norm(s))
        if not np.isfinite(residual_norm) or not np.all(np.isfinite(x)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        if residual_norm <= tolerance:
            true_r = true_residual()
            true_norm = float(np.linalg.norm(true_r))
            if not np.isfinite(true_norm):
                return mx.array(x.astype(np.float32)), -3, true_norm, iteration
            if true_norm <= tolerance:
                return mx.array(x.astype(np.float32)), 0, true_norm, iteration

        try:
            s_hat = apply_preconditioner(s)
            t = _apply_operator(A, s_hat)
        except Exception:
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        if s_hat.shape != s.shape or not np.all(np.isfinite(s_hat)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        if t.shape != s.shape or not np.all(np.isfinite(t)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration

        omega_den = float(t @ t)
        omega_num = float(t @ s)
        if not np.isfinite(omega_num) or not np.isfinite(omega_den):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        if omega_den <= eps * max(1.0, float(np.linalg.norm(t))):
            return mx.array(x.astype(np.float32)), -1, residual_norm, iteration
        omega = omega_num / omega_den
        if not np.isfinite(omega):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration

        x = x + omega * s_hat
        r = s - omega * t
        residual_norm = float(np.linalg.norm(r))
        if (
            not np.isfinite(residual_norm)
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(r))
        ):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        if residual_norm <= tolerance:
            true_r = true_residual()
            true_norm = float(np.linalg.norm(true_r))
            if not np.isfinite(true_norm):
                return mx.array(x.astype(np.float32)), -3, true_norm, iteration
            if true_norm <= tolerance:
                return mx.array(x.astype(np.float32)), 0, true_norm, iteration
            r = true_r
            residual_norm = true_norm
        if abs(omega) <= eps:
            return mx.array(x.astype(np.float32)), -1, residual_norm, iteration
        rho_prev = rho

    try:
        true_r = true_residual()
        residual_norm = float(np.linalg.norm(true_r))
    except Exception:
        return mx.array(x.astype(np.float32)), -3, float("nan"), maxiter
    status = 0 if np.isfinite(residual_norm) and residual_norm <= tolerance else maxiter
    return mx.array(x.astype(np.float32)), status, residual_norm, maxiter


def gmres_matrix_free_host(
    A: LinearOperator,
    b: mx.array,
    x0: mx.array,
    M: preconditioners.Preconditioner,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
) -> tuple[mx.array, int, float, int]:
    """Solve ``A @ x = b`` with host left-preconditioned matrix-free GMRES."""

    rhs = _host_vector(b, name="b", expected_size=A.shape[0])
    x = _host_vector(x0, name="x0", expected_size=A.shape[1]).copy()
    apply_preconditioner = _preconditioner_apply(M)

    b_norm = float(np.linalg.norm(rhs))
    tolerance = max(float(atol), float(rtol) * b_norm)
    iterations = 0

    try:
        true_residual = rhs - _apply_operator(A, x)
    except Exception:
        return mx.array(x.astype(np.float32)), -3, float("nan"), iterations
    residual_norm = float(np.linalg.norm(true_residual))
    if not np.isfinite(residual_norm):
        return mx.array(x.astype(np.float32)), -3, residual_norm, iterations
    if residual_norm <= tolerance:
        return mx.array(x.astype(np.float32)), 0, residual_norm, iterations
    if maxiter == 0:
        return mx.array(x.astype(np.float32)), 1, residual_norm, iterations

    while iterations < maxiter:
        try:
            true_residual = rhs - _apply_operator(A, x)
        except Exception:
            return mx.array(x.astype(np.float32)), -3, float("nan"), iterations
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

        steps = min(int(restart), maxiter - iterations, A.shape[0])
        basis = np.zeros((A.shape[0], steps + 1), dtype=np.float64)
        hessenberg = np.zeros((steps + 1, steps), dtype=np.float64)
        basis[:, 0] = z0 / beta
        used = 0

        for col in range(steps):
            try:
                Av = _apply_operator(A, basis[:, col])
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

    try:
        true_residual = rhs - _apply_operator(A, x)
        residual_norm = float(np.linalg.norm(true_residual))
    except Exception:
        residual_norm = float("nan")
        return mx.array(x.astype(np.float32)), -3, residual_norm, iterations
    status = 0 if np.isfinite(residual_norm) and residual_norm <= tolerance else maxiter
    return mx.array(x.astype(np.float32)), status, residual_norm, iterations
