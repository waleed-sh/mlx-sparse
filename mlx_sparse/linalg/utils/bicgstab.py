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

"""Host fallback routines for BiCGSTAB preconditioner support."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_sparse._csr import CSRArray
from mlx_sparse.linalg import preconditioners
from mlx_sparse.linalg.utils.gmres import (
    _csr_matvec,
    _host_csr,
    _host_vector,
    _preconditioner_apply,
)


def left_preconditioned_bicgstab_host(
    csr: CSRArray,
    b: mx.array,
    x0: mx.array,
    M: preconditioners.Preconditioner,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
) -> tuple[mx.array, int, float, int]:
    """Solve ``A @ x = b`` with host left-preconditioned BiCGSTAB.

    This path exists only for Python callables and custom inverse-apply
    objects. It intentionally synchronizes on every sparse matvec and
    preconditioner application, so native identity/Jacobi/ILU0/exact-factor
    paths should be preferred for performance. The coefficient matrix remains
    sparse CSR; this fallback never densifies ``A``.

    Returns:
        ``(x, info, residual_norm, iterations)`` with the same SciPy-style
        status convention as native solvers.
    """

    data, indices, indptr = _host_csr(csr)
    rhs = _host_vector(b, name="b")
    x = _host_vector(x0, name="x0").copy()
    apply_preconditioner = _preconditioner_apply(M)
    b_norm = float(np.linalg.norm(rhs))
    tolerance = max(float(atol), float(rtol) * b_norm)
    eps = np.finfo(np.float32).eps

    def true_residual() -> np.ndarray:
        return rhs - _csr_matvec(data, indices, indptr, x)

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
        except Exception:
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1
        if p_hat.shape != p.shape or not np.all(np.isfinite(p_hat)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration - 1

        v = _csr_matvec(data, indices, indptr, p_hat)
        if not np.all(np.isfinite(v)):
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
        except Exception:
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration
        if s_hat.shape != s.shape or not np.all(np.isfinite(s_hat)):
            return mx.array(x.astype(np.float32)), -3, residual_norm, iteration

        t = _csr_matvec(data, indices, indptr, s_hat)
        if not np.all(np.isfinite(t)):
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

    true_r = true_residual()
    residual_norm = float(np.linalg.norm(true_r))
    status = 0 if np.isfinite(residual_norm) and residual_norm <= tolerance else maxiter
    return mx.array(x.astype(np.float32)), status, residual_norm, maxiter
