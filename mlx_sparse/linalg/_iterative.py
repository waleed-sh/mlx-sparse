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

import mlx.core as mx

import mlx_sparse._native as _native
from mlx_sparse.linalg.utils.arrays import finite_scalar as _finite_scalar
from mlx_sparse.linalg.utils.arrays import host_bool as _host_bool
from mlx_sparse.linalg.utils.diagnostics import finish_solver_result as _finish
from mlx_sparse.linalg.utils.gmres import (
    left_preconditioned_gmres_host as _left_pgmres_host,
)
from mlx_sparse.linalg.utils.iterative import as_csr as _as_csr
from mlx_sparse.linalg.utils.iterative import float32_array as _float32_array
from mlx_sparse.linalg.utils.iterative import float32_csr as _float32_csr
from mlx_sparse.linalg.utils.iterative import initial_guess as _guess
from mlx_sparse.linalg.utils.iterative import max_iterations as _maxiter


def cg(
    A,
    b,
    *,
    x0=None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    M=None,
    callback=None,
    return_info: bool = False,
):
    """Solve a sparse SPD linear system with the conjugate gradient method.

    Conjugate gradients (CG) is an iterative Krylov solver for symmetric
    positive-definite (SPD) systems ``A @ x = b``.  Each iteration requires
    one sparse matrix-vector product dispatched to a Metal kernel via the
    native CSR engine.  CG converges in at most ``n`` steps in exact
    arithmetic and typically far fewer for well-conditioned problems.

    GPU note:
        When GPU execution is selected, the native CG iteration runs in a
        Metal kernel.  Sparse matrix-vector products, vector updates, dot
        products, and residual checks stay on the GPU.  Python argument
        validation and conversion of the returned ``info`` flag to a Python
        integer happen on the host.

    Args:
        A: Coefficient matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray` that
            is symmetric positive-definite.  A sparse-backed
            :class:`LinearOperator` is also accepted.
        b: Right-hand side vector of shape ``(n,)``.  Must be a rank-1
            ``mlx.core.array`` or anything convertible to one.
        x0: Initial guess of shape ``(n,)``.  Defaults to the zero vector
            when ``None``.
        rtol: Relative tolerance.  The solver stops when
            ``||r_k|| <= max(atol, rtol * ||b||)``.  Defaults to ``1e-5``.
        atol: Absolute tolerance floor.  Useful when ``b`` is near zero.
            Defaults to ``0.0``.
        maxiter: Maximum number of iterations.  Defaults to ``10 * n`` when
            ``None``.
        M: Optional native-backed preconditioner.  ``None`` uses the existing
            unpreconditioned native CG path.  ``preconditioners.identity`` also
            uses that path.  ``preconditioners.diagonal`` and
            ``preconditioners.jacobi`` dispatch to native Jacobi-preconditioned
            CG. ``preconditioners.ichol0`` dispatches to a native CPU
            IC(0)-preconditioned CG loop whose preconditioner application uses
            the stored incomplete Cholesky factors. ``preconditioners.chebyshev``
            dispatches to a native Chebyshev-polynomial PCG loop whose
            preconditioner application uses only sparse matrix-vector products
            and vector updates.
        callback: Optional callable invoked once after the native solve
            completes.  Native CPU/Metal Krylov loops do not call Python inside
            each iteration; using a callback synchronizes only the final
            solution.
        return_info: If ``True``, return a structured diagnostic object instead
            of the integer ``info`` flag.  The default remains ``False``.

    Returns:
        A tuple ``(x, info)`` where ``x`` is the approximate solution array
        of shape ``(n,)`` and ``info`` is an integer convergence flag by
        default.  With ``return_info=True``, ``info`` is a ``SolverInfo``
        diagnostic object.
        ``info == 0`` means the solver converged to the requested tolerance.
        ``info > 0`` is the iteration count at which the solver stopped without
        converging. ``info < 0`` indicates numerical breakdown, such as a
        non-positive preconditioned residual product, non-finite scalar, or
        scale-aware near-zero denominator.

    Raises:
        TypeError: If ``A`` is a dense ``mlx.core.array`` or an unsupported
            type, or if ``M`` is not a native-backed identity, diagonal, or
            Jacobi preconditioner.
        ValueError: If ``b`` is not rank-1 or its length does not match
            ``A.shape[0]``.

    Example:
        Build a 2-D Poisson Laplacian and solve it with CG::

            import mlx.core as mx
            import numpy as np
            import scipy.sparse
            import mlx_sparse as ms
            from mlx_sparse import linalg

            n = 16
            L = scipy.sparse.diags([-1, 4, -1], [-1, 0, 1],
                                   shape=(n, n), format='csr').astype(np.float32)
            A = ms.csr_array(
                (mx.array(L.data), mx.array(L.indices), mx.array(L.indptr)),
                shape=L.shape, canonical=True,
            )
            b = mx.ones((n,), dtype=mx.float32)
            x, info = linalg.cg(A, b, rtol=1e-6)
            print(info)  # 0 means converged
    """

    csr = _float32_csr(_as_csr(A))
    rhs = _float32_array(b)
    guess = _guess(csr, rhs, x0)
    maxiter_value = _maxiter(csr, maxiter)
    preconditioner_kind = None
    if M is not None:
        from mlx_sparse.linalg import preconditioners

        pc = preconditioners.aspreconditioner(M, csr)
        preconditioner_kind = pc.kind
        if isinstance(pc, preconditioners.IdentityPreconditioner):
            pass
        elif isinstance(pc, preconditioners.DiagonalPreconditioner):
            x, info, residual, iterations = _native.csr_pcg_jacobi(
                csr.data,
                csr.indices,
                csr.indptr,
                rhs,
                guess,
                pc.inverse_diagonal,
                csr.shape,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
            )
            return _finish(
                x,
                info,
                residual,
                iterations,
                solver="cg",
                return_info=bool(return_info),
                callback=callback,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
                preconditioner=preconditioner_kind,
            )
        elif isinstance(pc, preconditioners.IC0Preconditioner):
            upper = pc._upper()
            x, info, residual, iterations = _native.csr_pcg_ic0(
                csr.data,
                csr.indices,
                csr.indptr,
                rhs,
                guess,
                pc.L.data,
                pc.L.indices,
                pc.L.indptr,
                upper.data,
                upper.indices,
                upper.indptr,
                csr.shape,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
            )
            return _finish(
                x,
                info,
                residual,
                iterations,
                solver="cg",
                return_info=bool(return_info),
                callback=callback,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
                preconditioner=preconditioner_kind,
            )
        elif isinstance(pc, preconditioners.ChebyshevPreconditioner):
            x, info, residual, iterations = _native.csr_pcg_chebyshev(
                csr.data,
                csr.indices,
                csr.indptr,
                rhs,
                guess,
                pc.A.data,
                pc.A.indices,
                pc.A.indptr,
                csr.shape,
                degree=pc.degree,
                lambda_min=pc.lambda_min,
                lambda_max=pc.lambda_max,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
            )
            return _finish(
                x,
                info,
                residual,
                iterations,
                solver="cg",
                return_info=bool(return_info),
                callback=callback,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
                preconditioner=preconditioner_kind,
            )
        else:
            raise TypeError(
                "cg currently supports only identity, diagonal, Jacobi, IC(0), "
                "and Chebyshev native-backed preconditioners."
            )
    x, info, residual, iterations = _native.csr_cg(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        guess,
        csr.shape,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=maxiter_value,
    )
    return _finish(
        x,
        info,
        residual,
        iterations,
        solver="cg",
        return_info=bool(return_info),
        callback=callback,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=maxiter_value,
        preconditioner=preconditioner_kind,
    )


def gmres(
    A,
    b,
    *,
    x0=None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    restart: int | None = None,
    maxiter: int | None = None,
    M=None,
    callback=None,
    callback_type: str = "x",
    return_info: bool = False,
):
    """Solve a sparse linear system with restarted GMRES.

    Generalised Minimum RESidual (GMRES) is an iterative Krylov solver for
    any non-singular square system ``A @ x = b``.  Unlike CG it does not
    require ``A`` to be symmetric or positive-definite, making it the default
    choice for non-symmetric PDEs and general sparse systems.

    Each restart cycle builds an Arnoldi basis of dimension ``restart`` using
    one sparse matrix-vector product per step, then minimises the residual
    over that Krylov subspace.  Memory scales as ``O(restart * n)``.

    GPU note:
        When GPU execution is selected, Arnoldi basis construction uses the
        native Arnoldi kernel.  The restart loop, residual setup, small
        least-squares solve, solution update, and convergence bookkeeping run
        on the CPU.  The host also copies the Arnoldi basis and Hessenberg
        coefficients back before updating the solution.

    Args:
        A: Coefficient matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray`.  A
            sparse-backed :class:`LinearOperator` is also accepted.  Need not be
            symmetric.
        b: Right-hand side vector of shape ``(n,)``.
        x0: Initial guess of shape ``(n,)``.  Defaults to the zero vector.
        rtol: Relative tolerance for the residual stopping criterion.
            Defaults to ``1e-5``.
        atol: Absolute tolerance floor.  Defaults to ``0.0``.
        restart: Size of the Krylov subspace built before each restart.
            Larger values accelerate convergence at the cost of more memory.
            Defaults to ``min(20, n)``.
        maxiter: Maximum total number of matrix-vector products across all
            restart cycles.  Defaults to ``10 * n``.
        M: Optional inverse-apply preconditioner.  ``None`` uses the existing
            native unpreconditioned GMRES path. ``preconditioners.identity``
            is treated equivalently. ``preconditioners.diagonal`` and
            ``preconditioners.jacobi`` and ``preconditioners.ilu0`` dispatch to
            native left-preconditioned GMRES. Custom callables and
            inverse-apply objects use a host fallback loop. All preconditioned
            paths build Krylov vectors for ``M^{-1} A`` while convergence is
            tested against the true residual ``b - A @ x``.
        callback: Optional callable invoked once after the native solve
            completes.  Native CPU/Metal solver loops do not call Python inside
            each iteration. ``callback_type="x"`` receives the final solution;
            ``"pr_norm"`` and ``"legacy"`` receive the final reported residual
            norm.
        callback_type: One of ``"x"``, ``"pr_norm"``, or ``"legacy"``.
        return_info: If ``True``, return a structured diagnostic object instead
            of the integer ``info`` flag.  The default remains ``False``.

    Returns:
        A tuple ``(x, info)`` where ``x`` is the approximate solution of
        shape ``(n,)`` and ``info`` is an integer convergence flag by default.
        With ``return_info=True``, ``info`` is a ``SolverInfo`` diagnostic
        object.
        ``info == 0`` means the solver converged.  ``info > 0`` is the
        iteration count at termination without convergence.

    Raises:
        TypeError: If ``A`` is a dense array or an unsupported type.
        ValueError: If ``b`` is not rank-1, its length does not match
            ``A.shape[0]``, or ``restart`` is not positive.
    """

    if callback_type not in {"x", "pr_norm", "legacy"}:
        raise ValueError("callback_type must be 'x', 'pr_norm', or 'legacy'.")
    csr = _float32_csr(_as_csr(A))
    rhs = _float32_array(b)
    guess = _guess(csr, rhs, x0)
    restart_value = min(20, csr.shape[0]) if restart is None else int(restart)
    if restart_value <= 0:
        raise ValueError("restart must be positive.")
    maxiter_value = _maxiter(csr, maxiter)
    preconditioner_kind = None
    if M is not None:
        from mlx_sparse.linalg import preconditioners

        pc = preconditioners.aspreconditioner(M, csr)
        preconditioner_kind = pc.kind
        if isinstance(pc, preconditioners.DiagonalPreconditioner):
            x, info, residual, iterations = _native.csr_gmres_jacobi(
                csr.data,
                csr.indices,
                csr.indptr,
                rhs,
                guess,
                pc.inverse_diagonal,
                csr.shape,
                rtol=float(rtol),
                atol=float(atol),
                restart=restart_value,
                maxiter=maxiter_value,
            )
            return _finish(
                x,
                info,
                residual,
                iterations,
                solver="gmres",
                return_info=bool(return_info),
                callback=callback,
                callback_type=callback_type,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
                restart=restart_value,
                preconditioner=preconditioner_kind,
            )
        if isinstance(pc, preconditioners.ExactFactorPreconditioner):
            factor = pc.native_factorization
            if pc.native_apply_kind == "lu" and factor is not None:
                x, info, residual, iterations = _native.csr_gmres_exact_lu(
                    csr.data,
                    csr.indices,
                    csr.indptr,
                    rhs,
                    guess,
                    factor.perm,
                    factor.L.data,
                    factor.L.indices,
                    factor.L.indptr,
                    factor.U.data,
                    factor.U.indices,
                    factor.U.indptr,
                    csr.shape,
                    rtol=float(rtol),
                    atol=float(atol),
                    restart=restart_value,
                    maxiter=maxiter_value,
                )
                return _finish(
                    x,
                    info,
                    residual,
                    iterations,
                    solver="gmres",
                    return_info=bool(return_info),
                    callback=callback,
                    callback_type=callback_type,
                    rtol=float(rtol),
                    atol=float(atol),
                    maxiter=maxiter_value,
                    restart=restart_value,
                    preconditioner=preconditioner_kind,
                )
            if pc.native_apply_kind == "cholesky" and factor is not None:
                upper = factor._upper()
                x, info, residual, iterations = _native.csr_gmres_exact_cholesky(
                    csr.data,
                    csr.indices,
                    csr.indptr,
                    rhs,
                    guess,
                    factor.L.data,
                    factor.L.indices,
                    factor.L.indptr,
                    upper.data,
                    upper.indices,
                    upper.indptr,
                    csr.shape,
                    rtol=float(rtol),
                    atol=float(atol),
                    restart=restart_value,
                    maxiter=maxiter_value,
                )
                return _finish(
                    x,
                    info,
                    residual,
                    iterations,
                    solver="gmres",
                    return_info=bool(return_info),
                    callback=callback,
                    callback_type=callback_type,
                    rtol=float(rtol),
                    atol=float(atol),
                    maxiter=maxiter_value,
                    restart=restart_value,
                    preconditioner=preconditioner_kind,
                )
            if pc.native_apply_kind == "accelerate" and factor is not None:
                x, info, residual, iterations = _native.csr_gmres_exact_accelerate(
                    csr.data,
                    csr.indices,
                    csr.indptr,
                    rhs,
                    guess,
                    factor,
                    csr.shape,
                    rtol=float(rtol),
                    atol=float(atol),
                    restart=restart_value,
                    maxiter=maxiter_value,
                )
                return _finish(
                    x,
                    info,
                    residual,
                    iterations,
                    solver="gmres",
                    return_info=bool(return_info),
                    callback=callback,
                    callback_type=callback_type,
                    rtol=float(rtol),
                    atol=float(atol),
                    maxiter=maxiter_value,
                    restart=restart_value,
                    preconditioner=preconditioner_kind,
                )
            raise TypeError(
                "gmres exact-factor preconditioners require a native LU, "
                "native Cholesky, or guarded Accelerate factorization."
            )
        if isinstance(pc, preconditioners.ILU0Preconditioner):
            x, info, residual, iterations = _native.csr_gmres_ilu0(
                csr.data,
                csr.indices,
                csr.indptr,
                rhs,
                guess,
                pc.L.data,
                pc.L.indices,
                pc.L.indptr,
                pc.U.data,
                pc.U.indices,
                pc.U.indptr,
                csr.shape,
                rtol=float(rtol),
                atol=float(atol),
                restart=restart_value,
                maxiter=maxiter_value,
            )
            return _finish(
                x,
                info,
                residual,
                iterations,
                solver="gmres",
                return_info=bool(return_info),
                callback=callback,
                callback_type=callback_type,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
                restart=restart_value,
                preconditioner=preconditioner_kind,
            )
        if not isinstance(pc, preconditioners.IdentityPreconditioner):
            x, info, residual, iterations = _left_pgmres_host(
                csr,
                rhs,
                guess,
                pc,
                rtol=float(rtol),
                atol=float(atol),
                restart=restart_value,
                maxiter=maxiter_value,
            )
            return _finish(
                x,
                info,
                residual,
                iterations,
                solver="gmres",
                return_info=bool(return_info),
                callback=callback,
                callback_type=callback_type,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
                restart=restart_value,
                preconditioner=preconditioner_kind,
            )
    x, info, residual, iterations = _native.csr_gmres(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        guess,
        csr.shape,
        rtol=float(rtol),
        atol=float(atol),
        restart=restart_value,
        maxiter=maxiter_value,
    )
    return _finish(
        x,
        info,
        residual,
        iterations,
        solver="gmres",
        return_info=bool(return_info),
        callback=callback,
        callback_type=callback_type,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=maxiter_value,
        restart=restart_value,
        preconditioner=preconditioner_kind,
    )


def minres(
    A,
    b,
    *,
    x0=None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    shift: float = 0.0,
    maxiter: int | None = None,
    M=None,
    check_preconditioner: bool = True,
    callback=None,
    return_info: bool = False,
):
    """Solve a sparse symmetric linear system with MINRES.

    MINimum RESidual (MINRES) is an iterative Krylov solver for symmetric
    systems ``A @ x = b`` where ``A`` may be symmetric indefinite (having
    both positive and negative eigenvalues).  It minimises the 2-norm of the
    residual at every step using a Paige-Saunders Lanczos recurrence and
    requires only a constant number of vectors in memory, making it more
    memory-efficient than GMRES for large symmetric indefinite problems.

    GPU note:
        When GPU execution is selected, the unpreconditioned and
        diagonal/Jacobi-preconditioned recurrences dispatch to native Metal
        kernels.  The CPU path uses the matching native C++ recurrence.  No
        Python callback is invoked inside the Krylov loop.

    Args:
        A: Coefficient matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray` that
            is real and symmetric.
        b: Right-hand side vector of shape ``(n,)``.
        x0: Initial guess of shape ``(n,)``.  Defaults to the zero vector.
        rtol: Relative tolerance for the residual stopping criterion.
            Defaults to ``1e-5``.
        atol: Absolute tolerance floor.  Defaults to ``0.0``.
        shift: Optional scalar shift.  The solver applies MINRES to
            ``(A - shift * I) @ x = b``, matching SciPy's convention.
        maxiter: Maximum number of iterations.  Defaults to ``10 * n``.
        M: Optional symmetric positive-definite inverse-apply preconditioner.
            ``None`` and ``preconditioners.identity`` use the unpreconditioned
            path. ``preconditioners.diagonal`` and ``preconditioners.jacobi``
            dispatch to native diagonal-preconditioned MINRES. Other
            preconditioner kinds are rejected until they have native SPD MINRES
            kernels.
        check_preconditioner: When ``True`` (default), diagonal/Jacobi
            preconditioners must have finite strictly positive inverse diagonal
            entries before entering native MINRES. Setting this to ``False``
            disables the Python-side SPD validation but the native solver still
            reports numerical breakdown for invalid preconditioners.
        callback: Optional callable invoked once after the native solve
            completes.  Native CPU/Metal Krylov loops do not call Python inside
            each iteration; using a callback synchronizes only the final
            solution.
        return_info: If ``True``, return a structured diagnostic object instead
            of the integer ``info`` flag.  The default remains ``False``.

    Returns:
        A tuple ``(x, info)`` where ``x`` is the approximate solution of
        shape ``(n,)`` and ``info`` is an integer convergence flag by default.
        With ``return_info=True``, ``info`` is a ``SolverInfo`` diagnostic
        object.
        ``info == 0`` means the solver converged to the requested tolerance.
        ``info > 0`` is the iteration count at which the solver stopped
        without converging.

    Raises:
        TypeError: If ``A`` is a dense array or an unsupported type.
        ValueError: If ``b`` is not rank-1, its length does not match
            ``A.shape[0]``, or a checked diagonal preconditioner is not SPD.

    Note:
        MINRES requires ``A`` to be symmetric but not positive-definite.
        Preconditioned MINRES additionally requires an SPD preconditioner. For
        SPD systems :func:`cg` is typically faster.  For non-symmetric systems
        use :func:`gmres`.
    """

    csr = _float32_csr(_as_csr(A))
    rhs = _float32_array(b)
    guess = _guess(csr, rhs, x0)
    shift_value = _finite_scalar("shift", shift)
    maxiter_value = _maxiter(csr, maxiter)
    preconditioner_kind = None
    if M is not None:
        from mlx_sparse.linalg import preconditioners

        pc = preconditioners.aspreconditioner(M, csr)
        preconditioner_kind = pc.kind
        if isinstance(pc, preconditioners.IdentityPreconditioner):
            pass
        elif isinstance(pc, preconditioners.DiagonalPreconditioner):
            if check_preconditioner and (
                not pc.is_symmetric
                or not _host_bool(mx.all(mx.isfinite(pc.inverse_diagonal)))
                or not _host_bool(mx.all(pc.inverse_diagonal > 0.0))
            ):
                raise ValueError(
                    "minres requires a symmetric positive-definite "
                    "preconditioner; diagonal inverse entries must be finite "
                    "and strictly positive."
                )
            x, info, residual, iterations = _native.csr_minres_jacobi(
                csr.data,
                csr.indices,
                csr.indptr,
                rhs,
                guess,
                pc.inverse_diagonal,
                csr.shape,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
                shift=shift_value,
            )
            return _finish(
                x,
                info,
                residual,
                iterations,
                solver="minres",
                return_info=bool(return_info),
                callback=callback,
                rtol=float(rtol),
                atol=float(atol),
                maxiter=maxiter_value,
                preconditioner=preconditioner_kind,
            )
        else:
            raise TypeError(
                "minres currently supports only identity, diagonal, and Jacobi "
                "symmetric positive-definite native-backed preconditioners."
            )
    x, info, residual, iterations = _native.csr_minres(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        guess,
        csr.shape,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=maxiter_value,
        shift=shift_value,
    )
    return _finish(
        x,
        info,
        residual,
        iterations,
        solver="minres",
        return_info=bool(return_info),
        callback=callback,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=maxiter_value,
        preconditioner=preconditioner_kind,
    )
