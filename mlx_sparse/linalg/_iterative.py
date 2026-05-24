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
import numpy as np

import mlx_sparse._native as _native
from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._host import to_numpy
from mlx_sparse._validation import ensure_mx_array


def _as_csr(A) -> CSRArray:
    if isinstance(A, CSRArray):
        return A.canonicalize()
    if isinstance(A, COOArray):
        return A.tocsr(canonical=True)
    if isinstance(A, CSCArray):
        return A.tocsr(canonical=True)
    # Accept LinearOperator when it wraps a sparse array (created via
    # aslinearoperator). The _sparse_array attribute is normalized to CSR for
    # fast native solver dispatch.
    from mlx_sparse.linalg._interface import LinearOperator

    if isinstance(A, LinearOperator):
        if A._sparse_array is not None:
            return _as_csr(A._sparse_array)
        raise TypeError(
            "Iterative solvers accept LinearOperator only when it wraps a "
            "CSRArray, COOArray, or CSCArray (use aslinearoperator(sparse_array)). "
            "For fully matrix-free operators, implement a Python-level "
            "iterative solver loop."
        )
    raise TypeError(
        "sparse iterative solvers expect CSRArray, COOArray, CSCArray, or a "
        "sparse-backed LinearOperator. Use mlx.linalg for dense arrays."
    )


def _float32_array(x) -> mx.array:
    array = ensure_mx_array(x)
    if array.dtype == mx.float32:
        return array
    if array.dtype in {mx.float16, mx.bfloat16}:
        return array.astype(mx.float32)
    raise TypeError("sparse iterative solvers currently require real float data.")


def _float32_csr(A: CSRArray) -> CSRArray:
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
    raise TypeError("sparse iterative solvers currently require real float data.")


def _info(info) -> int:
    mx.eval(info)
    return int(np.asarray(to_numpy(info)).item())


def _guess(csr: CSRArray, b: mx.array, x0) -> mx.array:
    if b.ndim != 1:
        raise ValueError(f"right-hand side must be rank-1, got {b.shape}.")
    if b.shape[0] != csr.shape[0]:
        raise ValueError(f"b has length {b.shape[0]}, expected {csr.shape[0]}.")
    if x0 is None:
        return mx.zeros((csr.shape[1],), dtype=mx.float32)
    x = _float32_array(x0)
    if x.ndim != 1 or x.shape[0] != csr.shape[1]:
        raise ValueError(f"x0 must have shape ({csr.shape[1]},), got {x.shape}.")
    return x


def _maxiter(csr: CSRArray, maxiter: int | None) -> int:
    value = 10 * csr.shape[1] if maxiter is None else int(maxiter)
    if value < 0:
        raise ValueError("maxiter must be non-negative.")
    return value


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
        M: Not supported.  Pass ``None`` (the default).
        callback: Not supported.  Pass ``None`` (the default).

    Returns:
        A tuple ``(x, info)`` where ``x`` is the approximate solution array
        of shape ``(n,)`` and ``info`` is an integer convergence flag.
        ``info == 0`` means the solver converged to the requested tolerance.
        ``info > 0`` is the iteration count at which the solver stopped without
        converging.

    Raises:
        NotImplementedError: If ``M`` or ``callback`` is not ``None``.
        TypeError: If ``A`` is a dense ``mlx.core.array`` or an unsupported
            type.
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

    if M is not None or callback is not None:
        raise NotImplementedError("native sparse cg does not use Python callbacks.")
    csr = _float32_csr(_as_csr(A))
    rhs = _float32_array(b)
    guess = _guess(csr, rhs, x0)
    x, info, _, _ = _native.csr_cg(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        guess,
        csr.shape,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=_maxiter(csr, maxiter),
    )
    return x, _info(info)


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
        M: Not supported.  Pass ``None`` (the default).
        callback: Not supported.  Pass ``None`` (the default).
        callback_type: Accepted for API compatibility but ignored.

    Returns:
        A tuple ``(x, info)`` where ``x`` is the approximate solution of
        shape ``(n,)`` and ``info`` is an integer convergence flag.
        ``info == 0`` means the solver converged.  ``info > 0`` is the
        iteration count at termination without convergence.

    Raises:
        NotImplementedError: If ``M`` or ``callback`` is not ``None``.
        TypeError: If ``A`` is a dense array or an unsupported type.
        ValueError: If ``b`` is not rank-1, its length does not match
            ``A.shape[0]``, or ``restart`` is not positive.
    """

    if M is not None or callback is not None:
        raise NotImplementedError("native sparse gmres does not use Python callbacks.")
    if callback_type not in {"x", "pr_norm", "legacy"}:
        raise ValueError("callback_type must be 'x', 'pr_norm', or 'legacy'.")
    csr = _float32_csr(_as_csr(A))
    rhs = _float32_array(b)
    guess = _guess(csr, rhs, x0)
    restart_value = min(20, csr.shape[0]) if restart is None else int(restart)
    if restart_value <= 0:
        raise ValueError("restart must be positive.")
    x, info, _, _ = _native.csr_gmres(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        guess,
        csr.shape,
        rtol=float(rtol),
        atol=float(atol),
        restart=restart_value,
        maxiter=_maxiter(csr, maxiter),
    )
    return x, _info(info)


def minres(
    A,
    b,
    *,
    x0=None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    callback=None,
):
    """Solve a sparse symmetric linear system with MINRES.

    MINimum RESidual (MINRES) is an iterative Krylov solver for symmetric
    systems ``A @ x = b`` where ``A`` may be symmetric indefinite (having
    both positive and negative eigenvalues).  It minimises the 2-norm of the
    residual at every step using a Lanczos-based recurrence and requires only
    a constant number of vectors in memory, making it more memory-efficient
    than GMRES for large symmetric indefinite problems.

    GPU note:
        When GPU execution is selected, Lanczos basis construction uses the
        native Lanczos kernel.  Residual setup, the small least-squares solve,
        solution update, and final residual check run on the CPU.  The host
        also copies the Lanczos coefficients and basis back before forming
        the solution.

    Args:
        A: Coefficient matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray` that
            is real and symmetric.
        b: Right-hand side vector of shape ``(n,)``.
        x0: Initial guess of shape ``(n,)``.  Defaults to the zero vector.
        rtol: Relative tolerance for the residual stopping criterion.
            Defaults to ``1e-5``.
        atol: Absolute tolerance floor.  Defaults to ``0.0``.
        maxiter: Maximum number of iterations.  Defaults to ``10 * n``.
        callback: Not supported.  Pass ``None`` (the default).

    Returns:
        A tuple ``(x, info)`` where ``x`` is the approximate solution of
        shape ``(n,)`` and ``info`` is an integer convergence flag.
        ``info == 0`` means the solver converged to the requested tolerance.
        ``info > 0`` is the iteration count at which the solver stopped
        without converging.

    Raises:
        NotImplementedError: If ``callback`` is not ``None``.
        TypeError: If ``A`` is a dense array or an unsupported type.
        ValueError: If ``b`` is not rank-1 or its length does not match
            ``A.shape[0]``.

    Note:
        MINRES requires ``A`` to be symmetric but not positive-definite.  For
        SPD systems :func:`cg` is typically faster.  For non-symmetric systems
        use :func:`gmres`.
    """

    if callback is not None:
        raise NotImplementedError("native sparse minres does not use Python callbacks.")
    csr = _float32_csr(_as_csr(A))
    rhs = _float32_array(b)
    guess = _guess(csr, rhs, x0)
    x, info, _, _ = _native.csr_minres(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        guess,
        csr.shape,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=_maxiter(csr, maxiter),
    )
    return x, _info(info)
