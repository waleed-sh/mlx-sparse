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

from dataclasses import dataclass, field

import mlx.core as mx

import mlx_sparse._native as _native
from mlx_sparse._csr import CSRArray
from mlx_sparse.linalg.utils.arrays import ensure_array
from mlx_sparse.linalg.utils.factorization import (
    ACCELERATE_ONLY_METHODS as _ACCELERATE_ONLY_METHODS,
)
from mlx_sparse.linalg.utils.factorization import (
    NativeFactorizedSolve,
    accelerate_factorize_sparse,
)
from mlx_sparse.linalg.utils.factorization import (
    accelerate_method_available as _accelerate_method_available,
)
from mlx_sparse.linalg.utils.factorization import as_csr as _as_csr
from mlx_sparse.linalg.utils.factorization import as_sparse as _as_sparse
from mlx_sparse.linalg.utils.factorization import (
    auto_factorized_method as _auto_factorized_method,
)
from mlx_sparse.linalg.utils.factorization import (
    ensure_factor_rhs,
)
from mlx_sparse.linalg.utils.factorization import float32_csr as _float32_csr
from mlx_sparse.linalg.utils.factorization import (
    normalize_factorized_method as _normalize_factorized_method,
)
from mlx_sparse.linalg.utils.factorization import (
    should_use_accelerate as _should_use_accelerate,
)
from mlx_sparse.linalg.utils.factorization import (
    solve_accelerate_spsolve_checked as _solve_accelerate_spsolve_checked,
)
from mlx_sparse.linalg.utils.factorization import triangular_solve as _triangular_solve


@dataclass(frozen=True, slots=True)
class SparseCholesky:
    """Sparse Cholesky factorization ``A = L @ L.T``.

    The factor ``L`` is stored as a :class:`mlx_sparse.CSRArray`. Numeric
    factorization is performed by the native sparse left-looking routine and
    the solves use sparse triangular CSR kernels.

    GPU note:
        The numeric factorization that creates ``L`` runs on the CPU.  Calls
        to :meth:`solve` use native triangular-solve kernels, which run on the
        GPU when GPU execution is selected.
    """

    L: CSRArray
    _upper_factor: CSRArray | None = field(
        init=False, default=None, repr=False, compare=False
    )

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the factored square matrix."""

        return self.L.shape

    def _upper(self) -> CSRArray:
        upper = self._upper_factor
        if upper is None:
            upper = self.L.T
            object.__setattr__(self, "_upper_factor", upper)
        return upper

    def solve(self, b) -> mx.array:
        """Solve ``A @ x = b`` using the stored Cholesky factor.

        Performs two sparse triangular solves: a forward solve with ``L``
        followed by a backward solve with ``L.T``.  Both steps use native
        CSR triangular-solve kernels.

        GPU note:
            When GPU execution is selected, both triangular solves dispatch
            native GPU kernels.  The Python method call and shape checks run
            on the host.

        Args:
            b: Right-hand side array of shape ``(n,)`` or ``(n, nrhs)``.

        Returns:
            Solution array of shape ``(n,)`` or ``(n, nrhs)`` as an
            ``mlx.core.array``.

        Raises:
            ValueError: If ``b`` has the wrong shape.
        """
        y = _triangular_solve(self.L, b, lower=True, unit_diagonal=False)
        return _triangular_solve(self._upper(), y, lower=False, unit_diagonal=False)

    def __call__(self, b) -> mx.array:
        """Alias for :meth:`solve`.  Allows the factorization to be called
        directly as ``factor(b)``."""
        return self.solve(b)


@dataclass(frozen=True, slots=True)
class SparseLU:
    """Sparse LU factorization ``P @ A = L @ U``.

    ``L`` and ``U`` are CSR sparse factors. ``perm`` stores the row permutation
    applied before factorization.

    GPU note:
        The numeric LU factorization that creates ``perm``, ``L``, and ``U``
        runs on the CPU.  Calls to :meth:`solve` use native permutation and
        triangular-solve kernels, which run on the GPU when GPU execution is
        selected.
    """

    perm: mx.array
    L: CSRArray
    U: CSRArray

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the factored square matrix."""

        return self.L.shape

    def solve(self, b) -> mx.array:
        """Solve ``A @ x = b`` using the stored LU factors.

        Applies the row permutation ``P``, then performs a forward solve with
        the unit lower-triangular factor ``L`` and a backward solve with the
        upper-triangular factor ``U``.  All steps use native CSR kernels.

        GPU note:
            When GPU execution is selected, the row permutation and both
            triangular solves dispatch native GPU kernels.  The Python method
            call and shape checks run on the host.

        Args:
            b: Right-hand side array of shape ``(n,)`` or ``(n, nrhs)``.

        Returns:
            Solution array of shape ``(n,)`` or ``(n, nrhs)`` as an
            ``mlx.core.array``.

        Raises:
            ValueError: If ``b`` has the wrong shape.
        """
        rhs = ensure_factor_rhs(b, leading_dim=self.shape[0])
        permuted = _native.csr_permute_vector(rhs, self.perm)
        y = _triangular_solve(self.L, permuted, lower=True, unit_diagonal=True)
        return _triangular_solve(self.U, y, lower=False, unit_diagonal=False)

    def __call__(self, b) -> mx.array:
        """Alias for :meth:`solve`.  Allows the factorization to be called
        directly as ``factor(b)``."""
        return self.solve(b)


@dataclass(frozen=True, slots=True)
class FactorizedSolve:
    """Reusable sparse solve object.

    ``FactorizedSolve`` intentionally exposes only solve behavior and metadata,
    not explicit sparse factors.  Accelerate-enabled Apple builds use opaque
    Accelerate factorization objects for supported methods.  Other supported
    square methods fall back to the existing native explicit-factor path.
    """

    _solver: object
    shape: tuple[int, int]
    method: str
    backend: str
    rhs_size: int
    solution_size: int

    def solve(self, b) -> mx.array:
        """Solve for one or more right-hand sides using the stored factorization.

        Args:
            b: Right-hand side array of shape ``(rhs_size,)`` or
                ``(rhs_size, nrhs)``.

        Returns:
            Solution array of shape ``(solution_size,)`` or
            ``(solution_size, nrhs)``.
        """
        rhs = ensure_array(b, dtype=mx.float32)
        return self._solver.solve(rhs)

    def __call__(self, b) -> mx.array:
        """Alias for :meth:`solve`."""
        return self.solve(b)


def sparse_cholesky(A, *, upper: bool = False) -> SparseCholesky:
    """Compute the sparse Cholesky factorization ``A = L @ L.T``.

    Performs a left-looking sparse Cholesky factorization on a real symmetric
    positive-definite (SPD) matrix stored in CSR, COO, or CSC format.  COO/CSC
    inputs are converted once to canonical CSR before factorization.  The resulting
    lower-triangular factor ``L`` is returned as a :class:`SparseCholesky`
    object whose :meth:`~SparseCholesky.solve` method applies both triangular
    solves in sequence.

    GPU note:
        The factorization step runs on the CPU using the native sparse
        routine.  The resulting ``SparseCholesky.solve`` dispatches
        triangular-solve kernels to the GPU when GPU execution is selected.

    Args:
        A: The matrix to factorize.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray` that
            is real, symmetric, and positive-definite.  Float16 and bfloat16
            inputs are promoted to float32 automatically.
        upper: Not yet supported.  Must be ``False`` (the default).

    Returns:
        A :class:`SparseCholesky` dataclass holding the lower-triangular
        factor ``L`` as a :class:`~mlx_sparse.CSRArray`.

    Raises:
        NotImplementedError: If ``upper=True``.
        TypeError: If ``A`` is a dense array or has an unsupported dtype.

    Example:
        Factorize a small SPD matrix and solve a linear system::

            import mlx.core as mx
            import numpy as np
            import scipy.sparse
            import mlx_sparse as ms
            from mlx_sparse import linalg

            n = 8
            L_sp = scipy.sparse.diags([-1, 4, -1], [-1, 0, 1],
                                      shape=(n, n), format='csr').astype(np.float32)
            A = ms.csr_array(
                (mx.array(L_sp.data), mx.array(L_sp.indices), mx.array(L_sp.indptr)),
                shape=L_sp.shape, canonical=True,
            )
            factor = linalg.sparse_cholesky(A)
            b = mx.ones((n,), dtype=mx.float32)
            x = factor.solve(b)
    """
    if upper:
        raise NotImplementedError(
            "sparse_cholesky currently returns the lower CSR factor."
        )
    csr = _float32_csr(_as_csr(A))
    data, indices, indptr = _native.csr_cholesky(
        csr.data, csr.indices, csr.indptr, csr.shape
    )
    return SparseCholesky(
        L=CSRArray(
            data=data,
            indices=indices,
            indptr=indptr,
            shape=csr.shape,
            sorted_indices=True,
            has_canonical_format=True,
        )
    )


def cholesky(A, *, upper: bool = False) -> SparseCholesky:
    """Alias for :func:`sparse_cholesky`.

    Provided for compatibility with SciPy-style naming.  All arguments and
    return values are identical to :func:`sparse_cholesky`.

    GPU note:
        The factorization step runs on the CPU.  The returned factor uses GPU
        triangular-solve kernels when GPU execution is selected.

    Args:
        A: SPD matrix in :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray`
            format.
        upper: Not yet supported.  Must be ``False`` (the default).

    Returns:
        A :class:`SparseCholesky` holding the lower-triangular factor ``L``.
    """
    return sparse_cholesky(A, upper=upper)


def sparse_lu(A) -> SparseLU:
    """Compute the sparse LU factorization ``P @ A = L @ U``.

    Performs a sparse LU factorization with partial pivoting on a general
    (possibly non-symmetric) real square matrix stored in CSR, COO, or CSC
    format.  COO/CSC inputs are converted once to canonical CSR before
    factorization.  The row permutation ``P``, unit lower-triangular factor ``L``, and
    upper-triangular factor ``U`` are returned as a :class:`SparseLU` object
    whose :meth:`~SparseLU.solve` method applies the full solve sequence.

    GPU note:
        The factorization step runs on the CPU using the native sparse
        routine.  The resulting ``SparseLU.solve`` dispatches permutation and
        triangular-solve kernels to the GPU when GPU execution is selected.

    Args:
        A: The matrix to factorize.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray` that
            is real and non-singular.  Float16 and bfloat16 inputs are promoted
            to float32 automatically.

    Returns:
        A :class:`SparseLU` dataclass with fields ``perm`` (row permutation
        as an ``mlx.core.array``), ``L`` (unit lower-triangular
        :class:`~mlx_sparse.CSRArray`), and ``U`` (upper-triangular
        :class:`~mlx_sparse.CSRArray`).

    Raises:
        TypeError: If ``A`` is a dense array or has an unsupported dtype.

    Example:
        Factorize a non-symmetric sparse matrix and solve a system::

            import mlx.core as mx
            import numpy as np
            import scipy.sparse
            import mlx_sparse as ms
            from mlx_sparse import linalg

            n = 8
            rng = np.random.default_rng(0)
            B = scipy.sparse.random(n, n, density=0.4, format='csr',
                                    dtype=np.float32, random_state=rng)
            B = B + scipy.sparse.eye(n, dtype=np.float32) * n
            A = ms.csr_array(
                (mx.array(B.data), mx.array(B.indices), mx.array(B.indptr)),
                shape=B.shape, canonical=True,
            )
            factor = linalg.sparse_lu(A)
            b = mx.ones((n,), dtype=mx.float32)
            x = factor.solve(b)
    """
    csr = _float32_csr(_as_csr(A))
    perm, l_data, l_indices, l_indptr, u_data, u_indices, u_indptr = _native.csr_lu(
        csr.data, csr.indices, csr.indptr, csr.shape
    )
    return SparseLU(
        perm=perm,
        L=CSRArray(
            data=l_data,
            indices=l_indices,
            indptr=l_indptr,
            shape=csr.shape,
            sorted_indices=True,
            has_canonical_format=True,
        ),
        U=CSRArray(
            data=u_data,
            indices=u_indices,
            indptr=u_indptr,
            shape=csr.shape,
            sorted_indices=True,
            has_canonical_format=True,
        ),
    )


def splu(A) -> SparseLU:
    """Alias for :func:`sparse_lu`.

    Provided for compatibility with SciPy-style naming.  All arguments and
    return values are identical to :func:`sparse_lu`.

    GPU note:
        The factorization step runs on the CPU.  The returned factor uses GPU
        permutation and triangular-solve kernels when GPU execution is
        selected.

    Args:
        A: Matrix to factorize in :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray`
            format.

    Returns:
        A :class:`SparseLU` dataclass with fields ``perm``, ``L``, and ``U``.
    """
    return sparse_lu(A)


def factorized(A, *, method: str = "auto") -> FactorizedSolve:
    """Factorize a sparse matrix once and return a reusable solve object.

    ``method="auto"`` chooses LU for square matrices and QR for rectangular
    matrices.  On Accelerate-enabled Apple builds, supported real ``float32``
    direct solves use opaque Accelerate factorization objects.  Without
    Accelerate, square LU and Cholesky methods fall back to the existing native
    explicit sparse factors.

    Args:
        A: Coefficient matrix in CSR, CSC, or COO format.  Float16 and
            bfloat16 inputs are promoted to float32 automatically.
        method: One of ``"auto"``, ``"lu"``, ``"cholesky"``, ``"ldlt"``,
            ``"qr"``, or ``"cholesky_ata"``.  ``"ldlt"``, ``"qr"``, and
            ``"cholesky_ata"`` require an Accelerate-enabled Apple build.

    Returns:
        A :class:`FactorizedSolve` object with :meth:`~FactorizedSolve.solve`
        and ``__call__`` methods.

    Raises:
        TypeError: If ``A`` is dense or has an unsupported dtype.
        ValueError: If the selected method is incompatible with ``A.shape``.
        NotImplementedError: If the selected method requires Accelerate but
            this build does not provide it.
    """
    sparse = _as_sparse(A)
    selected = _normalize_factorized_method(method)
    if selected == "auto":
        selected = _auto_factorized_method(sparse)

    if selected in _ACCELERATE_ONLY_METHODS:
        if not _accelerate_method_available(selected):
            raise NotImplementedError(
                f"factorized(method={selected!r}) requires an "
                "Accelerate-enabled Apple build."
            )
        sparse, solver = accelerate_factorize_sparse(sparse, selected)
        return FactorizedSolve(
            _solver=solver,
            shape=sparse.shape,
            method=selected,
            backend="accelerate",
            rhs_size=int(solver.rhs_size),
            solution_size=int(solver.solution_size),
        )

    if _should_use_accelerate(sparse, selected):
        sparse, solver = accelerate_factorize_sparse(sparse, selected)
        return FactorizedSolve(
            _solver=solver,
            shape=sparse.shape,
            method=selected,
            backend="accelerate",
            rhs_size=int(solver.rhs_size),
            solution_size=int(solver.solution_size),
        )
    if selected == "lu":
        if sparse.shape[0] != sparse.shape[1]:
            raise ValueError("LU factorized solves require a square matrix.")
        factor = sparse_lu(sparse)
        solver = NativeFactorizedSolve(factor, rhs_size=sparse.shape[0])
        return FactorizedSolve(
            _solver=solver,
            shape=sparse.shape,
            method="lu",
            backend="native",
            rhs_size=sparse.shape[0],
            solution_size=sparse.shape[1],
        )
    if selected == "cholesky":
        if sparse.shape[0] != sparse.shape[1]:
            raise ValueError("Cholesky factorized solves require a square matrix.")
        factor = sparse_cholesky(sparse)
        solver = NativeFactorizedSolve(factor, rhs_size=sparse.shape[0])
        return FactorizedSolve(
            _solver=solver,
            shape=sparse.shape,
            method="cholesky",
            backend="native",
            rhs_size=sparse.shape[0],
            solution_size=sparse.shape[1],
        )
    raise NotImplementedError(
        f"factorized(method={selected!r}) requires an Accelerate-enabled Apple build."
    )


def spsolve(A, b) -> mx.array:
    """Solve the sparse linear system ``A @ x = b`` directly.

    Computes a direct sparse factorization of ``A`` and immediately applies it
    to ``b``.  Accelerate-enabled Apple builds transparently use the
    Accelerate sparse LU path for supported real square systems.  Other builds
    and unsupported cases use the native :func:`sparse_lu` path.

    For repeated solves with the same ``A`` but different right-hand sides,
    call :func:`factorized` once and reuse the resulting
    :class:`FactorizedSolve` object to avoid re-factorizing.

    GPU note:
        Direct factorization runs on the CPU.  The native fallback row
        permutation and triangular solves use GPU kernels when GPU execution is
        selected.  The Accelerate fast path uses Apple's CPU sparse solver.

    Args:
        A: Coefficient matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray` that
            is real and non-singular.
        b: Right-hand side vector of shape ``(n,)``.

    Returns:
        Solution vector ``x`` of shape ``(n,)`` as an ``mlx.core.array``.

    Raises:
        TypeError: If ``A`` is a dense array or has an unsupported dtype.
        ValueError: If ``A`` is not square or ``b`` has an incompatible shape.
    """
    sparse = _as_sparse(A)
    if sparse.shape[0] != sparse.shape[1]:
        raise ValueError("spsolve requires a square sparse matrix.")
    rhs = ensure_array(b, dtype=mx.float32)
    solver = factorized(sparse, method="lu")
    if solver.backend == "accelerate":
        return _solve_accelerate_spsolve_checked(
            sparse,
            solver,
            rhs,
            singularity_checker=sparse_lu,
        )
    return solver.solve(rhs)
