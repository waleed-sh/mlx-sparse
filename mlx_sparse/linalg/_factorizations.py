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

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

import mlx_sparse._native as _native
from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._host import to_numpy
from mlx_sparse._validation import ensure_mx_array

_REAL_DIRECT_DTYPES = {mx.float32, mx.float16, mx.bfloat16}
_FACTORIZED_METHODS = {"auto", "lu", "cholesky", "ldlt", "qr", "cholesky_ata"}
_ACCELERATE_ONLY_METHODS = {"ldlt", "qr", "cholesky_ata"}
_ACCELERATE_SPSOLVE_RESIDUAL_RTOL = 1e-3


def _as_csr(A) -> CSRArray:
    if isinstance(A, CSRArray):
        return A.canonicalize()
    if isinstance(A, COOArray):
        return A.tocsr(canonical=True)
    if isinstance(A, CSCArray):
        return A.tocsr(canonical=True)
    raise TypeError(
        "sparse factorization expects CSRArray, COOArray, or CSCArray. "
        "Dense MLX arrays belong in mlx.linalg, not mlx_sparse.linalg."
    )


def _as_sparse(A) -> CSRArray | CSCArray | COOArray:
    if isinstance(A, (CSRArray, CSCArray, COOArray)):
        return A
    raise TypeError(
        "sparse factorization expects CSRArray, COOArray, or CSCArray. "
        "Dense MLX arrays belong in mlx.linalg, not mlx_sparse.linalg."
    )


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
    raise TypeError("sparse direct factorizations currently require real float data.")


def _float32_sparse(
    A: CSRArray | CSCArray | COOArray,
) -> CSRArray | CSCArray | COOArray:
    if A.data.dtype == mx.float32:
        return A
    if A.data.dtype not in _REAL_DIRECT_DTYPES:
        raise TypeError(
            "sparse direct factorizations currently require real float data."
        )
    if isinstance(A, CSRArray):
        return CSRArray(
            data=A.data.astype(mx.float32),
            indices=A.indices,
            indptr=A.indptr,
            shape=A.shape,
            sorted_indices=A.sorted_indices,
            has_canonical_format=A.has_canonical_format,
        )
    if isinstance(A, CSCArray):
        return CSCArray(
            data=A.data.astype(mx.float32),
            indices=A.indices,
            indptr=A.indptr,
            shape=A.shape,
            sorted_indices=A.sorted_indices,
            has_canonical_format=A.has_canonical_format,
        )
    return COOArray(
        data=A.data.astype(mx.float32),
        row=A.row,
        col=A.col,
        shape=A.shape,
        has_canonical_format=A.has_canonical_format,
    )


def _triangular_solve(factor: CSRArray, b, *, lower: bool, unit_diagonal: bool):
    rhs = ensure_mx_array(b, dtype=mx.float32)
    if rhs.ndim == 1:
        return _native.csr_triangular_solve(
            factor.data,
            factor.indices,
            factor.indptr,
            rhs,
            factor.shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
        )
    if rhs.ndim == 2:
        raise NotImplementedError(
            "sparse triangular solve currently accepts rank-1 RHS."
        )
    raise ValueError(f"right-hand side must be rank-1 or rank-2, got {rhs.shape}.")


def _normalize_factorized_method(method: str) -> str:
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
    if normalized not in _FACTORIZED_METHODS:
        allowed = ", ".join(sorted(_FACTORIZED_METHODS))
        raise ValueError(f"factorized method must be one of {allowed}.")
    return normalized


def _auto_factorized_method(A: CSRArray | CSCArray | COOArray) -> str:
    if A.shape[0] == A.shape[1]:
        return "lu"
    return "qr"


def _accelerate_method_available(method: str) -> bool:
    if not _native.accelerate_solvers_available():
        return False
    if method == "lu":
        return _native.accelerate_lu_solvers_available()
    return True


def _should_use_accelerate(A: CSRArray | CSCArray | COOArray, method: str) -> bool:
    return A.data.dtype in _REAL_DIRECT_DTYPES and _accelerate_method_available(method)


def _solve_columns(solver, b, *, rhs_size: int) -> mx.array:
    rhs = ensure_mx_array(b, dtype=mx.float32)
    if rhs.ndim == 1:
        if rhs.shape[0] != rhs_size:
            raise ValueError(
                f"right-hand side has incompatible shape {rhs.shape}; "
                f"expected ({rhs_size},)."
            )
        return solver.solve(rhs)
    if rhs.ndim == 2:
        if rhs.shape[0] != rhs_size:
            raise ValueError(
                f"right-hand side has incompatible shape {rhs.shape}; "
                f"expected first dimension {rhs_size}."
            )
        if rhs.shape[1] <= 0:
            raise ValueError("right-hand side must include at least one column.")
        columns = [solver.solve(rhs[:, col])[:, None] for col in range(rhs.shape[1])]
        return mx.concatenate(columns, axis=1)
    raise ValueError(f"right-hand side must be rank-1 or rank-2, got {rhs.shape}.")


def _host_norm(values) -> float:
    array = np.asarray(values, dtype=np.float64).ravel()
    return float(np.sqrt(np.sum(array * array)))


def _check_accelerate_direct_residual(
    A: CSRArray | CSCArray | COOArray,
    x: mx.array,
    rhs: mx.array,
) -> None:
    x_np = np.asarray(to_numpy(x), dtype=np.float64)
    if not np.all(np.isfinite(x_np)):
        raise RuntimeError(
            "Accelerate sparse direct solve produced non-finite values; "
            "the matrix may be singular or ill-conditioned."
        )

    residual = A @ x - rhs
    residual_np = np.asarray(to_numpy(residual), dtype=np.float64)
    rhs_np = np.asarray(to_numpy(rhs), dtype=np.float64)
    scale = max(_host_norm(rhs_np), 1.0)
    relative_residual = _host_norm(residual_np) / scale
    if (
        not np.isfinite(relative_residual)
        or relative_residual > _ACCELERATE_SPSOLVE_RESIDUAL_RTOL
    ):
        raise RuntimeError(
            "Accelerate sparse direct solve residual is too large; "
            "the matrix may be singular or ill-conditioned."
        )


def _accelerate_singularity_probe(n: int) -> mx.array:
    values = np.ones((n,), dtype=np.float32)
    values[1::2] = -1.0
    return mx.array(values, dtype=mx.float32)


def _solve_accelerate_spsolve_checked(
    A: CSRArray | CSCArray | COOArray,
    solver,
    rhs: mx.array,
) -> mx.array:
    probe = _accelerate_singularity_probe(A.shape[0])
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

    A_float32 = _float32_sparse(A)
    _check_accelerate_direct_residual(A_float32, user_x, rhs)
    try:
        _check_accelerate_direct_residual(A_float32, probe_x, probe)
    except RuntimeError as exc:
        try:
            sparse_lu(A_float32)
        except RuntimeError:
            raise exc
    return user_x


@dataclass(frozen=True, slots=True)
class _NativeFactorizedSolve:
    factorization: SparseCholesky | SparseLU
    rhs_size: int

    def solve(self, b) -> mx.array:
        return _solve_columns(self.factorization, b, rhs_size=self.rhs_size)


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

    @property
    def shape(self) -> tuple[int, int]:
        return self.L.shape

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
            b: Right-hand side vector of shape ``(n,)``.

        Returns:
            Solution vector ``x`` of shape ``(n,)`` as an ``mlx.core.array``.

        Raises:
            ValueError: If ``b`` has the wrong shape.
        """
        y = _triangular_solve(self.L, b, lower=True, unit_diagonal=False)
        return _triangular_solve(self.L.T, y, lower=False, unit_diagonal=False)

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
            b: Right-hand side vector of shape ``(n,)``.

        Returns:
            Solution vector ``x`` of shape ``(n,)`` as an ``mlx.core.array``.

        Raises:
            NotImplementedError: If ``b`` is not rank-1.
        """
        rhs = ensure_mx_array(b, dtype=mx.float32)
        if rhs.ndim != 1:
            raise NotImplementedError("SparseLU.solve currently accepts rank-1 RHS.")
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
        rhs = ensure_mx_array(b, dtype=mx.float32)
        return self._solver.solve(rhs)

    def __call__(self, b) -> mx.array:
        """Alias for :meth:`solve`."""
        return self.solve(b)


def _make_native_factorized(
    A: CSRArray | CSCArray | COOArray,
    method: str,
) -> FactorizedSolve:
    if method == "lu":
        if A.shape[0] != A.shape[1]:
            raise ValueError("LU factorized solves require a square matrix.")
        factor = sparse_lu(A)
        solver = _NativeFactorizedSolve(factor, rhs_size=A.shape[0])
        return FactorizedSolve(
            _solver=solver,
            shape=A.shape,
            method="lu",
            backend="native",
            rhs_size=A.shape[0],
            solution_size=A.shape[1],
        )
    if method == "cholesky":
        if A.shape[0] != A.shape[1]:
            raise ValueError("Cholesky factorized solves require a square matrix.")
        factor = sparse_cholesky(A)
        solver = _NativeFactorizedSolve(factor, rhs_size=A.shape[0])
        return FactorizedSolve(
            _solver=solver,
            shape=A.shape,
            method="cholesky",
            backend="native",
            rhs_size=A.shape[0],
            solution_size=A.shape[1],
        )
    raise NotImplementedError(
        f"factorized(method={method!r}) requires an Accelerate-enabled Apple build."
    )


def _make_accelerate_factorized(
    A: CSRArray | CSCArray | COOArray,
    method: str,
) -> FactorizedSolve:
    sparse = _float32_sparse(A)
    if isinstance(sparse, CSRArray):
        solver = _native.accelerate_factorize_csr_float32(
            sparse.data, sparse.indices, sparse.indptr, sparse.shape, method
        )
    elif isinstance(sparse, CSCArray):
        solver = _native.accelerate_factorize_csc_float32(
            sparse.data, sparse.indices, sparse.indptr, sparse.shape, method
        )
    else:
        solver = _native.accelerate_factorize_coo_float32(
            sparse.data, sparse.row, sparse.col, sparse.shape, method
        )
    return FactorizedSolve(
        _solver=solver,
        shape=sparse.shape,
        method=method,
        backend="accelerate",
        rhs_size=int(solver.rhs_size),
        solution_size=int(solver.solution_size),
    )


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
        return _make_accelerate_factorized(sparse, selected)

    if _should_use_accelerate(sparse, selected):
        return _make_accelerate_factorized(sparse, selected)
    return _make_native_factorized(sparse, selected)


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
    rhs = ensure_mx_array(b, dtype=mx.float32)
    solver = factorized(sparse, method="lu")
    if solver.backend == "accelerate":
        return _solve_accelerate_spsolve_checked(sparse, solver, rhs)
    return solver.solve(rhs)
