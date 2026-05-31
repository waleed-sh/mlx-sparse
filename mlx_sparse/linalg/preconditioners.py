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

"""Native-backed sparse solver preconditioners.

The Python objects in this module are containers and dispatch helpers.
Application and Krylov iteration dispatch to native mlx-sparse primitives rather
than Python solver loops. Constructors may use existing sparse native kernels
and MLX scalar array expressions to build immutable setup data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import mlx.core as mx

import mlx_sparse._native as _native
from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse.linalg.utils.arrays import (
    ensure_float32_vector,
    ensure_rank1_or_rank2_rhs,
    finite_scalar,
    host_bool,
)
from mlx_sparse.linalg.utils.preconditioners import normalize_identity_dtype
from mlx_sparse.linalg.utils.sparse import canonical_csr, square_shape


@runtime_checkable
class Preconditioner(Protocol):
    """Protocol for objects that apply an approximate inverse.

    Solver-facing preconditioners represent the operation ``M^{-1} @ x``, not
    the matrix ``M`` itself.  Implementations expose a square ``shape``, value
    ``dtype``, a stable ``kind`` identifier, symmetry and positive-definiteness
    metadata, setup/apply device descriptors, effective storage ``nnz``, and a
    structured ``setup_info`` mapping.  The required :meth:`solve` method and
    ``__call__`` alias must accept rank-1 vector RHS and rank-2 dense RHS
    matrices without mutating the matrix used during setup.
    """

    shape: tuple[int, int]
    dtype: object
    kind: str
    is_symmetric: bool
    is_positive_definite: bool
    setup_device: str
    apply_device: str
    nnz: int
    setup_info: Mapping[str, object]

    def solve(self, x) -> mx.array:
        """Apply the preconditioner solve to ``x``."""

    def __call__(self, x) -> mx.array:
        """Alias for :meth:`solve`."""


@dataclass(frozen=True, slots=True)
class IdentityPreconditioner:
    """No-op inverse-apply preconditioner.

    ``IdentityPreconditioner`` is useful as an explicit baseline and as the
    normalized representation of ``M=None`` inside solver plumbing.

    Stored fields include the compatible square ``shape``, the native value
    ``dtype`` (currently ``mlx.core.float32``), the stable ``kind`` string, and
    symmetry/positive-definiteness metadata.
    """

    shape: tuple[int, int]
    dtype: object = mx.float32
    kind: str = "identity"
    is_symmetric: bool = True
    is_positive_definite: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate the stored square shape."""

        object.__setattr__(self, "shape", square_shape(self.shape))

    @property
    def nnz(self) -> int:
        """Number of effective diagonal entries."""

        return self.shape[0]

    @property
    def setup_device(self) -> str:
        """Device used during setup."""

        return "none"

    @property
    def apply_device(self) -> str:
        """Device used during inverse application."""

        return "none"

    @property
    def setup_info(self) -> Mapping[str, object]:
        """Structured metadata describing setup choices and assumptions."""

        return {
            "kind": self.kind,
            "shape": self.shape,
            "is_symmetric": self.is_symmetric,
            "is_positive_definite": self.is_positive_definite,
        }

    def solve(self, x) -> mx.array:
        """Return ``x`` after validating rank, shape, dtype, and finiteness.

        Args:
            x: Right-hand side vector ``(n,)`` or matrix ``(n, nrhs)``.

        Returns:
            ``x`` as a finite ``float32`` MLX array.
        """

        return ensure_rank1_or_rank2_rhs(
            x, leading_dim=self.shape[0], require_finite=True
        )

    def __call__(self, x) -> mx.array:
        """Alias for :meth:`solve`."""

        return self.solve(x)


@dataclass(frozen=True, slots=True)
class DiagonalPreconditioner:
    """Explicit diagonal inverse-apply preconditioner.

    The stored vector is the inverse diagonal that should multiply each row of
    a right-hand side. Application dispatches to the native
    ``diagonal_preconditioner_apply`` primitive, including rank-2 RHS support.

    Stored fields include the finite ``float32`` ``inverse_diagonal`` vector,
    the compatible square ``shape``, the stable ``kind`` string, and
    symmetry/positive-definiteness metadata.
    """

    inverse_diagonal: mx.array
    shape: tuple[int, int]
    kind: str = "diagonal"
    is_symmetric: bool = True
    is_positive_definite: bool = False

    def __post_init__(self) -> None:
        """Validate shape and inverse diagonal storage."""

        shape = square_shape(self.shape)
        inv_diag = ensure_float32_vector(
            "inverse_diagonal", self.inverse_diagonal, require_finite=True
        )
        if inv_diag.shape[0] != shape[0]:
            raise ValueError(
                f"inverse_diagonal has length {inv_diag.shape[0]}, "
                f"expected {shape[0]}."
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "inverse_diagonal", inv_diag)

    @property
    def dtype(self):
        """Value dtype of ``inverse_diagonal``."""

        return self.inverse_diagonal.dtype

    @property
    def nnz(self) -> int:
        """Number of stored inverse diagonal entries."""

        return int(self.inverse_diagonal.shape[0])

    @property
    def setup_device(self) -> str:
        """Device category used for setup validation."""

        return "host_validation"

    @property
    def apply_device(self) -> str:
        """Device category used for inverse application."""

        return "native_cpu_or_metal"

    @property
    def setup_info(self) -> Mapping[str, object]:
        """Structured metadata describing setup choices and assumptions."""

        return {
            "kind": self.kind,
            "shape": self.shape,
            "nnz": self.nnz,
            "is_symmetric": self.is_symmetric,
            "is_positive_definite": self.is_positive_definite,
        }

    def solve(self, x) -> mx.array:
        """Apply the diagonal inverse to a vector or dense RHS matrix.

        Args:
            x: Right-hand side with shape ``(n,)`` or ``(n, nrhs)``.

        Returns:
            Native-applied ``inverse_diagonal[:, None] * x`` for matrix RHS, or
            ``inverse_diagonal * x`` for vector RHS.
        """

        rhs = ensure_rank1_or_rank2_rhs(
            x, leading_dim=self.shape[0], require_finite=True
        )
        return _native.diagonal_preconditioner_apply(self.inverse_diagonal, rhs)

    def matvec(self, x) -> mx.array:
        """Alias for :meth:`solve` for SciPy-style inverse-operator use."""

        return self.solve(x)

    def __call__(self, x) -> mx.array:
        """Alias for :meth:`solve`."""

        return self.solve(x)


@dataclass(frozen=True, slots=True)
class JacobiPreconditioner(DiagonalPreconditioner):
    """Jacobi preconditioner built from a sparse matrix diagonal.

    ``JacobiPreconditioner`` is a specialized diagonal inverse-apply object
    that records the setup parameters used by :func:`jacobi`. Passing it to
    :func:`mlx_sparse.linalg.cg` dispatches to the native Jacobi-PCG primitive.

    Stored fields include ``omega``, ``shift``, ``zero_policy``, ``zero_atol``,
    whether validation was ``checked``, and ``positive_diagonal`` when the
    cheap positive shifted-diagonal check was requested.
    """

    kind: str = "jacobi"
    omega: float = 1.0
    shift: float = 0.0
    zero_policy: str = "raise"
    zero_atol: float = 0.0
    checked: bool = False
    positive_diagonal: bool | None = None

    @property
    def setup_device(self) -> str:
        """Device category used for Jacobi setup."""

        return "native_sparse_diagonal"

    @property
    def setup_info(self) -> Mapping[str, object]:
        """Structured metadata describing Jacobi setup choices."""

        return {
            "kind": self.kind,
            "shape": self.shape,
            "nnz": self.nnz,
            "omega": self.omega,
            "shift": self.shift,
            "zero_policy": self.zero_policy,
            "zero_atol": self.zero_atol,
            "checked": self.checked,
            "positive_diagonal": self.positive_diagonal,
            "is_symmetric": self.is_symmetric,
            "is_positive_definite": self.is_positive_definite,
        }


@dataclass(frozen=True, slots=True)
class CallablePreconditioner:
    """Python host inverse-apply preconditioner wrapper.

    ``CallablePreconditioner`` is the explicit normalization layer for custom
    inverse-apply objects.  The wrapped callable receives a rank-1 or rank-2
    ``float32`` MLX array and must return the same shape containing
    ``M^{-1} @ x``.  Solver integrations may use this wrapper only on documented
    host fallback paths because each application crosses through Python.

    Stored fields include the callable ``apply`` object, compatible square
    ``shape``, stable ``kind`` metadata, conservative symmetry/positive
    definiteness flags, and structured setup information.
    """

    apply: object
    shape: tuple[int, int]
    dtype: object = mx.float32
    kind: str = "callable"
    is_symmetric: bool = False
    is_positive_definite: bool = False

    def __post_init__(self) -> None:
        """Validate the callable contract metadata."""

        if not callable(self.apply):
            raise TypeError("callable preconditioner apply object must be callable.")
        object.__setattr__(self, "shape", square_shape(self.shape))
        if self.dtype != mx.float32:
            raise TypeError("callable preconditioners currently use float32 values.")

    @property
    def nnz(self) -> int:
        """Unknown effective storage count for custom callables."""

        return -1

    @property
    def setup_device(self) -> str:
        """Device category used during setup."""

        return "python_host"

    @property
    def apply_device(self) -> str:
        """Device category used during inverse application."""

        return "python_host"

    @property
    def setup_info(self) -> Mapping[str, object]:
        """Structured metadata describing the callable contract."""

        return {
            "kind": self.kind,
            "shape": self.shape,
            "assume_inverse": True,
            "is_symmetric": self.is_symmetric,
            "is_positive_definite": self.is_positive_definite,
        }

    def solve(self, x) -> mx.array:
        """Apply the wrapped inverse callable and validate its output.

        Args:
            x: Right-hand side with shape ``(n,)`` or ``(n, nrhs)``.

        Returns:
            Finite ``float32`` output with the exact same shape as ``x``.
        """

        rhs = ensure_rank1_or_rank2_rhs(
            x, leading_dim=self.shape[0], require_finite=True
        )
        result = self.apply(rhs)
        try:
            out = ensure_rank1_or_rank2_rhs(
                result, leading_dim=self.shape[0], require_finite=True
            )
        except ValueError as exc:
            raise ValueError(
                "preconditioner output shape or finite-value validation failed."
            ) from exc
        if out.shape != rhs.shape:
            raise ValueError(
                f"preconditioner output shape {out.shape} does not match "
                f"input shape {rhs.shape}."
            )
        return out

    def __call__(self, x) -> mx.array:
        """Alias for :meth:`solve`."""

        return self.solve(x)


@dataclass(frozen=True, slots=True)
class ExactFactorPreconditioner:
    """Exact inverse-apply preconditioner backed by a sparse factorization.

    This wrapper composes existing direct sparse solve objects with the
    iterative ``M`` protocol. It does not refactorize on application: setup is
    completed before construction. :meth:`solve` uses explicit native
    LU/Cholesky apply bindings when factors are available, uses the guarded
    Accelerate solver for real Accelerate factorized objects, and otherwise
    delegates to the stored reusable solve object.

    Stored fields include the reusable ``solver``, compatible square ``shape``,
    factorization ``method``, implementation ``backend``, and conservative
    symmetry/positive-definiteness metadata.  Accelerate-backed
    ``FactorizedSolve`` instances keep their Accelerate CPU apply boundary;
    native explicit factors keep their native CPU/Metal triangular-solve apply
    boundary.
    """

    solver: object
    shape: tuple[int, int]
    method: str
    backend: str
    kind: str = "exact"
    is_symmetric: bool = False
    is_positive_definite: bool = False
    factor_nnz: int = -1
    native_apply_kind: str | None = None
    native_factorization: object | None = None

    def __post_init__(self) -> None:
        """Validate the wrapped exact factorization metadata."""

        if not hasattr(self.solver, "solve") or not callable(self.solver.solve):
            raise TypeError("exact factor preconditioners require solve(x).")
        object.__setattr__(self, "shape", square_shape(self.shape))
        if self.factor_nnz < -1:
            raise ValueError("factor_nnz must be -1 or a non-negative integer.")
        if self.native_apply_kind not in {None, "lu", "cholesky", "accelerate"}:
            raise ValueError(
                "native_apply_kind must be None, 'lu', 'cholesky', or 'accelerate'."
            )

    @property
    def dtype(self):
        """Value dtype used by current sparse direct solve backends."""

        return mx.float32

    @property
    def nnz(self) -> int:
        """Stored factor nonzero count, or ``-1`` for opaque factors."""

        return int(self.factor_nnz)

    @property
    def setup_device(self) -> str:
        """Device category used during factorization setup."""

        if self.backend == "accelerate":
            return "accelerate_cpu"
        return "native_cpu"

    @property
    def apply_device(self) -> str:
        """Device category used during inverse application."""

        if self.backend == "accelerate":
            return "accelerate_cpu"
        return "native_cpu_or_metal"

    @property
    def setup_info(self) -> Mapping[str, object]:
        """Structured metadata describing the exact factorization wrapper."""

        return {
            "kind": self.kind,
            "shape": self.shape,
            "method": self.method,
            "backend": self.backend,
            "setup_device": self.setup_device,
            "apply_device": self.apply_device,
            "nnz": self.nnz,
            "is_symmetric": self.is_symmetric,
            "is_positive_definite": self.is_positive_definite,
            "solver_type": type(self.solver).__name__,
            "native_apply_kind": self.native_apply_kind,
            "has_native_solver_apply": self.native_apply_kind is not None,
        }

    def solve(self, x) -> mx.array:
        """Apply the exact factorized solve to a vector or dense RHS matrix.

        Args:
            x: Right-hand side with shape ``(n,)`` or ``(n, nrhs)``.

        Returns:
            Finite ``float32`` solution with the same shape as ``x``.
        """

        rhs = ensure_rank1_or_rank2_rhs(
            x, leading_dim=self.shape[0], require_finite=True
        )
        result = self._native_or_wrapped_solve(rhs)
        out = ensure_rank1_or_rank2_rhs(
            result, leading_dim=self.shape[1], require_finite=True
        )
        if out.shape != rhs.shape:
            raise ValueError(
                f"exact factor preconditioner output shape {out.shape} does "
                f"not match input shape {rhs.shape}."
            )
        return out

    def _native_or_wrapped_solve(self, rhs) -> mx.array:
        """Apply explicit native factors when available, otherwise delegate."""

        factor = self.native_factorization
        if self.native_apply_kind == "lu" and factor is not None:
            return _native.csr_exact_lu_preconditioner_apply(
                factor.perm,
                factor.L.data,
                factor.L.indices,
                factor.L.indptr,
                factor.U.data,
                factor.U.indices,
                factor.U.indptr,
                rhs,
                self.shape,
            )
        if self.native_apply_kind == "cholesky" and factor is not None:
            upper = factor._upper()
            return _native.csr_exact_cholesky_preconditioner_apply(
                factor.L.data,
                factor.L.indices,
                factor.L.indptr,
                upper.data,
                upper.indices,
                upper.indptr,
                rhs,
                self.shape,
            )
        return self.solver.solve(rhs)

    def matvec(self, x) -> mx.array:
        """Alias for :meth:`solve` for inverse-operator composition."""

        return self.solve(x)

    def __call__(self, x) -> mx.array:
        """Alias for :meth:`solve`."""

        return self.solve(x)


def identity(A_or_shape, *, dtype=None) -> IdentityPreconditioner:
    """Create a no-op preconditioner for a square shape or sparse matrix.

    Args:
        A_or_shape: Square sparse matrix, ``(n, n)`` shape tuple, or integer
            dimension.
        dtype: Optional dtype. The current native solver integration accepts
            only ``None`` or ``mlx.core.float32``.

    Returns:
        An :class:`IdentityPreconditioner`.
    """

    shape = square_shape(A_or_shape)
    return IdentityPreconditioner(shape=shape, dtype=normalize_identity_dtype(dtype))


def diagonal(
    inv_diag_or_diag,
    *,
    inverse: bool = False,
    shape=None,
    dtype=None,
    zero_atol: float = 0.0,
) -> DiagonalPreconditioner:
    """Create an explicit diagonal inverse-apply preconditioner.

    Args:
        inv_diag_or_diag: Rank-1 diagonal values. Interpreted as a diagonal by
            default, or as an inverse diagonal when ``inverse=True``.
        inverse: If ``True``, use ``inv_diag_or_diag`` directly as the inverse
            diagonal. If ``False``, validate and invert it.
        shape: Optional square shape. Defaults to ``(n, n)`` where ``n`` is the
            vector length.
        dtype: Optional dtype. The current native preconditioner path accepts
            only ``None`` or ``mlx.core.float32``.
        zero_atol: Absolute threshold used when rejecting zero diagonal entries
            before inversion.

    Returns:
        A :class:`DiagonalPreconditioner` with finite ``float32`` inverse
        diagonal storage.
    """

    values = ensure_float32_vector("diagonal", inv_diag_or_diag, require_finite=True)
    if dtype is not None and dtype != mx.float32:
        raise TypeError("diagonal preconditioners currently use float32 values.")
    pc_shape = (
        square_shape((values.shape[0], values.shape[0]))
        if shape is None
        else square_shape(shape)
    )
    if values.shape[0] != pc_shape[0]:
        raise ValueError(
            f"diagonal has length {values.shape[0]}, expected {pc_shape[0]}."
        )
    if inverse:
        inv_diag = values
    else:
        atol = float(zero_atol)
        if atol < 0.0:
            raise ValueError("zero_atol must be non-negative.")
        if host_bool(mx.any(mx.abs(values) <= atol)):
            raise ValueError("diagonal contains zero or near-zero entries.")
        inv_diag = 1.0 / values
    return DiagonalPreconditioner(inv_diag, pc_shape)


def jacobi(
    A,
    *,
    omega: float = 1.0,
    shift: float = 0.0,
    zero_policy: str = "raise",
    zero_atol: float = 0.0,
    check: bool = False,
) -> JacobiPreconditioner:
    """Create a Jacobi preconditioner from a sparse matrix diagonal.

    The inverse diagonal is computed as ``omega / (diag(A) + shift)`` after
    normalizing ``A`` to canonical CSR so duplicate diagonal entries are summed.
    The input sparse matrix is never mutated.

    Args:
        A: ``CSRArray``, ``COOArray``, ``CSCArray``, or sparse-backed
            ``LinearOperator``.
        omega: Damping/weighting factor.
        shift: Explicit diagonal shift applied before inversion.
        zero_policy: ``"raise"`` rejects zero/near-zero shifted diagonals.
            ``"unit"`` replaces those entries with ``1`` before inversion.
        zero_atol: Absolute threshold used to identify near-zero shifted
            diagonal entries.
        check: When ``True``, require ``omega > 0`` and a strictly positive
            shifted diagonal before any ``zero_policy`` replacement, then mark
            the preconditioner as positive definite.

    Returns:
        A :class:`JacobiPreconditioner` suitable for native PCG.
    """

    if zero_policy not in {"raise", "unit"}:
        raise ValueError("zero_policy must be 'raise' or 'unit'.")
    omega_value = finite_scalar("omega", omega)
    shift_value = finite_scalar("shift", shift)
    checked = bool(check)
    if checked and omega_value <= 0.0:
        raise ValueError("omega must be positive when check=True.")
    csr = canonical_csr(
        A,
        context="jacobi",
        dense_guidance="",
        allow_sparse_linear_operator=True,
    )
    if csr.shape[0] != csr.shape[1]:
        raise ValueError(f"jacobi requires a square matrix, got {csr.shape}.")
    diag = ensure_float32_vector("diagonal", csr.diagonal())
    shifted = diag + mx.array(shift_value, dtype=mx.float32)
    if not host_bool(mx.all(mx.isfinite(shifted))):
        raise ValueError("shifted diagonal must contain only finite values.")
    atol = float(zero_atol)
    if atol < 0.0:
        raise ValueError("zero_atol must be non-negative.")
    near_zero = mx.abs(shifted) <= atol
    positive_shifted_diagonal = host_bool(mx.all(shifted > atol)) if checked else None
    if host_bool(mx.any(near_zero)):
        if zero_policy == "raise":
            raise ValueError(
                "Jacobi shifted diagonal contains zero or near-zero entries."
            )
        shifted = mx.where(near_zero, mx.ones_like(shifted), shifted)
    positive_diagonal = None
    is_positive_definite = False
    if checked:
        positive_diagonal = positive_shifted_diagonal
        if not positive_diagonal:
            raise ValueError(
                "Jacobi shifted diagonal must be strictly positive when check=True."
            )
        is_positive_definite = True
    inv_diag = mx.array(omega_value, dtype=mx.float32) / shifted
    return JacobiPreconditioner(
        inv_diag,
        csr.shape,
        is_positive_definite=is_positive_definite,
        omega=omega_value,
        shift=shift_value,
        zero_policy=zero_policy,
        zero_atol=atol,
        checked=checked,
        positive_diagonal=positive_diagonal,
    )


def from_factorized(solver) -> ExactFactorPreconditioner:
    """Wrap an existing sparse factorization as an exact preconditioner.

    Args:
        solver: A :class:`~mlx_sparse.linalg.FactorizedSolve`,
            :class:`~mlx_sparse.linalg.SparseLU`, or
            :class:`~mlx_sparse.linalg.SparseCholesky` instance.

    Returns:
        An :class:`ExactFactorPreconditioner` whose inverse application uses a
        native exact-apply path when the factorization exposes one.

    Raises:
        TypeError: If ``solver`` is not one of the supported factorization
            objects.
        ValueError: If the factorization does not represent a square operator.
    """

    from mlx_sparse.linalg._factorizations import (
        FactorizedSolve,
        SparseCholesky,
        SparseLU,
    )
    from mlx_sparse.linalg.utils.factorization import NativeFactorizedSolve

    if isinstance(solver, SparseCholesky):
        return ExactFactorPreconditioner(
            solver=solver,
            shape=square_shape(solver.shape),
            method="cholesky",
            backend="native",
            is_symmetric=True,
            is_positive_definite=True,
            factor_nnz=int(solver.L.nnz),
            native_apply_kind="cholesky",
            native_factorization=solver,
        )
    if isinstance(solver, SparseLU):
        return ExactFactorPreconditioner(
            solver=solver,
            shape=square_shape(solver.shape),
            method="lu",
            backend="native",
            is_symmetric=False,
            is_positive_definite=False,
            factor_nnz=int(solver.L.nnz + solver.U.nnz),
            native_apply_kind="lu",
            native_factorization=solver,
        )
    if isinstance(solver, FactorizedSolve):
        shape = square_shape(solver.shape)
        if int(solver.rhs_size) != shape[0] or int(solver.solution_size) != shape[1]:
            raise ValueError(
                "exact factor preconditioners require matching RHS and "
                "solution dimensions."
            )
        method = str(solver.method)
        is_cholesky = method == "cholesky"
        native_apply_kind = None
        native_factorization = None
        factor_nnz = -1
        wrapped_solver = getattr(solver, "_solver", None)
        if isinstance(wrapped_solver, NativeFactorizedSolve):
            factor = wrapped_solver.factorization
            if isinstance(factor, SparseLU):
                native_apply_kind = "lu"
                native_factorization = factor
                factor_nnz = int(factor.L.nnz + factor.U.nnz)
            elif isinstance(factor, SparseCholesky):
                native_apply_kind = "cholesky"
                native_factorization = factor
                factor_nnz = int(factor.L.nnz)
        elif str(solver.backend) == "accelerate" and _native.is_accelerate_float_solve(
            wrapped_solver
        ):
            native_apply_kind = "accelerate"
            native_factorization = wrapped_solver
        return ExactFactorPreconditioner(
            solver=solver,
            shape=shape,
            method=method,
            backend=str(solver.backend),
            is_symmetric=method in {"cholesky", "ldlt"},
            is_positive_definite=is_cholesky,
            factor_nnz=factor_nnz,
            native_apply_kind=native_apply_kind,
            native_factorization=native_factorization,
        )
    raise TypeError(
        "from_factorized expects FactorizedSolve, SparseLU, or SparseCholesky."
    )


def exact(A, *, method: str = "auto") -> ExactFactorPreconditioner:
    """Factorize ``A`` once and return an exact inverse-apply preconditioner.

    ``exact`` is a convenience wrapper around
    :func:`mlx_sparse.linalg.factorized`. It is intended as a correctness
    baseline, diagnostic tool, and composition point for existing direct
    solvers rather than a performance headline.

    Args:
        A: Sparse coefficient matrix accepted by
            :func:`mlx_sparse.linalg.factorized`.
        method: Direct factorization method. Defaults to ``"auto"``.

    Returns:
        An :class:`ExactFactorPreconditioner` wrapping the reusable factorized
        solve object.
    """

    from mlx_sparse.linalg._factorizations import factorized

    return from_factorized(factorized(A, method=method))


def aspreconditioner(M, A=None, *, assume_inverse: bool = True) -> Preconditioner:
    """Normalize supported preconditioner-like objects.

    Args:
        M: ``None``, an existing preconditioner, an object with ``solve(x)``,
            or a callable. Sparse matrices are rejected because they do not
            explicitly define an inverse-apply contract.
        A: Optional reference matrix or shape used to validate compatibility.
        assume_inverse: Must be ``True`` for callables and custom objects,
            documenting that their output is already an inverse/preconditioner
            application.

    Returns:
        A supported preconditioner object.

    Raises:
        ValueError: If ``A`` is required or the preconditioner shape mismatches.
        TypeError: If ``M`` is a sparse matrix or unsupported object.
    """

    if M is None:
        if A is None:
            raise ValueError("A is required when M is None.")
        return identity(A)
    if isinstance(
        M,
        (
            IdentityPreconditioner,
            DiagonalPreconditioner,
            CallablePreconditioner,
            ExactFactorPreconditioner,
        ),
    ):
        if A is not None and M.shape != square_shape(A):
            raise ValueError(f"preconditioner shape {M.shape} does not match A.shape.")
        return M
    if isinstance(M, (CSRArray, COOArray, CSCArray)):
        raise TypeError(
            "sparse matrices are not inverse-apply preconditioners; use "
            "preconditioners.jacobi(A) or preconditioners.diagonal(...)."
        )
    from mlx_sparse.linalg._factorizations import (
        FactorizedSolve,
        SparseCholesky,
        SparseLU,
    )

    if isinstance(M, (FactorizedSolve, SparseLU, SparseCholesky)):
        pc = from_factorized(M)
        if A is not None and pc.shape != square_shape(A):
            raise ValueError(f"preconditioner shape {pc.shape} does not match A.shape.")
        return pc
    if hasattr(M, "solve") and callable(M.solve):
        if not assume_inverse:
            raise TypeError("custom preconditioner objects must apply the inverse.")
        shape = square_shape(getattr(M, "shape", A)) if A is None else square_shape(A)
        if hasattr(M, "shape") and square_shape(M.shape) != shape:
            raise ValueError(
                f"preconditioner shape {square_shape(M.shape)} does not match "
                f"A.shape {shape}."
            )
        return CallablePreconditioner(M.solve, shape)
    if callable(M):
        if not assume_inverse:
            raise TypeError("callable preconditioners must apply the inverse.")
        if A is None:
            raise ValueError("A is required when M is a callable.")
        return CallablePreconditioner(M, square_shape(A))
    raise TypeError(
        "M must be None, a supported preconditioner, an inverse-apply object "
        "with solve(x), or an inverse-apply callable."
    )


__all__ = [
    "DiagonalPreconditioner",
    "CallablePreconditioner",
    "ExactFactorPreconditioner",
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "Preconditioner",
    "aspreconditioner",
    "diagonal",
    "exact",
    "from_factorized",
    "identity",
    "jacobi",
]
