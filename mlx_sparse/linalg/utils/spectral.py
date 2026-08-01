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

"""Spectral linalg helper routines."""

from __future__ import annotations

import mlx.core as mx

from mlx_sparse._csr import CSRArray
from mlx_sparse.linalg.utils.arrays import (
    ensure_float32_csr,
    ensure_float32_vector,
    host_bool,
)
from mlx_sparse.linalg.utils.sparse import canonical_csr


def as_csr(A) -> CSRArray:
    """Return spectral-routine input as canonical CSR."""

    return canonical_csr(
        A,
        context="sparse eigen routines",
        dense_guidance="Dense arrays belong in mlx.linalg.",
    )


def float32_csr(A: CSRArray) -> CSRArray:
    """Return spectral-routine CSR input with float32 values."""

    return ensure_float32_csr(A, context="sparse spectral routines")


def normalize_ncv(n: int, k: int, ncv: int | None) -> int:
    """Return the Lanczos/Arnoldi basis dimension used for Ritz extraction."""

    return min(n, max(k + 1, 2 * k + 1 if ncv is None else int(ncv)))


#: Fixed key for the default Krylov start vector, which is ``ones + pseudo-random``.
#:
#: The random term is what makes the start vector usable at all on structured input.
#: A constant vector is an exact eigenvector of any matrix with constant row sums, so
#: for a graph Laplacian (``L @ 1 == 0``) or a row-stochastic matrix (``P @ 1 == 1``)
#: the Krylov space built from a pure ``ones`` start is one-dimensional. Every solver
#: here then returns ``k`` copies of that single eigenpair, whatever ``which`` asked
#: for. Any component off the constant direction removes the degeneracy.
#:
#: The ones term is what keeps ``which="SM"`` cheap on a Laplacian, whose smallest
#: eigenvector is the constant vector: the sum keeps a constant-direction overlap of
#: ``1/sqrt(2)`` at every ``n``, where a pure random start would dilute it to
#: ``~1/sqrt(n)`` and need far more iterations to recover it.
#:
#: The key is fixed rather than drawn from the global stream, so ``v0=None`` is
#: reproducible across calls and processes and does not consume user random state.
_DEFAULT_START_KEY = mx.random.key(0)


def start_vector(v0, *, n: int, name: str = "v0") -> mx.array:
    """Return a finite float32 start vector for a Krylov spectral routine.

    Args:
        v0: Optional user-provided start vector.  ``None`` maps to a deterministic
            ``ones + pseudo-random`` vector, seeded from a fixed key so that repeated
            calls agree and the global random stream is left alone.  See
            ``_DEFAULT_START_KEY`` for why neither term can be dropped.
        n: Required vector length.
        name: Name used in validation errors.

    Returns:
        A finite, nonzero, rank-1 float32 vector of length ``n``.

    Raises:
        ValueError: If a user-provided vector has the wrong shape, contains
            non-finite values, or is numerically zero.
    """

    if v0 is None:
        length = int(n)
        return mx.ones((length,), dtype=mx.float32) + mx.random.normal(
            shape=(length,), dtype=mx.float32, key=_DEFAULT_START_KEY
        )
    vector = ensure_float32_vector(name, v0, require_finite=True)
    if vector.shape[0] != n:
        raise ValueError(f"{name} has length {vector.shape[0]}, expected {n}.")
    if not host_bool(mx.any(vector != 0.0)):
        raise ValueError(f"{name} must not be the zero vector.")
    return vector


def reject_iteration_controls(
    *,
    routine: str,
    tol: float = 0.0,
    maxiter: int | None = None,
) -> None:
    """Reject unsupported convergence controls for one-shot Ritz extraction.

    The current native spectral routines build one Lanczos/Arnoldi basis of
    dimension ``ncv`` and then perform Ritz extraction.  Honoring ``tol`` or
    ``maxiter`` would require an implicitly restarted convergence loop, so the
    public wrappers reject non-default values until that algorithm is present.
    """

    if maxiter is not None:
        raise NotImplementedError(
            f"{routine} maxiter requires an implicitly restarted convergence "
            "loop; the current implementation performs one ncv-bounded Ritz "
            "extraction."
        )
    if tol != 0.0:
        raise NotImplementedError(
            f"{routine} tol requires an implicitly restarted convergence loop; "
            "the current implementation performs one ncv-bounded Ritz extraction."
        )


def normalize_which(which, *, routine: str, accepted: tuple[str, ...]) -> str:
    """Return the upper-cased Ritz selector, rejecting anything unrecognized.

    The native Ritz selection sorts descending whenever the selector is not one of
    the small-end names, so an unrecognized selector silently produces the
    largest-algebraic pairs instead of failing.  That turns a typo, or a selector
    this routine does not implement, into a wrong answer rather than an error.

    Args:
        which: Selector as passed by the caller.  Matching is case-insensitive.
        routine: Name used in the error message.
        accepted: Selectors this routine implements, upper-cased.

    Returns:
        The upper-cased selector.

    Raises:
        ValueError: If ``which`` is not one of ``accepted``.
    """

    if not isinstance(which, str):
        raise ValueError(
            f"{routine} which must be a string, got {type(which).__name__}."
        )
    normalized = which.upper()
    if normalized not in accepted:
        raise ValueError(
            f"{routine} which must be one of {', '.join(accepted)}, got {which!r}."
        )
    return normalized
