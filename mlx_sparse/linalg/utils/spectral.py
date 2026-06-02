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


def start_vector(v0, *, n: int, name: str = "v0") -> mx.array:
    """Return a finite float32 start vector for a Krylov spectral routine.

    Args:
        v0: Optional user-provided start vector.  ``None`` maps to the current
            deterministic all-ones vector.
        n: Required vector length.
        name: Name used in validation errors.

    Returns:
        A finite, nonzero, rank-1 float32 vector of length ``n``.

    Raises:
        ValueError: If a user-provided vector has the wrong shape, contains
            non-finite values, or is numerically zero.
    """

    if v0 is None:
        return mx.ones((int(n),), dtype=mx.float32)
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
