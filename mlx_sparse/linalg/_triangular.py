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

"""Public sparse triangular solve wrappers."""

from __future__ import annotations

import mlx.core as mx

import mlx_sparse._native as _native
from mlx_sparse.linalg.utils.arrays import ensure_rank1_or_rank2_rhs
from mlx_sparse.linalg.utils.factorization import as_csr as _as_csr
from mlx_sparse.linalg.utils.factorization import float32_csr as _float32_csr


def spsolve_triangular(
    A,
    b,
    *,
    lower: bool = True,
    unit_diagonal: bool = False,
    analyzed=None,
) -> mx.array:
    """Solve a sparse triangular system with the native CSR kernel.

    This function solves ``A @ x = b`` for a square sparse triangular matrix
    ``A``.  Inputs are normalized to canonical CSR and float32 values before
    dispatching to the native ``csr_triangular_solve`` primitive.  The returned
    solution has the same rank as ``b``: rank-1 vector RHS inputs produce a
    rank-1 solution, and rank-2 matrix RHS inputs produce a rank-2 solution.

    The optional ``analyzed`` argument is reserved for a future public
    triangular-analysis object.  v0.0.5b0 keeps analysis private because the
    repeated-apply benchmark does not yet show a consistent advantage over the
    default native row-order solve.

    Args:
        A: Square sparse triangular matrix.  ``CSRArray``, ``COOArray``,
            ``CSCArray``, and sparse-backed ``LinearOperator`` inputs are
            accepted and converted to canonical CSR without mutating ``A``.
        b: Right-hand side with shape ``(n,)`` or ``(n, nrhs)``.
        lower: When ``True`` (the default), treat ``A`` as lower triangular.
            When ``False``, treat it as upper triangular.
        unit_diagonal: When ``True``, diagonal entries are assumed to be one and
            are not read from ``A``.
        analyzed: Reserved for a future public analysis object.  Must be
            ``None`` in this release.

    Returns:
        The native triangular-solve result as a finite float32 MLX array.

    Raises:
        TypeError: If the sparse data or RHS dtype is unsupported.
        ValueError: If ``A`` is not square or if ``b`` has an incompatible
            shape.
        NotImplementedError: If ``analyzed`` is not ``None``.
    """

    if analyzed is not None:
        raise NotImplementedError(
            "public triangular analysis is not exposed yet; pass analyzed=None."
        )
    csr = _float32_csr(_as_csr(A))
    if csr.shape[0] != csr.shape[1]:
        raise ValueError(
            f"spsolve_triangular requires a square matrix, got {csr.shape}."
        )
    rhs = ensure_rank1_or_rank2_rhs(
        b,
        leading_dim=csr.shape[0],
        dtype=mx.float32,
        require_finite=True,
    )
    return _native.csr_triangular_solve(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        csr.shape,
        lower=bool(lower),
        unit_diagonal=bool(unit_diagonal),
    )
