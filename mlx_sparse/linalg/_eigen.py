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
from mlx_sparse.linalg.utils.spectral import as_csr as _as_csr
from mlx_sparse.linalg.utils.spectral import float32_csr as _float32_csr
from mlx_sparse.linalg.utils.spectral import normalize_ncv as _ncv


def lanczos(
    A,
    k: int,
    *,
    v0=None,
    reorthogonalize: bool = True,
    return_basis: bool = True,
):
    """Run the Lanczos iteration on a sparse symmetric matrix.

    Builds a Krylov basis of dimension ``k`` for the symmetric matrix ``A``
    using the Lanczos three-term recurrence.  The basis vectors, together
    with the tridiagonal matrix they define, contain the information needed
    to approximate eigenvalues and eigenvectors via the Ritz pairs of ``A``.

    This is a low-level routine.  For eigenvalue computation, prefer
    :func:`eigsh` which calls Lanczos internally and returns the final
    eigenpairs directly.

    GPU note:
        When GPU execution is selected, the Lanczos recurrence runs in a
        Metal kernel.  Sparse matrix-vector products, orthogonalisation,
        tridiagonal coefficients, and basis writes stay on the GPU.  Python
        argument validation and returned array handling happen on the host.

    Args:
        A: Symmetric sparse matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray`.
            Float16 and bfloat16 inputs are promoted to float32.
        k: Number of Lanczos steps.  Must satisfy ``0 < k <= A.shape[0]``.
        v0: Not yet supported.  Pass ``None`` (the default).
        reorthogonalize: Whether to apply full reorthogonalisation at each
            step to suppress numerical loss of orthogonality.  Defaults to
            ``True``.
        return_basis: When ``True`` (the default), return the Lanczos basis
            matrix ``Q`` in addition to the tridiagonal coefficients.

    Returns:
        When ``return_basis=True``, a tuple ``(alphas, betas, Q)`` where
        ``alphas`` is the diagonal of shape ``(k,)``, ``betas`` is the
        sub-diagonal of shape ``(k-1,)`` or ``(k,)``, and ``Q`` is the
        basis matrix of shape ``(n, k)``.  When ``return_basis=False``,
        returns ``(alphas, betas)`` without the basis.

    Raises:
        NotImplementedError: If ``v0`` is not ``None``.
        ValueError: If ``k`` is out of range.
    """

    if v0 is not None:
        raise NotImplementedError("native lanczos currently owns its start vector.")
    csr = _float32_csr(_as_csr(A))
    if k <= 0 or k > csr.shape[0]:
        raise ValueError("k must satisfy 0 < k <= A.shape[0].")
    start = mx.ones((csr.shape[0],), dtype=mx.float32)
    alphas, betas, basis, _ = _native.csr_lanczos(
        csr.data,
        csr.indices,
        csr.indptr,
        start,
        csr.shape,
        k=int(k),
        reorthogonalize=bool(reorthogonalize),
    )
    if return_basis:
        return alphas, betas, basis
    return alphas, betas


def eigsh(
    A,
    k: int = 6,
    *,
    which: str = "LM",
    v0=None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0.0,
    return_eigenvectors: bool = True,
):
    """Compute a few eigenpairs of a sparse symmetric matrix.

    Uses the native CSR Lanczos-based eigensolver to find the ``k`` eigenpairs
    of the real symmetric (Hermitian) sparse matrix ``A`` that match the
    criterion specified by ``which``.  Each Lanczos step dispatches a sparse
    matrix-vector product to the GPU via the native Metal kernel.

    GPU note:
        When GPU execution is selected, Lanczos tridiagonalisation uses the
        native Lanczos kernel.  The small tridiagonal eigensolve, Ritz pair
        selection, and eigenvector back transformation run on the CPU after
        the basis and coefficients are copied back to host memory.

    Args:
        A: Real symmetric sparse matrix.  Must be a
            :class:`~mlx_sparse.CSRArray`, :class:`~mlx_sparse.COOArray`, or
            :class:`~mlx_sparse.CSCArray`.  Float16 and bfloat16 inputs are
            promoted to float32.
        k: Number of eigenpairs to compute.  Must satisfy
            ``0 < k < A.shape[0]``.  Defaults to ``6``.
        which: Which eigenpairs to return.  Accepted values:

            * ``"LM"``: eigenvalues Largest in Magnitude (default)
            * ``"SM"``: eigenvalues Smallest in Magnitude
            * ``"LA"``: Largest Algebraic (largest values)
            * ``"SA"``: Smallest Algebraic (smallest values)

        v0: Not yet supported.  Pass ``None`` (the default).
        ncv: Number of Lanczos basis vectors to build before extracting
            Ritz pairs.  A larger value improves accuracy at the cost of
            more memory.  Defaults to ``max(2*k+1, k+1)``.
        maxiter: Not yet supported.  Pass ``None`` (the default).
        tol: Not yet supported.  Pass ``0.0`` (the default).
        return_eigenvectors: When ``True`` (the default), return both
            eigenvalues and eigenvectors.  When ``False``, return only the
            eigenvalues.

    Returns:
        When ``return_eigenvectors=True``, a tuple ``(values, vectors)``
        where ``values`` has shape ``(k,)`` and ``vectors`` has shape
        ``(n, k)``.  When ``return_eigenvectors=False``, returns ``values``
        alone.

    Raises:
        NotImplementedError: If ``v0``, ``maxiter``, or ``tol`` are not at
            their default values.
        ValueError: If ``k`` is out of range or ``A`` is not square.
    """

    if v0 is not None or maxiter is not None or tol != 0.0:
        raise NotImplementedError(
            "native eigsh currently controls start vector, iteration count, and tolerance."
        )
    csr = _float32_csr(_as_csr(A))
    n = csr.shape[0]
    if csr.shape[0] != csr.shape[1]:
        raise ValueError(f"eigsh requires a square matrix, got {csr.shape}.")
    if k <= 0 or k >= n:
        raise ValueError("k must satisfy 0 < k < A.shape[0].")
    values, vectors = _native.csr_eigsh(
        csr.data,
        csr.indices,
        csr.indptr,
        csr.shape,
        k=int(k),
        ncv=_ncv(n, int(k), ncv),
        which=which.upper(),
    )
    return (values, vectors) if return_eigenvectors else values


def eigs(
    A,
    k: int = 6,
    *,
    which: str = "LM",
    v0=None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0.0,
    return_eigenvectors: bool = True,
):
    """Compute a few eigenpairs of a general sparse square matrix.

    Uses the native CSR Arnoldi-based eigensolver (an implicitly restarted
    Arnoldi method) to find the ``k`` eigenpairs of the general (possibly
    non-symmetric) sparse matrix ``A`` that match the criterion specified by
    ``which``.  Each Arnoldi step dispatches a sparse matrix-vector product
    to the GPU via the native Metal kernel.

    For symmetric matrices, :func:`eigsh` is faster and more accurate because
    it uses the symmetric Lanczos recurrence instead of the full Arnoldi
    factorization.

    GPU note:
        When GPU execution is selected, Arnoldi factorisation uses the native
        Arnoldi kernel.  The small Hessenberg eigensolve, Ritz value
        selection, and output vector assembly run on the CPU after the basis
        and Hessenberg matrix are copied back to host memory.

    Args:
        A: Sparse square matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray`.
            Float16 and bfloat16 inputs are promoted to float32.
        k: Number of eigenpairs to compute.  Must satisfy
            ``0 < k < A.shape[0]``.  Defaults to ``6``.
        which: Which eigenpairs to return.  Accepted values:

            * ``"LM"``: eigenvalues Largest in Magnitude (default)
            * ``"SM"``: eigenvalues Smallest in Magnitude
            * ``"LR"``: Largest Real part
            * ``"SR"``: Smallest Real part

        v0: Not yet supported.  Pass ``None`` (the default).
        ncv: Dimension of the Arnoldi factorization before restart.
            Defaults to ``max(2*k+1, k+1)``.
        maxiter: Not yet supported.  Pass ``None`` (the default).
        tol: Not yet supported.  Pass ``0.0`` (the default).
        return_eigenvectors: When ``True`` (the default), return both
            eigenvalues and eigenvectors.  When ``False``, return only the
            eigenvalues.

    Returns:
        When ``return_eigenvectors=True``, a tuple ``(values, vectors)``
        where ``values`` has shape ``(k,)`` and ``vectors`` has shape
        ``(n, k)``.  When ``return_eigenvectors=False``, returns ``values``
        alone.

    Raises:
        NotImplementedError: If ``v0``, ``maxiter``, or ``tol`` are not at
            their default values.
        ValueError: If ``k`` is out of range or ``A`` is not square.
    """

    if v0 is not None or maxiter is not None or tol != 0.0:
        raise NotImplementedError(
            "native eigs currently controls start vector, iteration count, and tolerance."
        )
    csr = _float32_csr(_as_csr(A))
    n = csr.shape[0]
    if csr.shape[0] != csr.shape[1]:
        raise ValueError(f"eigs requires a square matrix, got {csr.shape}.")
    if k <= 0 or k >= n:
        raise ValueError("k must satisfy 0 < k < A.shape[0].")
    values, vectors = _native.csr_eigs(
        csr.data,
        csr.indices,
        csr.indptr,
        csr.shape,
        k=int(k),
        ncv=_ncv(n, int(k), ncv),
        which=which.upper(),
    )
    return (values, vectors) if return_eigenvectors else values


def svds(
    A,
    k: int = 6,
    *,
    which: str = "LM",
    ncv: int | None = None,
    tol: float = 0.0,
    return_singular_vectors: bool | str = True,
):
    """Compute a few singular triplets of a sparse matrix.

    Uses the native CSR Lanczos iteration applied to the normal operator
    ``A.T @ A`` to find the ``k`` singular triplets (left singular vectors,
    singular values, and right singular vectors) of the sparse matrix ``A``
    that match the criterion specified by ``which``.

    GPU note:
        When GPU execution is selected, the normal-operator Lanczos recurrence
        uses a dedicated native ``A.T @ (A @ v)`` path.  The two sparse
        products are kept inside one native step and the intermediate
        ``A @ v`` vector is not materialized on the host.  The small
        tridiagonal eigensolve, Ritz vector back transformation, and returned
        singular-vector assembly still run on the CPU after the Lanczos basis
        is synchronized.

    Args:
        A: Sparse matrix of shape ``(m, n)``.  Must be a
            :class:`~mlx_sparse.CSRArray`, :class:`~mlx_sparse.COOArray`, or
            :class:`~mlx_sparse.CSCArray`.  Float16 and bfloat16 inputs are
            promoted to float32.
        k: Number of singular triplets to compute.  Must satisfy
            ``0 < k < min(A.shape)``.  Defaults to ``6``.
        which: Which singular values to return.  Accepted values:

            * ``"LM"``: Largest in Magnitude (default)
            * ``"SM"``: Smallest in Magnitude

        ncv: Number of Lanczos basis vectors to build.  Defaults to
            ``max(2*k+1, k+1)``.
        tol: Not yet supported.  Pass ``0.0`` (the default).
        return_singular_vectors: Controls which vectors are returned.

            * ``True``: return ``(u, s, vh)`` (default)
            * ``False``: return only ``s`` (the singular values)
            * ``"u"``: return ``(u, s, None)``
            * ``"vh"``: return ``(None, s, vh)``

    Returns:
        When ``return_singular_vectors=True``, a tuple ``(u, s, vh)`` where
        ``u`` has shape ``(m, k)``, ``s`` has shape ``(k,)``, and ``vh`` has
        shape ``(k, n)``.  See ``return_singular_vectors`` for other forms.

    Raises:
        NotImplementedError: If ``tol`` is not ``0.0``.
        ValueError: If ``k`` is out of range or ``return_singular_vectors``
            is not a recognised value.
    """

    if tol != 0.0:
        raise NotImplementedError("native svds currently controls tolerance.")
    if return_singular_vectors not in {True, False, "u", "vh"}:
        raise ValueError("return_singular_vectors must be True, False, 'u', or 'vh'.")
    csr = _float32_csr(_as_csr(A))
    limit = min(csr.shape)
    if k <= 0 or k >= limit:
        raise ValueError("k must satisfy 0 < k < min(A.shape).")
    left, singular, vh = _native.csr_svds(
        csr.data,
        csr.indices,
        csr.indptr,
        csr.shape,
        k=int(k),
        ncv=_ncv(csr.shape[1], int(k), ncv),
        which=which.upper(),
    )
    if return_singular_vectors is False:
        return singular
    if return_singular_vectors == "u":
        return left, singular, None
    if return_singular_vectors == "vh":
        return None, singular, vh
    return left, singular, vh
