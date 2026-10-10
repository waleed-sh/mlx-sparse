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
from mlx_sparse.linalg.utils.spectral import normalize_which as _normalize_which
from mlx_sparse.linalg.utils.spectral import (
    reject_iteration_controls as _reject_controls,
)
from mlx_sparse.linalg.utils.spectral import start_vector as _start_vector


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
        v0: Optional finite, nonzero starting vector of shape ``(n,)``.
            ``None`` uses standard normal samples from a fixed key without
            consuming the global random stream. Explicit vectors retain
            their direction and are normalized in native code.
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
        returns ``(alphas, betas)`` without the basis. If the recurrence
        reaches an invariant subspace before ``k`` steps, unused coefficients
        and basis columns are zero. The last nonzero basis column ends the
        valid factorization.

    Raises:
        ValueError: If ``k`` is out of range or ``v0`` has the wrong shape,
            contains non-finite values, or is zero after float32 conversion.
    """

    csr = _float32_csr(_as_csr(A))
    if k <= 0 or k > csr.shape[0]:
        raise ValueError("k must satisfy 0 < k <= A.shape[0].")
    start = _start_vector(v0, n=csr.shape[0])
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

            Selectors are case-insensitive. Other values, including ``"BE"``,
            raise ``ValueError``.

        v0: Optional finite, nonzero starting vector of shape ``(n,)``.
            ``None`` uses standard normal samples from a fixed key without
            consuming the global random stream. Explicit vectors retain
            their direction and are normalized in native code.
        ncv: Number of Lanczos basis vectors to build before extracting
            Ritz pairs.  A larger value improves accuracy at the cost of
            more memory. Defaults to ``max(2*k+1, k+1)``. After invariant
            breakdown, the recurrence continues from an orthogonal random
            direction within this budget.
        maxiter: Not yet supported because the current implementation performs
            one ``ncv``-bounded Ritz extraction, not an implicitly restarted
            convergence loop.  Pass ``None`` (the default).
        tol: Not yet supported for the same reason.  Pass ``0.0`` (the
            default).
        return_eigenvectors: When ``True`` (the default), return both
            eigenvalues and eigenvectors.  When ``False``, return only the
            eigenvalues.

    Returns:
        When ``return_eigenvectors=True``, a tuple ``(values, vectors)``
        where ``values`` has shape ``(k,)`` and ``vectors`` has shape
        ``(n, k)``.  When ``return_eigenvectors=False``, returns ``values``
        alone.

    Raises:
        NotImplementedError: If ``maxiter`` or ``tol`` are not at their default
            values.
        ValueError: If ``which`` is not a supported string, ``k`` is out of
            range, ``A`` is not square, or ``v0`` is invalid.
        RuntimeError: If orthogonal continuation fails or the recurrence
            produces fewer than ``k`` independent Ritz candidates.
    """

    _reject_controls(routine="eigsh", tol=float(tol), maxiter=maxiter)
    which = _normalize_which(which, routine="eigsh", allowed=("LM", "SM", "LA", "SA"))
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
        _start_vector(v0, n=n),
        csr.shape,
        k=int(k),
        ncv=_ncv(n, int(k), ncv),
        which=which,
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

    Uses the native CSR Arnoldi-based eigensolver to find the ``k`` eigenpairs
    of the general, possibly non-symmetric sparse matrix ``A`` that match
    ``which``. Each Arnoldi step uses a native sparse matrix-vector product
    on the selected device.

    For symmetric matrices, :func:`eigsh` is faster and more accurate because
    it uses the symmetric Lanczos recurrence instead of the full Arnoldi
    factorization.

    GPU note:
        When GPU execution is selected, Arnoldi factorisation uses the native
        Arnoldi kernel. The small Hessenberg eigensolve runs on the CPU.
        Ritz vectors are reconstructed as ``Q @ y`` on the selected device.
        The Arnoldi basis is not copied to the host for reconstruction.

    Args:
        A: Real sparse square matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray`.
            Float16 and bfloat16 inputs are promoted to float32.
        k: Number of eigenpairs to compute.  Must satisfy
            ``0 < k < A.shape[0]``.  Defaults to ``6``.
        which: Which eigenpairs to return.  Accepted values:

            * ``"LM"``: eigenvalues Largest in Magnitude (default)
            * ``"SM"``: eigenvalues Smallest in Magnitude
            * ``"LR"``: Largest Real part
            * ``"SR"``: Smallest Real part

            Selectors are case-insensitive. Other values, including ``"LI"``
            and ``"SI"``, raise ``ValueError``.

        v0: Optional finite, nonzero starting vector of shape ``(n,)``.
            ``None`` uses standard normal samples from a fixed key without
            consuming the global random stream. Explicit vectors retain
            their direction and are normalized in native code.
        ncv: Dimension of the Arnoldi factorization before Ritz extraction.
            Defaults to ``max(2*k+1, k+1)``. After invariant breakdown, the
            recurrence continues from an orthogonal random direction within
            this budget.
        maxiter: Not yet supported because the current implementation performs
            one ``ncv``-bounded Ritz extraction, not an implicitly restarted
            convergence loop.  Pass ``None`` (the default).
        tol: Not yet supported for the same reason.  Pass ``0.0`` (the
            default).
        return_eigenvectors: When ``True`` (the default), return both
            eigenvalues and eigenvectors.  When ``False``, return only the
            eigenvalues.

    Returns:
        When ``return_eigenvectors=True``, a tuple ``(values, vectors)``
        where ``values`` has shape ``(k,)`` and ``vectors`` has shape
        ``(n, k)``.  When ``return_eigenvectors=False``, returns ``values``
        alone. Values and vectors have dtype ``complex64``, including when
        all selected eigenvalues are real. Vector columns have unit norm.
        General eigenvectors need not be orthogonal and a defective matrix
        need not have a complete set of independent eigenvectors.

        These are Ritz approximations from one Arnoldi factorization.
        Increase ``ncv`` and check ``A @ vectors - vectors * values`` when
        accuracy matters. A small basis does not guarantee convergence,
        especially for interior eigenvalues or nonnormal matrices.

    Raises:
        NotImplementedError: If ``maxiter`` or ``tol`` are not at their default
            values.
        ValueError: If ``which`` is not a supported string, ``k`` is out of
            range, ``A`` is not square, or ``v0`` is invalid.
        RuntimeError: If orthogonal continuation fails or the recurrence
            produces fewer than ``k`` independent Ritz candidates.
    """

    _reject_controls(routine="eigs", tol=float(tol), maxiter=maxiter)
    which = _normalize_which(which, routine="eigs", allowed=("LM", "SM", "LR", "SR"))
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
        _start_vector(v0, n=n),
        csr.shape,
        k=int(k),
        ncv=_ncv(n, int(k), ncv),
        which=which,
        compute_vectors=bool(return_eigenvectors),
    )
    return (values, vectors) if return_eigenvectors else values


def svds(
    A,
    k: int = 6,
    *,
    which: str = "LM",
    v0=None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0.0,
    return_singular_vectors: bool | str = True,
):
    """Compute a few singular triplets of a sparse matrix.

    Builds fully reorthogonalized left and right bases with native
    Golub-Kahan bidiagonalization and computes the SVD of their small
    projection. The method applies ``A`` and ``A.T`` directly without forming
    a normal matrix or squaring the singular values. Both vector families
    are reconstructed from orthonormal bases, including zero singular values.

    GPU note:
        Small problems use a fused native Metal recurrence. Larger problems
        use parallel native sparse products and two-pass basis projections,
        with scalar breakdown checks on the host.
        The projection and singular-vector reconstruction run on the selected
        device. Only the small projected matrix is passed to the CPU SVD.
        Unrequested vector families are not reconstructed.

    Args:
        A: Real sparse matrix of shape ``(m, n)``.  Must be a
            :class:`~mlx_sparse.CSRArray`, :class:`~mlx_sparse.COOArray`, or
            :class:`~mlx_sparse.CSCArray`.  Float16 and bfloat16 inputs are
            promoted to float32.
        k: Number of singular triplets to compute.  Must satisfy
            ``0 < k < min(A.shape)``.  Defaults to ``6``.
        which: Which singular values to return.  Accepted values:

            * ``"LM"``: Largest in Magnitude (default)
            * ``"SM"``: Smallest in Magnitude

            Selectors are case-insensitive. Other values raise ``ValueError``.

        v0: Optional finite, nonzero starting vector for the right
            singular-vector basis, with shape ``(A.shape[1],)``. ``None`` uses
            standard normal samples from a fixed key without consuming the
            global random stream. Explicit vectors retain their direction
            and are normalized in native code.
        ncv: Number of left basis vectors to build, capped at
            ``min(A.shape)``. Defaults to ``max(2*k+1, k+1)``. The right basis
            retains one additional vector when the column dimension permits
            it. After breakdown on either side, the recurrence continues
            from an orthogonal random direction within this budget.
        maxiter: Not yet supported because the current implementation performs
            one ``ncv``-bounded projected SVD. Pass ``None`` (the default).
        tol: Not yet supported for the same reason.  Pass ``0.0`` (the
            default).
        return_singular_vectors: Controls which vectors are returned.

            * ``True``: return ``(u, s, vh)`` (default)
            * ``False``: return only ``s`` (the singular values)
            * ``"u"``: return ``(u, s, None)``
            * ``"vh"``: return ``(None, s, vh)``

    Returns:
        When ``return_singular_vectors=True``, a tuple ``(u, s, vh)`` where
        ``u`` has shape ``(m, k)``, ``s`` has shape ``(k,)``, and ``vh`` has
        shape ``(k, n)``. All arrays have dtype ``float32``. Singular values
        are nonnegative, ordered largest first for ``"LM"`` and smallest
        first for ``"SM"``. Both vector families are orthonormal to working
        precision. See ``return_singular_vectors`` for other forms.

        Results approximate the singular triplets of ``A``. Check both
        ``A @ vh.T - u * s`` and ``A.T @ u - vh.T * s`` when accuracy matters.
        Increase ``ncv`` if residuals are too large. A small basis does not
        guarantee convergence, particularly for the smallest singular values.
        Near rank deficiency, singular values below float32 resolution can
        have large relative error. Bases within repeated or zero singular
        subspaces are not unique. A near-zero projected value can have a
        nonzero full-matrix residual and does not establish numerical rank.

    Raises:
        NotImplementedError: If ``maxiter`` or ``tol`` are not at their default
            values.
        ValueError: If ``which`` is not a supported string, ``k`` is out of
            range, ``v0`` is invalid, or ``return_singular_vectors`` is not
            a recognised value.
        RuntimeError: If orthogonal continuation fails or the recurrence
            produces fewer than ``k`` independent candidates.
    """

    _reject_controls(routine="svds", tol=float(tol), maxiter=maxiter)
    which = _normalize_which(which, routine="svds", allowed=("LM", "SM"))
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
        _start_vector(v0, n=csr.shape[1]),
        csr.shape,
        k=int(k),
        ncv=_ncv(limit, int(k), ncv),
        which=which,
        compute_u=return_singular_vectors in {True, "u"},
        compute_v=return_singular_vectors in {True, "vh"},
    )
    if return_singular_vectors is False:
        return singular
    if return_singular_vectors == "u":
        return left, singular, None
    if return_singular_vectors == "vh":
        return None, singular, vh
    return left, singular, vh
