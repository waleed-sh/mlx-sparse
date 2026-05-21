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
from mlx_sparse._coo import COOArray
from mlx_sparse._csr import CSRArray


def _as_csr(A) -> CSRArray:
    if isinstance(A, CSRArray):
        return A.canonicalize()
    if isinstance(A, COOArray):
        return A.tocsr(canonical=True)
    raise TypeError(
        "sparse eigen routines expect CSRArray or COOArray. Dense arrays belong "
        "in mlx.linalg."
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
    raise TypeError("sparse spectral routines currently require real float data.")


def _ncv(n: int, k: int, ncv: int | None) -> int:
    return min(n, max(k + 1, 2 * k + 1 if ncv is None else int(ncv)))


def lanczos(
    A,
    k: int,
    *,
    v0=None,
    reorthogonalize: bool = True,
    return_basis: bool = True,
):
    """Run native CSR Lanczos and return tridiagonal coefficients/basis."""

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
    """Selected Hermitian sparse eigenpairs from the native CSR solver."""

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
    """Selected sparse Arnoldi Ritz pairs from the native CSR solver."""

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
    """Selected sparse singular triplets from native CSR normal-operator Lanczos."""

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
