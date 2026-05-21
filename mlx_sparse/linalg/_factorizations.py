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

import mlx_sparse._native as _native
from mlx_sparse._coo import COOArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._validation import ensure_mx_array


def _as_csr(A) -> CSRArray:
    if isinstance(A, CSRArray):
        return A.canonicalize()
    if isinstance(A, COOArray):
        return A.tocsr(canonical=True)
    raise TypeError(
        "sparse factorization expects CSRArray or COOArray. "
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


@dataclass(frozen=True, slots=True)
class SparseCholesky:
    """Sparse Cholesky factorization ``A = L @ L.T``.

    The factor ``L`` is stored as a :class:`mlx_sparse.CSRArray`. Numeric
    factorization is performed by the native sparse left-looking routine, solves
    use sparse triangular CSR kernels.
    """

    L: CSRArray

    @property
    def shape(self) -> tuple[int, int]:
        return self.L.shape

    def solve(self, b) -> mx.array:
        y = _triangular_solve(self.L, b, lower=True, unit_diagonal=False)
        return _triangular_solve(self.L.T, y, lower=False, unit_diagonal=False)

    def __call__(self, b) -> mx.array:
        return self.solve(b)


@dataclass(frozen=True, slots=True)
class SparseLU:
    """Sparse LU factorization ``P @ A = L @ U``.

    ``L`` and ``U`` are CSR sparse factors. ``perm`` stores the row permutation
    applied before factorization.
    """

    perm: mx.array
    L: CSRArray
    U: CSRArray

    @property
    def shape(self) -> tuple[int, int]:
        return self.L.shape

    def solve(self, b) -> mx.array:
        rhs = ensure_mx_array(b, dtype=mx.float32)
        if rhs.ndim != 1:
            raise NotImplementedError("SparseLU.solve currently accepts rank-1 RHS.")
        permuted = _native.csr_permute_vector(rhs, self.perm)
        y = _triangular_solve(self.L, permuted, lower=True, unit_diagonal=True)
        return _triangular_solve(self.U, y, lower=False, unit_diagonal=False)

    def __call__(self, b) -> mx.array:
        return self.solve(b)


def sparse_cholesky(A, *, upper: bool = False) -> SparseCholesky:
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
    return sparse_cholesky(A, upper=upper)


def sparse_lu(A) -> SparseLU:
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
    return sparse_lu(A)


def spsolve(A, b) -> mx.array:
    return sparse_lu(A).solve(b)
