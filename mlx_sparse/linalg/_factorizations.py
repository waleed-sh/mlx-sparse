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
    factorization is performed by the native sparse left-looking routine and the solves
    use sparse triangular CSR kernels.
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


def sparse_cholesky(A, *, upper: bool = False) -> SparseCholesky:
    """Compute the sparse Cholesky factorization ``A = L @ L.T``.

    Performs a left-looking sparse Cholesky factorization on a real symmetric
    positive-definite (SPD) matrix stored in CSR or COO format.  The resulting
    lower-triangular factor ``L`` is returned as a :class:`SparseCholesky`
    object whose :meth:`~SparseCholesky.solve` method applies both triangular
    solves in sequence.

    The factorization step runs on CPU using the native sparse routine.  The
    resulting ``SparseCholesky.solve`` dispatches triangular-solve kernels to
    the GPU when :func:`~mlx_sparse.use_gpu` has been called.

    Args:
        A: The matrix to factorize.  Must be a :class:`~mlx_sparse.CSRArray`
            or :class:`~mlx_sparse.COOArray` that is real, symmetric, and
            positive-definite.  Float16 and bfloat16 inputs are promoted to
            float32 automatically.
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

    Args:
        A: SPD matrix in :class:`~mlx_sparse.CSRArray` or
            :class:`~mlx_sparse.COOArray` format.
        upper: Not yet supported.  Must be ``False`` (the default).

    Returns:
        A :class:`SparseCholesky` holding the lower-triangular factor ``L``.
    """
    return sparse_cholesky(A, upper=upper)


def sparse_lu(A) -> SparseLU:
    """Compute the sparse LU factorization ``P @ A = L @ U``.

    Performs a sparse LU factorization with partial pivoting on a general
    (possibly non-symmetric) real square matrix stored in CSR or COO format.
    The row permutation ``P``, unit lower-triangular factor ``L``, and
    upper-triangular factor ``U`` are returned as a :class:`SparseLU` object
    whose :meth:`~SparseLU.solve` method applies the full solve sequence.

    The factorization step runs on CPU using the native sparse routine.  The
    resulting ``SparseLU.solve`` dispatches permutation and triangular-solve
    kernels to the GPU when :func:`~mlx_sparse.use_gpu` has been called.

    Args:
        A: The matrix to factorize.  Must be a :class:`~mlx_sparse.CSRArray`
            or :class:`~mlx_sparse.COOArray` that is real and non-singular.
            Float16 and bfloat16 inputs are promoted to float32 automatically.

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

    Args:
        A: Matrix to factorize in :class:`~mlx_sparse.CSRArray` or
            :class:`~mlx_sparse.COOArray` format.

    Returns:
        A :class:`SparseLU` dataclass with fields ``perm``, ``L``, and ``U``.
    """
    return sparse_lu(A)


def spsolve(A, b) -> mx.array:
    """Solve the sparse linear system ``A @ x = b`` directly.

    Computes a sparse LU factorization of ``A`` via :func:`sparse_lu` and
    immediately applies :meth:`~SparseLU.solve` to ``b``.  This is a
    convenience wrapper equivalent to ``sparse_lu(A).solve(b)``.

    For repeated solves with the same ``A`` but different right-hand sides,
    call :func:`sparse_lu` once and reuse the resulting :class:`SparseLU`
    object to avoid re-factorizing.

    Args:
        A: Coefficient matrix.  Must be a :class:`~mlx_sparse.CSRArray` or
            :class:`~mlx_sparse.COOArray` that is real and non-singular.
        b: Right-hand side vector of shape ``(n,)``.

    Returns:
        Solution vector ``x`` of shape ``(n,)`` as an ``mlx.core.array``.

    Raises:
        TypeError: If ``A`` is a dense array or has an unsupported dtype.
        NotImplementedError: If ``b`` is not rank-1.
    """
    return sparse_lu(A).solve(b)
