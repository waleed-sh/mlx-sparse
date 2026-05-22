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

from mlx_sparse._coo import COOArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._validation import normalize_shape
from mlx_sparse.linalg._utils import ensure_array

from .._typing import Matmat, Matvec


class LinearOperator:
    """Sparse/matrix-free operator interface.

    This class stores callables only. It does not densify an operator or provide
    dense fallbacks. Sparse arrays are wrapped with native CSR matvec/matmat
    kernels through :func:`aslinearoperator`.
    """

    __slots__ = (
        "shape",
        "matvec_fn",
        "dtype",
        "matmat_fn",
        "rmatvec_fn",
        "_sparse_array",
    )

    def __init__(
        self,
        shape,
        matvec: Matvec | None = None,
        *,
        matvec_fn: Matvec | None = None,
        dtype=None,
        matmat: Matmat | None = None,
        matmat_fn: Matmat | None = None,
        rmatvec: Matvec | None = None,
        rmatvec_fn: Matvec | None = None,
        _sparse_array=None,
    ) -> None:
        matvec_impl = matvec_fn if matvec_fn is not None else matvec
        if matvec_impl is None:
            raise TypeError("LinearOperator requires a matvec callable.")
        self.shape = normalize_shape(shape)
        self.matvec_fn = matvec_impl
        self.dtype = dtype
        self.matmat_fn = matmat_fn if matmat_fn is not None else matmat
        self.rmatvec_fn = rmatvec_fn if rmatvec_fn is not None else rmatvec
        self._sparse_array = _sparse_array

    @property
    def ndim(self) -> int:
        return 2

    def matvec(self, x) -> mx.array:
        x = ensure_array(x)
        if x.ndim != 1:
            raise ValueError(f"matvec expects rank-1 input, got shape={x.shape}.")
        if x.shape[0] != self.shape[1]:
            raise ValueError(
                f"matvec input has length {x.shape[0]}, expected {self.shape[1]}."
            )
        return self.matvec_fn(x)

    def matmat(self, X) -> mx.array:
        X = ensure_array(X)
        if X.ndim != 2:
            raise ValueError(f"matmat expects rank-2 input, got shape={X.shape}.")
        if X.shape[0] != self.shape[1]:
            raise ValueError(
                f"matmat input has leading dimension {X.shape[0]}, "
                f"expected {self.shape[1]}."
            )
        if self.matmat_fn is None:
            raise NotImplementedError("matmat is not defined for this operator.")
        return self.matmat_fn(X)

    def rmatvec(self, x) -> mx.array:
        x = ensure_array(x)
        if x.ndim != 1:
            raise ValueError(f"rmatvec expects rank-1 input, got shape={x.shape}.")
        if x.shape[0] != self.shape[0]:
            raise ValueError(
                f"rmatvec input has length {x.shape[0]}, expected {self.shape[0]}."
            )
        if self.rmatvec_fn is None:
            raise NotImplementedError("rmatvec is not defined for this operator.")
        return self.rmatvec_fn(x)

    def __matmul__(self, rhs):
        rhs = ensure_array(rhs)
        if rhs.ndim == 1:
            return self.matvec(rhs)
        if rhs.ndim == 2:
            return self.matmat(rhs)
        raise ValueError(
            f"LinearOperator matmul expects rank-1 or rank-2 RHS, got {rhs.shape}."
        )

    @property
    def T(self) -> "LinearOperator":
        """Transpose operator.  ``(op.T) @ x`` computes ``A.T @ x``.

        For real operators ``A.T == A.H``, for complex operators the formula
        ``A.T @ x = conj(A.H @ conj(x))`` is used so no extra kernel is
        needed.  Requires :attr:`rmatvec_fn` to be defined.
        """
        if self.rmatvec_fn is None:
            raise NotImplementedError(
                "LinearOperator.T requires rmatvec to be defined."
            )
        # A.T @ x = conj( rmatvec( conj(x) ) )
        # For real dtypes mx.conjugate is a no-op in values, so this is exact.
        _rv = self.rmatvec_fn
        _mv = self.matvec_fn
        sparse = (
            self._sparse_array.transpose() if self._sparse_array is not None else None
        )
        return LinearOperator(
            shape=(self.shape[1], self.shape[0]),
            matvec_fn=lambda x: mx.conjugate(_rv(mx.conjugate(x))),
            rmatvec_fn=lambda x: mx.conjugate(_mv(mx.conjugate(x))),
            dtype=self.dtype,
            _sparse_array=sparse,
        )

    @property
    def H(self) -> "LinearOperator":
        """Hermitian (conjugate) transpose operator.  ``(op.H) @ x`` computes
        ``A.H @ x``.

        Requires :attr:`rmatvec_fn` to be defined (which stores ``A.H``).
        The double adjoint ``(A.H).H`` recovers the original ``A``.
        """
        if self.rmatvec_fn is None:
            raise NotImplementedError(
                "LinearOperator.H requires rmatvec to be defined."
            )
        _rv = self.rmatvec_fn
        _mv = self.matvec_fn
        sparse = self._sparse_array.H if self._sparse_array is not None else None
        return LinearOperator(
            shape=(self.shape[1], self.shape[0]),
            matvec_fn=_rv,
            rmatvec_fn=_mv,
            dtype=self.dtype,
            _sparse_array=sparse,
        )


def _sparse_operator(array: CSRArray | COOArray) -> LinearOperator:
    csr = array.tocsr(canonical=True) if isinstance(array, COOArray) else array
    adjoint = csr.H
    return LinearOperator(
        shape=csr.shape,
        matvec_fn=lambda x: csr @ x,
        matmat_fn=lambda X: csr @ X,
        rmatvec_fn=lambda x: adjoint @ x,
        dtype=csr.dtype,
        _sparse_array=csr,
    )


def aslinearoperator(A) -> LinearOperator:
    """Return a sparse object or explicit callable as a :class:`LinearOperator`."""

    if isinstance(A, LinearOperator):
        return A
    if isinstance(A, CSRArray | COOArray):
        return _sparse_operator(A)
    if isinstance(A, tuple) and len(A) >= 2:
        shape, matvec = A[:2]
        matmat = A[2] if len(A) > 2 else None
        return LinearOperator(shape=tuple(shape), matvec_fn=matvec, matmat_fn=matmat)
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None
    if sp is not None and sp.issparse(A):
        from mlx_sparse._construct import from_scipy

        return _sparse_operator(from_scipy(A))
    raise TypeError(
        "aslinearoperator accepts LinearOperator, CSRArray, COOArray, SciPy "
        "sparse matrices, or (shape, matvec[, matmat]) tuples."
    )


def as_linear_operator(A) -> LinearOperator:
    """Alias for :func:`mlx_sparse.linalg.aslinearoperator`."""
    return aslinearoperator(A)
