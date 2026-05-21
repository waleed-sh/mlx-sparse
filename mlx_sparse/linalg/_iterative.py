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
import numpy as np

import mlx_sparse._native as _native
from mlx_sparse._coo import COOArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._host import to_numpy
from mlx_sparse._validation import ensure_mx_array


def _as_csr(A) -> CSRArray:
    if isinstance(A, CSRArray):
        return A.canonicalize()
    if isinstance(A, COOArray):
        return A.tocsr(canonical=True)
    # Accept LinearOperator when it wraps a CSRArray (created via aslinearoperator).
    # The _sparse_array attribute holds the backing CSR for fast native dispatch.
    from mlx_sparse.linalg._interface import LinearOperator

    if isinstance(A, LinearOperator):
        if A._sparse_array is not None:
            return A._sparse_array.canonicalize()
        raise TypeError(
            "Iterative solvers accept LinearOperator only when it wraps a "
            "CSRArray (use aslinearoperator(csr_array)). "
            "For fully matrix-free operators, implement a Python-level "
            "iterative solver loop."
        )
    raise TypeError(
        "sparse iterative solvers expect CSRArray, COOArray, or a "
        "CSR-backed LinearOperator. Use mlx.linalg for dense arrays."
    )


def _float32_array(x) -> mx.array:
    array = ensure_mx_array(x)
    if array.dtype == mx.float32:
        return array
    if array.dtype in {mx.float16, mx.bfloat16}:
        return array.astype(mx.float32)
    raise TypeError("sparse iterative solvers currently require real float data.")


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
    raise TypeError("sparse iterative solvers currently require real float data.")


def _info(info) -> int:
    mx.eval(info)
    return int(np.asarray(to_numpy(info)).item())


def _guess(csr: CSRArray, b: mx.array, x0) -> mx.array:
    if b.ndim != 1:
        raise ValueError(f"right-hand side must be rank-1, got {b.shape}.")
    if b.shape[0] != csr.shape[0]:
        raise ValueError(f"b has length {b.shape[0]}, expected {csr.shape[0]}.")
    if x0 is None:
        return mx.zeros((csr.shape[1],), dtype=mx.float32)
    x = _float32_array(x0)
    if x.ndim != 1 or x.shape[0] != csr.shape[1]:
        raise ValueError(f"x0 must have shape ({csr.shape[1]},), got {x.shape}.")
    return x


def _maxiter(csr: CSRArray, maxiter: int | None) -> int:
    value = 10 * csr.shape[1] if maxiter is None else int(maxiter)
    if value < 0:
        raise ValueError("maxiter must be non-negative.")
    return value


def cg(
    A,
    b,
    *,
    x0=None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    M=None,
    callback=None,
):
    """Native sparse conjugate gradients for real SPD CSR/COO matrices."""

    if M is not None or callback is not None:
        raise NotImplementedError("native sparse cg does not use Python callbacks.")
    csr = _float32_csr(_as_csr(A))
    rhs = _float32_array(b)
    guess = _guess(csr, rhs, x0)
    x, info, _, _ = _native.csr_cg(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        guess,
        csr.shape,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=_maxiter(csr, maxiter),
    )
    return x, _info(info)


def gmres(
    A,
    b,
    *,
    x0=None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    restart: int | None = None,
    maxiter: int | None = None,
    M=None,
    callback=None,
    callback_type: str = "x",
):
    """Native sparse restarted GMRES for real CSR/COO matrices."""

    if M is not None or callback is not None:
        raise NotImplementedError("native sparse gmres does not use Python callbacks.")
    if callback_type not in {"x", "pr_norm", "legacy"}:
        raise ValueError("callback_type must be 'x', 'pr_norm', or 'legacy'.")
    csr = _float32_csr(_as_csr(A))
    rhs = _float32_array(b)
    guess = _guess(csr, rhs, x0)
    restart_value = min(20, csr.shape[0]) if restart is None else int(restart)
    if restart_value <= 0:
        raise ValueError("restart must be positive.")
    x, info, _, _ = _native.csr_gmres(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        guess,
        csr.shape,
        rtol=float(rtol),
        atol=float(atol),
        restart=restart_value,
        maxiter=_maxiter(csr, maxiter),
    )
    return x, _info(info)


def minres(
    A,
    b,
    *,
    x0=None,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    callback=None,
):
    """Native sparse MINRES for real symmetric/Hermitian CSR/COO matrices."""

    if callback is not None:
        raise NotImplementedError("native sparse minres does not use Python callbacks.")
    csr = _float32_csr(_as_csr(A))
    rhs = _float32_array(b)
    guess = _guess(csr, rhs, x0)
    x, info, _, _ = _native.csr_minres(
        csr.data,
        csr.indices,
        csr.indptr,
        rhs,
        guess,
        csr.shape,
        rtol=float(rtol),
        atol=float(atol),
        maxiter=_maxiter(csr, maxiter),
    )
    return x, _info(info)
