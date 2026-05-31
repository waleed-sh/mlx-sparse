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

"""Iterative-solver argument and return-value helpers."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_sparse._csr import CSRArray
from mlx_sparse._host import to_numpy
from mlx_sparse.linalg.utils.arrays import ensure_float32_array, ensure_float32_csr
from mlx_sparse.linalg.utils.sparse import canonical_csr


def as_csr(A) -> CSRArray:
    """Return iterative-solver input as canonical CSR."""

    return canonical_csr(
        A,
        context="sparse iterative solvers",
        dense_guidance="Use mlx.linalg for dense arrays.",
        allow_sparse_linear_operator=True,
    )


def float32_array(x) -> mx.array:
    """Return iterative-solver dense input with float32 values."""

    return ensure_float32_array(x, context="sparse iterative solvers")


def float32_csr(A: CSRArray) -> CSRArray:
    """Return iterative-solver CSR input with float32 values."""

    return ensure_float32_csr(A, context="sparse iterative solvers")


def solver_info_to_int(info) -> int:
    """Synchronize a scalar native ``info`` array and return it as ``int``."""

    mx.eval(info)
    return int(np.asarray(to_numpy(info)).item())


def initial_guess(csr: CSRArray, b: mx.array, x0) -> mx.array:
    """Validate ``b`` and return a float32 initial guess vector."""

    if b.ndim != 1:
        raise ValueError(f"right-hand side must be rank-1, got {b.shape}.")
    if b.shape[0] != csr.shape[0]:
        raise ValueError(f"b has length {b.shape[0]}, expected {csr.shape[0]}.")
    if x0 is None:
        return mx.zeros((csr.shape[1],), dtype=mx.float32)
    x = float32_array(x0)
    if x.ndim != 1 or x.shape[0] != csr.shape[1]:
        raise ValueError(f"x0 must have shape ({csr.shape[1]},), got {x.shape}.")
    return x


def max_iterations(csr: CSRArray, maxiter: int | None) -> int:
    """Normalize a solver iteration budget."""

    value = 10 * csr.shape[1] if maxiter is None else int(maxiter)
    if value < 0:
        raise ValueError("maxiter must be non-negative.")
    return value
