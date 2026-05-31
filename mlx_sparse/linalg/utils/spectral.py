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

from mlx_sparse._csr import CSRArray
from mlx_sparse.linalg.utils.arrays import ensure_float32_csr
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
