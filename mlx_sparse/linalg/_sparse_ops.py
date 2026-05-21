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

from mlx_sparse._coo import COOArray
from mlx_sparse._csr import CSRArray


def _as_csr(A) -> CSRArray:
    if isinstance(A, CSRArray):
        return A
    if isinstance(A, COOArray):
        return A.tocsr(canonical=True)
    raise TypeError(f"expected CSRArray or COOArray, got {type(A).__name__}.")


def vdot(a, b):
    """Sparse Frobenius inner product from native CSR merge kernels."""

    return _as_csr(a).vdot(_as_csr(b))


def dot(a, b):
    """Sparse Frobenius dot product from native CSR merge kernels."""

    return _as_csr(a).dot(_as_csr(b))
