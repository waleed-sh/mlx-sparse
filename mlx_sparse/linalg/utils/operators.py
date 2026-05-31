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

"""Linear-operator construction helpers."""

from __future__ import annotations

from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray


def sparse_operator(array: CSRArray | COOArray | CSCArray, operator_cls):
    """Wrap a sparse array in ``operator_cls`` using native CSR products."""

    csr = (
        array.canonicalize()
        if isinstance(array, CSRArray)
        else array.tocsr(canonical=True)
    )
    adjoint = csr.H
    return operator_cls(
        shape=csr.shape,
        matvec_fn=lambda x: csr @ x,
        matmat_fn=lambda X: csr @ X,
        rmatvec_fn=lambda x: adjoint @ x,
        dtype=csr.dtype,
        _sparse_array=csr,
    )
