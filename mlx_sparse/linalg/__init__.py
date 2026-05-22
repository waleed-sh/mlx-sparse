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

"""Native sparse linear algebra routines for mlx-sparse.

The linalg layer operates on sparse containers and native extension kernels.
Dense MLX arrays should use ``mlx.linalg`` directly.
"""

from mlx_sparse.linalg._eigen import eigs, eigsh, lanczos, svds
from mlx_sparse.linalg._factorizations import (
    SparseCholesky,
    SparseLU,
    cholesky,
    sparse_cholesky,
    sparse_lu,
    splu,
    spsolve,
)
from mlx_sparse.linalg._interface import LinearOperator, aslinearoperator
from mlx_sparse.linalg._iterative import cg, gmres, minres
from mlx_sparse.linalg._sparse_ops import dot, vdot

__all__ = [
    "LinearOperator",
    "SparseCholesky",
    "SparseLU",
    "aslinearoperator",
    "cg",
    "cholesky",
    "dot",
    "eigs",
    "eigsh",
    "gmres",
    "lanczos",
    "minres",
    "sparse_cholesky",
    "sparse_lu",
    "splu",
    "spsolve",
    "svds",
    "vdot",
]
