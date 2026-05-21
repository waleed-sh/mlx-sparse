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
