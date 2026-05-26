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

"""Sparse array containers and primitives for MLX.

mlx-sparse provides COO, CSR, and CSC sparse matrix containers backed by MLX
arrays, with native C++ primitives for sparse-dense products on Apple Silicon
CPU and Metal GPU. The public API is intentionally small: construct sparse matrices
from MLX, NumPy, SciPy, or explicit sparse buffers, run sparse-dense products,
differentiate through sparse values and dense operands, and convert back to
dense when needed.

Typical usage::

    import mlx.core as mx
    import numpy as np
    import mlx_sparse as ms

    ms.use_gpu()

    data = mx.array(np.array([2.0, -1.0, 4.0], dtype=np.float32))
    row = mx.array(np.array([0, 0, 1], dtype=np.int32))
    col = mx.array(np.array([0, 2, 1], dtype=np.int32))

    A = ms.coo_array((data, (row, col)), shape=(2, 3)).tocsr(canonical=True)
    x = mx.array(np.array([3.0, 10.0, 7.0], dtype=np.float32))

    y = A @ x  # CSR matvec
    dense = A.todense()  # materialise as dense
    At = A.T  # structural transpose
    Ah = A.H  # Hermitian (conjugate) transpose
"""

from mlx_sparse import linalg
from mlx_sparse._capabilities import (
    capabilities,
    has_capability,
)
from mlx_sparse._config import (
    config,
    config_context,
    get_config,
    set_config,
)
from mlx_sparse._construct import (
    asarray,
    diags,
    eye,
    from_dense,
    from_numpy,
    from_scipy,
    fromdense,
)
from mlx_sparse._coo import COOArray, coo_array
from mlx_sparse._csc import CSCArray, csc_array
from mlx_sparse._csr import CSRArray, csr_array
from mlx_sparse._device import use_cpu, use_device, use_gpu
from mlx_sparse._ops import (
    coo_batched_matmul,
    coo_batched_matvec,
    coo_col_norms,
    coo_col_sums,
    coo_column_norms,
    coo_column_sums,
    coo_diagonal,
    coo_matmat,
    coo_matmul,
    coo_matvec,
    coo_row_norms,
    coo_row_sums,
    coo_trace,
    csc_batched_matmul,
    csc_batched_matvec,
    csc_col_norms,
    csc_col_sums,
    csc_column_norms,
    csc_column_sums,
    csc_diagonal,
    csc_matmat,
    csc_matmul,
    csc_matvec,
    csc_matvec_transpose,
    csc_row_norms,
    csc_row_sums,
    csc_trace,
    csr_batched_matmul,
    csr_batched_matvec,
    csr_col_sums,
    csr_column_sums,
    csr_diagonal,
    csr_matmat,
    csr_matmul,
    csr_matvec,
    csr_row_norms,
    csr_row_sums,
    csr_trace,
    identity_like,
    todense,
)
from mlx_sparse._typing import is_available

try:
    from mlx_sparse._version import __version__
except ImportError:
    # Package was not installed via pip / build, running directly from source.
    __version__ = "0.0.0.dev0"


def issparse(x) -> bool:
    """Return ``True`` if ``x`` is a recognized mlx-sparse container.

    Currently returns ``True`` for :class:`COOArray`, :class:`CSRArray`, and
    :class:`CSCArray`
    instances. All other objects return ``False``.

    Args:
        x: Any Python object.

    Returns:
        ``True`` if ``x`` is a :class:`COOArray`, :class:`CSRArray`, or
        :class:`CSCArray`.

    Example::

        import mlx_sparse as ms

        ms.issparse(my_csr)  # True
        ms.issparse(mx.ones((3, 4)))  # False
    """
    return isinstance(x, (COOArray, CSRArray, CSCArray))


__all__ = [
    "COOArray",
    "CSCArray",
    "CSRArray",
    "asarray",
    "capabilities",
    "coo_array",
    "config",
    "config_context",
    "coo_batched_matmul",
    "coo_batched_matvec",
    "coo_col_norms",
    "coo_col_sums",
    "coo_column_norms",
    "coo_column_sums",
    "coo_diagonal",
    "coo_matmat",
    "coo_row_norms",
    "coo_row_sums",
    "coo_trace",
    "csc_array",
    "csc_batched_matmul",
    "csc_batched_matvec",
    "csc_col_norms",
    "csc_col_sums",
    "csc_column_norms",
    "csc_column_sums",
    "csc_diagonal",
    "csc_matmat",
    "csc_matmul",
    "csc_matvec",
    "csc_matvec_transpose",
    "csc_row_norms",
    "csc_row_sums",
    "csc_trace",
    "csr_array",
    "csr_batched_matmul",
    "csr_batched_matvec",
    "csr_col_sums",
    "csr_column_sums",
    "csr_diagonal",
    "csr_matmat",
    "csr_matmul",
    "csr_matvec",
    "csr_row_norms",
    "csr_row_sums",
    "csr_trace",
    "coo_matvec",
    "coo_matmul",
    "diags",
    "eye",
    "from_dense",
    "from_numpy",
    "from_scipy",
    "fromdense",
    "identity_like",
    "is_available",
    "issparse",
    "linalg",
    "has_capability",
    "get_config",
    "set_config",
    "todense",
    "use_cpu",
    "use_device",
    "use_gpu",
]
