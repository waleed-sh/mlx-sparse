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

from mlx_sparse.linalg.utils.sparse import inner_product_csr as _as_csr


def vdot(a, b):
    """Compute the Frobenius inner product of two sparse matrices.

    Returns ``sum(conj(a) * b)`` over all stored non-zero pairs, using the
    native CSR sorted-merge kernel for efficient sparse-sparse element-wise
    accumulation.  Equivalent to ``dot(conj(a), b)`` for real matrices.

    Args:
        a: First sparse matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray`.
        b: Second sparse matrix with the same shape as ``a``.  Must be a
            :class:`~mlx_sparse.CSRArray`, :class:`~mlx_sparse.COOArray`, or
            :class:`~mlx_sparse.CSCArray`.

    Returns:
        A scalar ``mlx.core.array`` equal to ``sum(conj(a) * b)``.

    Raises:
        TypeError: If ``a`` or ``b`` is not a supported sparse type.
    """
    return _as_csr(a).vdot(_as_csr(b))


def dot(a, b):
    """Compute the Frobenius dot product of two sparse matrices.

    Returns ``sum(a * b)`` over all stored non-zero pairs (no conjugation),
    using the native CSR sorted-merge kernel for efficient sparse-sparse
    element-wise accumulation.

    Args:
        a: First sparse matrix.  Must be a :class:`~mlx_sparse.CSRArray`,
            :class:`~mlx_sparse.COOArray`, or :class:`~mlx_sparse.CSCArray`.
        b: Second sparse matrix with the same shape as ``a``.  Must be a
            :class:`~mlx_sparse.CSRArray`, :class:`~mlx_sparse.COOArray`, or
            :class:`~mlx_sparse.CSCArray`.

    Returns:
        A scalar ``mlx.core.array`` equal to ``sum(a * b)``.

    Raises:
        TypeError: If ``a`` or ``b`` is not a supported sparse type.
    """
    return _as_csr(a).dot(_as_csr(b))
