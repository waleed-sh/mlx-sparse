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

import mlx_sparse._fallback as _fallback
from mlx_sparse._ext_loader import extension
from mlx_sparse._typing import Shape2D


def identity_like(x):
    ext = extension()
    if ext is None:
        import mlx.core as mx

        return mx.array(x)
    return ext.identity_like(x)


def coo_tocsr(
    data: mx.array,
    row: mx.array,
    col: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        return _fallback.coo_to_csr(data, row, col, shape)
    return ext.coo_tocsr(data, row, col, shape[0], shape[1])


def csr_todense(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_todense(data, indices, indptr, shape)
    return ext.csr_todense(data, indices, indptr, shape[0], shape[1])


def csr_matvec(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_matvec(data, indices, indptr, x, shape)
    return ext.csr_matvec(data, indices, indptr, x, shape[0], shape[1])


def csr_matvec_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    x: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_matvec_transpose(data, indices, indptr, x, shape)
    return ext.csr_matvec_transpose(data, indices, indptr, x, shape[0], shape[1])


def csr_matmul(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_matmul(data, indices, indptr, rhs, shape)
    return ext.csr_matmul(data, indices, indptr, rhs, shape[0], shape[1])


def csr_matmul_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    rhs: mx.array,
    shape: Shape2D,
) -> mx.array:
    ext = extension()
    if ext is None:
        return _fallback.csr_matmul_transpose(data, indices, indptr, rhs, shape)
    return ext.csr_matmul_transpose(data, indices, indptr, rhs, shape[0], shape[1])


def csr_transpose(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        return _fallback.csr_transpose(data, indices, indptr, shape)
    return ext.csr_transpose(data, indices, indptr, shape[0], shape[1])


def csr_sort_indices(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    ext = extension()
    if ext is None:
        return _fallback.sort_csr_indices(data, indices, indptr)
    return ext.csr_sort_indices(data, indices, indptr)
