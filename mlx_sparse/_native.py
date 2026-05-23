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

import mlx_sparse._fallback as _fallback
from mlx_sparse._ext_loader import extension
from mlx_sparse._typing import Shape2D


def _index_dtype_bits(index_dtype) -> int:
    if index_dtype == mx.int32:
        return 32
    if index_dtype == mx.int64:
        return 64
    raise TypeError(f"index_dtype must be mx.int32 or mx.int64, got {index_dtype}.")


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


def csr_matmat(lhs, rhs):
    ext = extension()
    if ext is None:
        return _fallback.csr_matmat(lhs, rhs)
    return ext.csr_matmat(
        lhs.data,
        lhs.indices,
        lhs.indptr,
        rhs.data,
        rhs.indices,
        rhs.indptr,
        lhs.shape[0],
        lhs.shape[1],
        rhs.shape[0],
        rhs.shape[1],
    )


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


def csr_sum_duplicates(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
):
    ext = extension()
    if ext is None:
        return _fallback.sum_csr_duplicates(data, indices, indptr)
    return ext.csr_sum_duplicates(data, indices, indptr)


def csr_fromdense(
    dense: mx.array,
    *,
    index_dtype,
    threshold: float,
):
    ext = extension()
    if ext is None:
        return _fallback.fromdense(dense, index_dtype=index_dtype, threshold=threshold)
    return ext.csr_fromdense(dense, _index_dtype_bits(index_dtype), float(threshold))


def csr_cg(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_cg requires the native mlx_sparse extension.")
    return ext.csr_cg(
        data,
        indices,
        indptr,
        b,
        x0,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(maxiter),
    )


def csr_lanczos(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    v0: mx.array,
    shape: Shape2D,
    *,
    k: int,
    reorthogonalize: bool,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_lanczos requires the native mlx_sparse extension.")
    return ext.csr_lanczos(
        data,
        indices,
        indptr,
        v0,
        shape[0],
        shape[1],
        int(k),
        bool(reorthogonalize),
    )


def csr_gmres(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_gmres requires the native mlx_sparse extension.")
    return ext.csr_gmres(
        data,
        indices,
        indptr,
        b,
        x0,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(restart),
        int(maxiter),
    )


def csr_minres(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    x0: mx.array,
    shape: Shape2D,
    *,
    rtol: float,
    atol: float,
    maxiter: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_minres requires the native mlx_sparse extension.")
    return ext.csr_minres(
        data,
        indices,
        indptr,
        b,
        x0,
        shape[0],
        shape[1],
        float(rtol),
        float(atol),
        int(maxiter),
    )


def csr_arnoldi(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    v0: mx.array,
    shape: Shape2D,
    *,
    k: int,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_arnoldi requires the native mlx_sparse extension.")
    return ext.csr_arnoldi(
        data,
        indices,
        indptr,
        v0,
        shape[0],
        shape[1],
        int(k),
    )


def csr_eigsh(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    *,
    k: int,
    ncv: int,
    which: str,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_eigsh requires the native mlx_sparse extension.")
    return ext.csr_eigsh(
        data,
        indices,
        indptr,
        shape[0],
        shape[1],
        int(k),
        int(ncv),
        str(which),
    )


def csr_eigs(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    *,
    k: int,
    ncv: int,
    which: str,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_eigs requires the native mlx_sparse extension.")
    return ext.csr_eigs(
        data,
        indices,
        indptr,
        shape[0],
        shape[1],
        int(k),
        int(ncv),
        str(which),
    )


def csr_svds(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
    *,
    k: int,
    ncv: int,
    which: str,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_svds requires the native mlx_sparse extension.")
    return ext.csr_svds(
        data,
        indices,
        indptr,
        shape[0],
        shape[1],
        int(k),
        int(ncv),
        str(which),
    )


def csr_cholesky(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_cholesky requires the native mlx_sparse extension.")
    return ext.csr_cholesky(data, indices, indptr, shape[0], shape[1])


def csr_lu(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_lu requires the native mlx_sparse extension.")
    return ext.csr_lu(data, indices, indptr, shape[0], shape[1])


def csr_triangular_solve(
    data: mx.array,
    indices: mx.array,
    indptr: mx.array,
    b: mx.array,
    shape: Shape2D,
    *,
    lower: bool,
    unit_diagonal: bool,
):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_triangular_solve requires the native mlx_sparse extension."
        )
    return ext.csr_triangular_solve(
        data,
        indices,
        indptr,
        b,
        shape[0],
        shape[1],
        bool(lower),
        bool(unit_diagonal),
    )


def csr_vdot(
    lhs_data: mx.array,
    lhs_indices: mx.array,
    lhs_indptr: mx.array,
    rhs_data: mx.array,
    rhs_indices: mx.array,
    rhs_indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_vdot requires the native mlx_sparse extension.")
    return ext.csr_vdot(
        lhs_data,
        lhs_indices,
        lhs_indptr,
        rhs_data,
        rhs_indices,
        rhs_indptr,
        shape[0],
        shape[1],
    )


def csr_dot(
    lhs_data: mx.array,
    lhs_indices: mx.array,
    lhs_indptr: mx.array,
    rhs_data: mx.array,
    rhs_indices: mx.array,
    rhs_indptr: mx.array,
    shape: Shape2D,
):
    ext = extension()
    if ext is None:
        raise RuntimeError("csr_dot requires the native mlx_sparse extension.")
    return ext.csr_dot(
        lhs_data,
        lhs_indices,
        lhs_indptr,
        rhs_data,
        rhs_indices,
        rhs_indptr,
        shape[0],
        shape[1],
    )


def csr_permute_vector(x: mx.array, perm: mx.array):
    ext = extension()
    if ext is None:
        raise RuntimeError(
            "csr_permute_vector requires the native mlx_sparse extension."
        )
    return ext.csr_permute_vector(x, perm)
