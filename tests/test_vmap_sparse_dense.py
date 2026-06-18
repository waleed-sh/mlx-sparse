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

import numpy as np
import pytest

import mlx_sparse as ms
import mlx_sparse._native as native
from mlx_sparse._host import to_numpy


def _sample_sparse(mx, fmt: str, *, dtype=np.float32, index_dtype=np.int32):
    dense = np.array(
        [
            [2.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 1.5, 0.0],
            [0.0, 0.0, 4.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, -2.0, 0.0],
        ],
        dtype=np.float32,
    )
    if dtype == np.complex64:
        dense = dense.astype(np.complex64)
        dense += 1j * np.array(
            [
                [0.25, 0.0, -0.5, 0.0, 0.0],
                [0.0, 0.75, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0],
                [-0.25, 0.0, 0.0, 1.25, 0.0],
            ],
            dtype=np.float32,
        )
    else:
        dense = dense.astype(dtype)

    csr_data = dense[dense != 0].astype(dtype, copy=False)
    csr_indices = np.array([0, 2, 1, 3, 2, 0, 3], dtype=index_dtype)
    csr_indptr = np.array([0, 2, 4, 5, 7], dtype=index_dtype)
    row = np.array([0, 0, 1, 1, 2, 3, 3], dtype=index_dtype)
    col = csr_indices.copy()

    csc_data = dense[[0, 3, 1, 0, 2, 1, 3], [0, 0, 1, 2, 2, 3, 3]].astype(
        dtype,
        copy=False,
    )
    csc_indices = np.array([0, 3, 1, 0, 2, 1, 3], dtype=index_dtype)
    csc_indptr = np.array([0, 2, 3, 5, 7, 7], dtype=index_dtype)

    if fmt == "csr":
        array = ms.csr_array(
            (mx.array(csr_data), mx.array(csr_indices), mx.array(csr_indptr)),
            shape=dense.shape,
            sorted_indices=True,
            canonical=True,
        )
    elif fmt == "coo":
        array = ms.coo_array(
            (mx.array(csr_data), (mx.array(row), mx.array(col))),
            shape=dense.shape,
            canonical=True,
        )
    elif fmt == "csc":
        array = ms.csc_array(
            (mx.array(csc_data), mx.array(csc_indices), mx.array(csc_indptr)),
            shape=dense.shape,
            sorted_indices=True,
            canonical=True,
        )
    else:
        raise AssertionError(f"unknown sparse format {fmt!r}")
    return array, dense


def _empty_sparse(mx, fmt: str):
    shape = (3, 5)
    if fmt == "csr":
        return ms.csr_array(
            (
                mx.array(np.array([], dtype=np.float32)),
                mx.array(np.array([], dtype=np.int64)),
                mx.array(np.zeros(shape[0] + 1, dtype=np.int64)),
            ),
            shape=shape,
            sorted_indices=True,
            canonical=True,
        )
    if fmt == "coo":
        return ms.coo_array(
            (
                mx.array(np.array([], dtype=np.float32)),
                (
                    mx.array(np.array([], dtype=np.int64)),
                    mx.array(np.array([], dtype=np.int64)),
                ),
            ),
            shape=shape,
            canonical=True,
        )
    if fmt == "csc":
        return ms.csc_array(
            (
                mx.array(np.array([], dtype=np.float32)),
                mx.array(np.array([], dtype=np.int64)),
                mx.array(np.zeros(shape[1] + 1, dtype=np.int64)),
            ),
            shape=shape,
            sorted_indices=True,
            canonical=True,
        )
    raise AssertionError(f"unknown sparse format {fmt!r}")


def _batched_matvec(fmt: str, sparse, rhs):
    return getattr(ms, f"{fmt}_batched_matvec")(sparse, rhs)


def _batched_matmul(fmt: str, sparse, rhs):
    return getattr(ms, f"{fmt}_batched_matmul")(sparse, rhs)


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_vmap_sparse_dense_matvec_matches_batched_helper_and_dense(
    mx, fmt, index_dtype, dtype
):
    sparse, dense = _sample_sparse(mx, fmt, dtype=dtype, index_dtype=index_dtype)
    vectors_np = (
        np.arange(3 * sparse.shape[1], dtype=np.float32).reshape(3, sparse.shape[1])
        / 7.0
        - 1.0
    ).astype(dtype)
    if dtype == np.complex64:
        vectors_np = vectors_np + 1j * (vectors_np[::-1] / 3.0)

    vectors = mx.array(vectors_np)
    out = mx.vmap(lambda x: sparse @ x)(vectors)
    helper = _batched_matvec(fmt, sparse, vectors)
    expected = vectors_np @ dense.T

    np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(out), to_numpy(helper), rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_vmap_sparse_dense_matmul_matches_batched_helper_and_dense(
    mx, fmt, index_dtype
):
    sparse, dense = _sample_sparse(mx, fmt, index_dtype=index_dtype)
    matrices_np = (
        np.arange(3 * sparse.shape[1] * 2, dtype=np.float32).reshape(
            3, sparse.shape[1], 2
        )
        / 11.0
        - 0.5
    )

    matrices = mx.array(matrices_np)
    out = mx.vmap(lambda rhs: sparse @ rhs)(matrices)
    helper = _batched_matmul(fmt, sparse, matrices)
    expected = dense @ matrices_np

    np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(out), to_numpy(helper), rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
def test_vmap_dense_rhs_axis_and_output_axis_are_respected(mx, fmt):
    sparse, dense = _sample_sparse(mx, fmt)
    rhs_np = (
        np.arange(sparse.shape[1] * 4, dtype=np.float32).reshape(sparse.shape[1], 4)
        / 5.0
        - 2.0
    )

    out = mx.vmap(lambda x: sparse @ x, in_axes=1, out_axes=1)(mx.array(rhs_np))
    expected = dense @ rhs_np

    np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
@pytest.mark.parametrize("kind", ["matvec", "matmul"])
def test_vmap_over_explicit_batched_helpers_flattens_outer_axis(mx, fmt, kind):
    sparse, dense = _sample_sparse(mx, fmt)
    if kind == "matvec":
        rhs_np = (
            np.arange(2 * 3 * sparse.shape[1], dtype=np.float32).reshape(
                2, 3, sparse.shape[1]
            )
            / 13.0
            - 1.0
        )
        rhs = mx.array(rhs_np)
        out = mx.vmap(lambda batch: _batched_matvec(fmt, sparse, batch))(rhs)
        expected = rhs_np @ dense.T
    else:
        rhs_np = (
            np.arange(2 * 3 * sparse.shape[1] * 2, dtype=np.float32).reshape(
                2, 3, sparse.shape[1], 2
            )
            / 17.0
            - 0.5
        )
        rhs = mx.array(rhs_np)
        out = mx.vmap(lambda batch: _batched_matmul(fmt, sparse, batch))(rhs)
        expected = dense @ rhs_np

    np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
def test_vmap_zero_nnz_sparse_dense_products_return_stable_shapes(mx, fmt):
    sparse = _empty_sparse(mx, fmt)
    vectors = mx.array(np.ones((2, sparse.shape[1]), dtype=np.float32))
    matrices = mx.array(np.ones((2, sparse.shape[1], 3), dtype=np.float32))

    matvec_out = mx.vmap(lambda x: sparse @ x)(vectors)
    matmul_out = mx.vmap(lambda rhs: sparse @ rhs)(matrices)

    np.testing.assert_array_equal(
        to_numpy(matvec_out), np.zeros((2, sparse.shape[0]), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        to_numpy(matmul_out), np.zeros((2, sparse.shape[0], 3), dtype=np.float32)
    )


@pytest.mark.parametrize("mapped_axis", ["data", "indices", "indptr"])
def test_vmap_mapped_sparse_primitive_axes_are_rejected_precisely(mx, mapped_axis):
    if native.extension() is None:
        pytest.skip("native extension is required for primitive vmap errors")

    data = mx.array(np.array([2.0, -1.0, 0.5, 1.5], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 4], dtype=np.int32))
    x = mx.array(np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32))

    if mapped_axis == "data":
        batched = mx.stack([data, data * 2.0])
        fn = lambda values: native.csr_matvec(values, indices, indptr, x, (2, 4))
    elif mapped_axis == "indices":
        batched = mx.stack([indices, indices])
        fn = lambda idx: native.csr_matvec(data, idx, indptr, x, (2, 4))
    else:
        batched = mx.stack([indptr, indptr])
        fn = lambda ptr: native.csr_matvec(data, indices, ptr, x, (2, 4))

    with pytest.raises((ValueError, RuntimeError), match="must be unmapped"):
        out = mx.vmap(fn)(batched)
        mx.eval(out)


@pytest.mark.gpu
@pytest.mark.parametrize("fmt", ["csr", "coo", "csc"])
def test_vmap_sparse_dense_matmul_gpu_smoke(mx, fmt):
    sparse, dense = _sample_sparse(mx, fmt)
    rhs_np = (
        np.arange(2 * sparse.shape[1] * 3, dtype=np.float32).reshape(
            2, sparse.shape[1], 3
        )
        / 19.0
    )

    out = mx.vmap(lambda rhs: sparse @ rhs)(mx.array(rhs_np))

    np.testing.assert_allclose(to_numpy(out), dense @ rhs_np, rtol=1e-5, atol=1e-5)
