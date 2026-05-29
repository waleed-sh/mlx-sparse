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


def test_csr_matmul_matches_hand_computed(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    rhs = mx.array(
        np.array(
            [
                [3.0, 1.0],
                [10.0, -2.0],
                [7.0, 4.0],
                [-2.0, 6.0],
            ],
            dtype=np.float32,
        )
    )

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    out = csr @ rhs

    expected = np.array([[-1.0, -2.0], [0.0, 0.0], [30.0, 22.0]], dtype=np.float32)
    np.testing.assert_allclose(to_numpy(out), expected)


def test_csr_matmul_matches_scipy_random(mx, scipy_sparse):
    rng = np.random.default_rng(789)
    scipy_csr = scipy_sparse.random(
        64,
        80,
        density=0.04,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    rhs_np = rng.normal(size=(80, 7)).astype(np.float32)

    csr = ms.csr_array(
        (
            mx.array(scipy_csr.data.astype(np.float32)),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
        canonical=True,
    )

    np.testing.assert_allclose(
        to_numpy(csr @ mx.array(rhs_np)),
        scipy_csr @ rhs_np,
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_matmul_batched_rhs_matches_dense_mlx(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    rhs_np = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) / 10

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    out = csr @ mx.array(rhs_np)

    expected = to_numpy(csr.todense()) @ rhs_np
    np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-5, atol=1e-5)


def test_explicit_csr_batched_matmul_matches_dense_and_scipy(mx, scipy_sparse):
    rng = np.random.default_rng(790)
    scipy_csr = scipy_sparse.random(
        36,
        45,
        density=0.07,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    scipy_csr.sum_duplicates()
    scipy_csr.sort_indices()
    rhs_np = rng.normal(size=(2, 4, 45, 6)).astype(np.float32)

    csr = ms.csr_array(
        (
            mx.array(scipy_csr.data.astype(np.float32)),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
        canonical=True,
    )

    out = ms.csr_batched_matmul(csr, mx.array(rhs_np))
    dense_expected = to_numpy(csr.todense()) @ rhs_np
    scipy_expected = np.stack(
        [scipy_csr @ rhs_np.reshape(-1, 45, 6)[i] for i in range(8)],
        axis=0,
    ).reshape(2, 4, 36, 6)

    np.testing.assert_allclose(to_numpy(out), dense_expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(out), scipy_expected, rtol=1e-5, atol=1e-5)


def test_native_csr_transpose_matmul_matches_dense_and_scipy(mx, scipy_sparse):
    rng = np.random.default_rng(791)
    scipy_csr = scipy_sparse.random(
        70,
        55,
        density=0.05,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    rhs_np = rng.normal(size=(70, 5)).astype(np.float32)
    csr = ms.csr_array(
        (
            mx.array(scipy_csr.data.astype(np.float32)),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
    )

    out = native.csr_matmul_transpose(
        csr.data, csr.indices, csr.indptr, mx.array(rhs_np), csr.shape
    )
    dense_expected = to_numpy(csr.todense()).T @ rhs_np
    scipy_expected = scipy_csr.T @ rhs_np

    np.testing.assert_allclose(to_numpy(out), dense_expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(out), scipy_expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("dtype_name", "rtol", "atol"),
    [
        ("float16", 5e-3, 5e-3),
        ("bfloat16", 3e-2, 3e-2),
        ("complex64", 1e-5, 1e-5),
    ],
)
def test_native_csr_transpose_matmul_segmented_dtypes(
    mx, scipy_sparse, dtype_name, rtol, atol
):
    data_np = np.array([2.0, -1.0, 0.5, 3.0, -2.5, 1.25], dtype=np.float32)
    rhs_np = np.array(
        [
            [1.0, -0.5, 0.25],
            [0.5, 2.0, -1.5],
            [-2.0, 0.75, 1.0],
            [1.25, -1.0, 0.5],
        ],
        dtype=np.float32,
    )
    if dtype_name == "complex64":
        data_np = data_np.astype(np.complex64) + 1j * np.array(
            [0.5, 0.0, -1.0, 0.25, 0.75, -0.5], dtype=np.float32
        )
        rhs_np = rhs_np.astype(np.complex64) + 1j * np.array(
            [
                [0.25, 0.0, -0.5],
                [-0.75, 0.5, 0.25],
                [0.5, -1.0, 0.75],
                [1.0, 0.25, -0.25],
            ],
            dtype=np.float32,
        )

    indices_np = np.array([0, 3, 1, 3, 2, 4], dtype=np.int64)
    indptr_np = np.array([0, 2, 3, 5, 6], dtype=np.int64)
    dtype = getattr(mx, dtype_name)
    csr = ms.csr_array(
        (
            mx.array(data_np).astype(dtype),
            mx.array(indices_np),
            mx.array(indptr_np),
        ),
        shape=(4, 5),
        sorted_indices=True,
    )
    rhs = mx.array(rhs_np).astype(dtype)

    out = native.csr_matmul_transpose(csr.data, csr.indices, csr.indptr, rhs, csr.shape)
    dense_expected = mx.transpose(csr.todense()) @ rhs
    scipy_expected = (
        scipy_sparse.csr_matrix(
            (data_np, indices_np.astype(np.int32), indptr_np.astype(np.int32)),
            shape=csr.shape,
        ).T
        @ rhs_np
    )

    out_np = to_numpy(out)
    dense_np = to_numpy(dense_expected)
    if dtype_name != "complex64":
        out_np = out_np.astype(np.float32)
        dense_np = dense_np.astype(np.float32)
        scipy_expected = np.asarray(scipy_expected, dtype=np.float32)

    np.testing.assert_allclose(out_np, dense_np, rtol=rtol, atol=atol)
    np.testing.assert_allclose(out_np, scipy_expected, rtol=rtol, atol=atol)


def test_csr_sparse_sparse_matmul_matches_dense(mx):
    lhs = ms.csr_array(
        (
            mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32)),
            mx.array(np.array([0, 2, 1, 3], dtype=np.int32)),
            mx.array(np.array([0, 2, 2, 4], dtype=np.int32)),
        ),
        shape=(3, 4),
        sorted_indices=True,
        canonical=True,
    )
    rhs = ms.csr_array(
        (
            mx.array(np.array([3.0, 7.0, -2.0, 6.0], dtype=np.float32)),
            mx.array(np.array([0, 1, 1, 2], dtype=np.int32)),
            mx.array(np.array([0, 1, 2, 3, 4], dtype=np.int32)),
        ),
        shape=(4, 3),
        sorted_indices=True,
        canonical=True,
    )

    out = lhs @ rhs

    assert isinstance(out, ms.CSRArray)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        to_numpy(lhs.todense()) @ to_numpy(rhs.todense()),
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_spgemm_prunes_exact_cancellation_and_keeps_sorted_rows(mx, scipy_sparse):
    lhs_data = np.array([1.0, -1.0, 2.0, 3.0, -2.0], dtype=np.float32)
    lhs_indices = np.array([1, 2, 0, 2, 3], dtype=np.int32)
    lhs_indptr = np.array([0, 2, 3, 5], dtype=np.int32)
    rhs_data = np.array([4.0, 7.0, -7.0, 5.0, -3.0, 3.0], dtype=np.float32)
    rhs_indices = np.array([2, 0, 0, 1, 2, 2], dtype=np.int32)
    rhs_indptr = np.array([0, 1, 3, 4, 6], dtype=np.int32)

    lhs = ms.csr_array(
        (
            mx.array(lhs_data),
            mx.array(lhs_indices),
            mx.array(lhs_indptr),
        ),
        shape=(3, 4),
        sorted_indices=True,
        canonical=True,
    )
    rhs = ms.csr_array(
        (
            mx.array(rhs_data),
            mx.array(rhs_indices),
            mx.array(rhs_indptr),
        ),
        shape=(4, 3),
        sorted_indices=True,
        canonical=True,
    )

    expected = scipy_sparse.csr_matrix(
        (lhs_data, lhs_indices, lhs_indptr), shape=lhs.shape
    ) @ scipy_sparse.csr_matrix((rhs_data, rhs_indices, rhs_indptr), shape=rhs.shape)
    expected.eliminate_zeros()
    out = lhs @ rhs

    assert isinstance(out, ms.CSRArray)
    assert out.sorted_indices
    assert out.has_canonical_format
    np.testing.assert_allclose(
        to_numpy(out.todense()), expected.toarray(), rtol=1e-5, atol=1e-5
    )
    data = to_numpy(out.data)
    indices = to_numpy(out.indices)
    indptr = to_numpy(out.indptr)
    assert np.all(data != 0)
    for row in range(out.shape[0]):
        row_indices = indices[indptr[row] : indptr[row + 1]]
        np.testing.assert_array_equal(row_indices, np.sort(row_indices))
        assert np.unique(row_indices).size == row_indices.size


@pytest.mark.cpu_only
def test_csr_spgemm_fixed_parallel_matches_serial_and_scipy(mx, scipy_sparse):
    lhs_data = np.array(
        [1.0, 1.0, 3.0, -2.0, 4.0, -1.0, 0.5, 2.5, -3.0, 1.5],
        dtype=np.float32,
    )
    lhs_indices = np.array([0, 1, 1, 4, 2, 3, 5, 0, 4, 5], dtype=np.int32)
    lhs_indptr = np.array([0, 2, 4, 6, 7, 10], dtype=np.int32)
    rhs_data = np.array(
        [2.0, -2.0, 1.5, -0.5, 3.0, 4.0, -4.0, 2.0, -1.0, 0.25, 5.0],
        dtype=np.float32,
    )
    rhs_indices = np.array([0, 0, 2, 5, 1, 3, 3, 0, 4, 5, 2], dtype=np.int32)
    rhs_indptr = np.array([0, 1, 4, 5, 7, 9, 11], dtype=np.int32)

    scipy_lhs = scipy_sparse.csr_matrix(
        (lhs_data, lhs_indices, lhs_indptr), shape=(5, 6)
    )
    scipy_rhs = scipy_sparse.csr_matrix(
        (rhs_data, rhs_indices, rhs_indptr), shape=(6, 6)
    )
    expected = scipy_lhs @ scipy_rhs
    expected.eliminate_zeros()
    lhs = ms.from_scipy(scipy_lhs)
    rhs = ms.from_scipy(scipy_rhs)

    with ms.runtime.context(spgemm_parallel=False):
        serial = lhs @ rhs
    with ms.runtime.context(spgemm_parallel=True, spgemm_threads=2):
        parallel = lhs @ rhs

    assert serial.sorted_indices
    assert serial.has_canonical_format
    assert parallel.sorted_indices
    assert parallel.has_canonical_format
    np.testing.assert_allclose(
        to_numpy(serial.todense()), expected.toarray(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(parallel.todense()), expected.toarray(), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(to_numpy(parallel.indptr), to_numpy(serial.indptr))
    np.testing.assert_array_equal(to_numpy(parallel.indices), to_numpy(serial.indices))
    np.testing.assert_allclose(to_numpy(parallel.data), to_numpy(serial.data))


def test_csr_spgemm_dense_ordered_extraction_keeps_sorted_rows(mx, scipy_sparse):
    block = 8
    blocks = 16
    n_cols = block * blocks
    lhs_data = np.ones(blocks, dtype=np.float32)
    lhs_indices = np.arange(blocks, dtype=np.int32)
    lhs_indptr = np.array([0, blocks], dtype=np.int32)

    rhs_indices_parts = []
    for rhs_row in range(blocks):
        start = (blocks - rhs_row - 1) * block
        rhs_indices_parts.append(np.arange(start, start + block, dtype=np.int32))
    rhs_indices = np.concatenate(rhs_indices_parts)
    rhs_data = np.linspace(0.5, 2.0, rhs_indices.size, dtype=np.float32)
    rhs_indptr = np.arange(blocks + 1, dtype=np.int32) * block

    lhs = ms.csr_array(
        (mx.array(lhs_data), mx.array(lhs_indices), mx.array(lhs_indptr)),
        shape=(1, blocks),
        sorted_indices=True,
        canonical=True,
    )
    rhs = ms.csr_array(
        (mx.array(rhs_data), mx.array(rhs_indices), mx.array(rhs_indptr)),
        shape=(blocks, n_cols),
        sorted_indices=True,
        canonical=True,
    )
    expected = scipy_sparse.csr_matrix(
        (lhs_data, lhs_indices, lhs_indptr), shape=lhs.shape
    ) @ scipy_sparse.csr_matrix((rhs_data, rhs_indices, rhs_indptr), shape=rhs.shape)

    out = lhs @ rhs

    assert out.sorted_indices
    assert out.has_canonical_format
    np.testing.assert_allclose(to_numpy(out.todense()), expected.toarray())
    np.testing.assert_array_equal(to_numpy(out.indices), np.arange(n_cols))


def test_csr_matmul_rejects_wrong_rhs_shape(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
        ),
        shape=(1, 2),
    )

    try:
        csr @ mx.array(np.ones((1, 2), dtype=np.float32))
    except ValueError as exc:
        assert "n_cols" in str(exc)
    else:
        raise AssertionError("expected csr_matmul shape validation to fail")
