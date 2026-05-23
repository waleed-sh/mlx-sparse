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


def test_csr_matvec_matches_hand_computed(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    x = mx.array(np.array([3.0, 10.0, 7.0, -2.0], dtype=np.float32))

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    y = csr @ x

    np.testing.assert_allclose(
        to_numpy(y), np.array([-1.0, 0.0, 30.0], dtype=np.float32)
    )


def test_csr_matvec_matches_scipy_random(mx, scipy_sparse):
    rng = np.random.default_rng(456)
    scipy_csr = scipy_sparse.random(
        96,
        128,
        density=0.03,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    x_np = rng.normal(size=(128,)).astype(np.float32)

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
    y = csr @ mx.array(x_np)

    np.testing.assert_allclose(to_numpy(y), scipy_csr @ x_np, rtol=1e-5, atol=1e-5)


def test_csr_batched_matvec_matches_dense_and_scipy(mx, scipy_sparse):
    rng = np.random.default_rng(457)
    scipy_csr = scipy_sparse.random(
        48,
        64,
        density=0.05,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    scipy_csr.sum_duplicates()
    scipy_csr.sort_indices()
    rhs_np = rng.normal(size=(3, 5, 64)).astype(np.float32)

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

    out = ms.csr_batched_matvec(csr, mx.array(rhs_np))
    dense_expected = rhs_np @ to_numpy(csr.todense()).T
    scipy_expected = rhs_np.reshape(-1, 64) @ scipy_csr.T
    scipy_expected = np.asarray(scipy_expected).reshape(3, 5, 48)

    np.testing.assert_allclose(to_numpy(out), dense_expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(out), scipy_expected, rtol=1e-5, atol=1e-5)


def test_native_csr_transpose_matvec_matches_dense_and_scipy(mx, scipy_sparse):
    rng = np.random.default_rng(458)
    scipy_csr = scipy_sparse.random(
        80,
        72,
        density=0.04,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    x_np = rng.normal(size=(80,)).astype(np.float32)
    csr = ms.csr_array(
        (
            mx.array(scipy_csr.data.astype(np.float32)),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
    )

    out = native.csr_matvec_transpose(
        csr.data, csr.indices, csr.indptr, mx.array(x_np), csr.shape
    )
    dense_expected = to_numpy(csr.todense()).T @ x_np
    scipy_expected = scipy_csr.T @ x_np

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
def test_native_csr_transpose_matvec_segmented_dtypes(
    mx, scipy_sparse, dtype_name, rtol, atol
):
    data_np = np.array([2.0, -1.0, 0.5, 3.0, -2.5, 1.25], dtype=np.float32)
    x_np = np.array([1.0, -0.5, 2.0, 0.75], dtype=np.float32)
    if dtype_name == "complex64":
        data_np = data_np.astype(np.complex64) + 1j * np.array(
            [0.5, 0.0, -1.0, 0.25, 0.75, -0.5], dtype=np.float32
        )
        x_np = x_np.astype(np.complex64) + 1j * np.array(
            [0.25, -0.75, 0.5, 1.0], dtype=np.float32
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
    x = mx.array(x_np).astype(dtype)

    out = native.csr_matvec_transpose(csr.data, csr.indices, csr.indptr, x, csr.shape)
    dense_expected = mx.transpose(csr.todense()) @ x
    scipy_expected = (
        scipy_sparse.csr_matrix(
            (data_np, indices_np.astype(np.int32), indptr_np.astype(np.int32)),
            shape=csr.shape,
        ).T
        @ x_np
    )

    out_np = to_numpy(out)
    dense_np = to_numpy(dense_expected)
    if dtype_name != "complex64":
        out_np = out_np.astype(np.float32)
        dense_np = dense_np.astype(np.float32)
        scipy_expected = np.asarray(scipy_expected, dtype=np.float32)

    np.testing.assert_allclose(out_np, dense_np, rtol=rtol, atol=atol)
    np.testing.assert_allclose(out_np, scipy_expected, rtol=rtol, atol=atol)


def test_csr_matvec_accepts_noncontiguous_inputs(mx):
    data = mx.array(
        np.array([99.0, 2.0, 99.0, -1.0, 99.0, 4.0, 99.0, 5.0], dtype=np.float32)
    )[1::2]
    indices = mx.array(np.array([9, 0, 9, 2, 9, 1, 9, 3], dtype=np.int32))[1::2]
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    x = mx.array(
        np.array([99.0, 3.0, 99.0, 10.0, 99.0, 7.0, 99.0, -2.0], dtype=np.float32)
    )[1::2]

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))
    y = csr @ x

    np.testing.assert_allclose(
        to_numpy(y), np.array([-1.0, 0.0, 30.0], dtype=np.float32)
    )


def test_csr_matvec_rejects_wrong_rhs_shape(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
        ),
        shape=(1, 2),
    )

    with pytest.raises(ValueError, match="n_cols"):
        csr @ mx.array(np.array([1.0], dtype=np.float32))


def test_csr_matvec_rejects_unsupported_dtype(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
        ),
        shape=(1, 1),
    )

    with pytest.raises(TypeError, match="dtype must be one"):
        csr @ mx.array(np.array([1], dtype=np.int32))


def test_no_hidden_eager_eval_composes_with_mlx_ops(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0, 2.0], dtype=np.float32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 2], dtype=np.int32)),
        ),
        shape=(1, 2),
    )
    x = mx.array(np.array([3.0, 4.0], dtype=np.float32))

    z = mx.sin(csr @ x) + 2.0

    np.testing.assert_allclose(
        to_numpy(z), np.sin(np.array([11.0], dtype=np.float32)) + 2.0
    )
