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

"""Coverage for the public ``csr_matvec_transpose`` op and ``CSRArray.matvec_transpose``.

``docs/supported.rst`` lists ``csr_matvec_transpose`` among the native primitives, but it
had no Python entrypoint, so the only route to a transposed product from a CSR array was
``A.T @ x``, which rebuilds the whole matrix. These tests check the exported op against
SciPy, check that it agrees with the materializing route it replaces, and check the
validation errors.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse._host import to_numpy


def _to_csr(mx, scipy_csr):
    return ms.csr_array(
        (
            mx.array(scipy_csr.data.astype(np.float32)),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
        canonical=True,
    )


def test_csr_matvec_transpose_matches_hand_computed(mx):
    #       [[2, 0, -1, 0],
    #   A =  [0, 0,  0, 0],   A.T @ [3, 10, 7] = [6, 28, -3, 35]
    #        [0, 4,  0, 5]]
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    x = mx.array(np.array([3.0, 10.0, 7.0], dtype=np.float32))

    csr = ms.csr_array((data, indices, indptr), shape=(3, 4))

    np.testing.assert_allclose(
        to_numpy(ms.csr_matvec_transpose(csr, x)),
        np.array([6.0, 28.0, -3.0, 35.0], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "n_rows,n_cols,density,seed",
    [
        (64, 64, 0.05, 1),  # square, asymmetric
        (30, 50, 0.15, 2),  # wide: A.T @ x consumes the shorter vector
        (50, 30, 0.15, 3),  # tall
        (128, 96, 0.002, 4),  # sparse enough to leave empty rows and columns
    ],
)
def test_csr_matvec_transpose_matches_scipy(
    mx, scipy_sparse, n_rows, n_cols, density, seed
):
    rng = np.random.default_rng(seed)
    scipy_csr = scipy_sparse.random(
        n_rows,
        n_cols,
        density=density,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    scipy_csr.sum_duplicates()
    scipy_csr.sort_indices()
    x_np = rng.normal(size=(n_rows,)).astype(np.float32)

    csr = _to_csr(mx, scipy_csr)
    expected = scipy_csr.T @ x_np

    np.testing.assert_allclose(
        to_numpy(ms.csr_matvec_transpose(csr, mx.array(x_np))),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(csr.matvec_transpose(mx.array(x_np))),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_csr_matvec_transpose_agrees_with_materialized_transpose(mx, scipy_sparse):
    """The exported op must return what the ``A.T @ x`` route it replaces returns."""
    rng = np.random.default_rng(5)
    scipy_csr = scipy_sparse.random(
        40, 24, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    scipy_csr.sum_duplicates()
    scipy_csr.sort_indices()
    x_np = rng.normal(size=(40,)).astype(np.float32)

    csr = _to_csr(mx, scipy_csr)

    np.testing.assert_allclose(
        to_numpy(ms.csr_matvec_transpose(csr, mx.array(x_np))),
        to_numpy(csr.T @ mx.array(x_np)),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "dtype_name,rtol,atol",
    [
        ("float32", 1e-5, 1e-5),
        ("float16", 5e-3, 5e-3),
        ("bfloat16", 4e-2, 4e-2),
        ("complex64", 1e-5, 1e-5),
    ],
)
def test_csr_matvec_transpose_matches_dense_for_each_value_dtype(
    mx, scipy_sparse, dtype_name, rtol, atol
):
    """Non-``float32`` GPU data reaches the kernel through a different route.

    ``csr_matvec_transpose`` only has an in-place GPU kernel for ``float32``; the other
    value dtypes lower through ``csr_transpose`` plus ``csr_matvec``. Both routes have
    to produce the dense answer.
    """
    dtype = getattr(mx, dtype_name)
    rng = np.random.default_rng(11)
    scipy_csr = scipy_sparse.random(
        24, 18, density=0.2, format="csr", dtype=np.float32, random_state=rng
    )
    scipy_csr.sum_duplicates()
    scipy_csr.sort_indices()
    x_np = rng.normal(size=(24,)).astype(np.float32)
    is_complex = dtype_name == "complex64"
    reference_dtype = np.complex128 if is_complex else np.float64

    if is_complex:
        # Give both operands an imaginary part, or the cell passes on real arithmetic.
        scipy_csr = scipy_csr.astype(np.complex64)
        scipy_csr.data += 1j * rng.normal(size=scipy_csr.data.shape).astype(np.float32)
        x_np = (x_np + 1j * rng.normal(size=x_np.shape)).astype(np.complex64)

    csr = ms.csr_array(
        (
            mx.array(scipy_csr.data).astype(dtype),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
        canonical=True,
    )
    result = ms.csr_matvec_transpose(csr, mx.array(x_np).astype(dtype))
    expected = scipy_csr.astype(reference_dtype).T @ x_np.astype(reference_dtype)

    assert result.dtype == dtype
    got = to_numpy(result)
    assert np.iscomplexobj(got) == is_complex
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


def test_csr_matvec_transpose_handles_empty_pattern(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([], dtype=np.float32)),
            mx.array(np.array([], dtype=np.int32)),
            mx.array(np.zeros(6, dtype=np.int32)),
        ),
        shape=(5, 4),
        sorted_indices=True,
        canonical=True,
    )
    result = ms.csr_matvec_transpose(csr, mx.ones((5,), dtype=mx.float32))

    assert result.shape == (4,)
    np.testing.assert_allclose(to_numpy(result), np.zeros(4, dtype=np.float32))


def _rejection_fixture(mx):
    return ms.csr_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.int32)),
        ),
        shape=(6, 4),
    )


def test_csr_matvec_transpose_rejects_wrong_length_rhs(mx):
    csr = _rejection_fixture(mx)
    # The transposed product consumes n_rows entries, not n_cols; passing n_cols is the
    # mistake a caller carries over from csr_matvec.
    with pytest.raises(ValueError, match="length 4, but sparse n_rows=6"):
        ms.csr_matvec_transpose(csr, mx.ones((4,), dtype=mx.float32))

    with pytest.raises(ValueError, match="rank-1 RHS"):
        ms.csr_matvec_transpose(csr, mx.ones((6, 2), dtype=mx.float32))


def test_csr_matvec_transpose_rejects_mismatched_dtype(mx):
    csr = _rejection_fixture(mx)
    with pytest.raises(TypeError, match="same dtype"):
        ms.csr_matvec_transpose(csr, mx.ones((6,), dtype=mx.float16))


def test_csr_matvec_transpose_is_exported():
    assert "csr_matvec_transpose" in ms.__all__
    assert callable(ms.csr_matvec_transpose)
