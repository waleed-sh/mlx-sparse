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
from mlx_sparse._host import to_numpy

import mlx_sparse as ms


@pytest.mark.parametrize(
    ("np_dtype", "mx_dtype", "rtol", "atol"),
    [
        (np.float16, "float16", 5e-3, 5e-3),
        (np.float32, "float32", 1e-5, 1e-5),
        (np.complex64, "complex64", 1e-5, 1e-5),
    ],
)
def test_csr_matvec_supported_value_dtypes(mx, np_dtype, mx_dtype, rtol, atol):
    mlx_dtype = getattr(mx, mx_dtype)
    values = np.array([2.0, -1.0, 4.0, 5.0], dtype=np_dtype)
    x_np = np.array([3.0, 10.0, 7.0, -2.0], dtype=np_dtype)
    if np.issubdtype(np_dtype, np.complexfloating):
        values = values + np.array([1.0j, -2.0j, 0.5j, 3.0j], dtype=np_dtype)
        x_np = x_np + np.array([-1.0j, 2.0j, 0.0j, -0.5j], dtype=np_dtype)

    csr = ms.csr_array(
        (
            mx.array(values, dtype=mlx_dtype),
            mx.array(np.array([0, 2, 1, 3], dtype=np.int32)),
            mx.array(np.array([0, 2, 2, 4], dtype=np.int32)),
        ),
        shape=(3, 4),
    )

    expected = np.array(
        [
            values[0] * x_np[0] + values[1] * x_np[2],
            0,
            values[2] * x_np[1] + values[3] * x_np[3],
        ],
        dtype=np_dtype,
    )
    np.testing.assert_allclose(
        to_numpy(csr @ mx.array(x_np, dtype=mlx_dtype)),
        expected,
        rtol=rtol,
        atol=atol,
    )


def test_csr_matvec_bfloat16_runs_and_returns_bfloat16(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([2.0, -1.0], dtype=np.float32), dtype=mx.bfloat16),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 2], dtype=np.int32)),
        ),
        shape=(1, 2),
    )
    x = mx.array(np.array([3.0, 7.0], dtype=np.float32), dtype=mx.bfloat16)

    y = csr @ x

    assert y.dtype == mx.bfloat16
    np.testing.assert_allclose(to_numpy(y).astype(np.float32), [-1.0], atol=1e-2)


@pytest.mark.parametrize(
    ("storage_dtype", "tolerance"),
    [
        ("float16", 5e-3),
        ("bfloat16", 2e-2),
    ],
)
def test_csr_matmul_low_precision_dtypes(mx, storage_dtype, tolerance):
    mlx_dtype = getattr(mx, storage_dtype)
    csr = ms.csr_array(
        (
            mx.array(np.array([2.0, -1.0, 4.0], dtype=np.float32), dtype=mlx_dtype),
            mx.array(np.array([0, 2, 1], dtype=np.int32)),
            mx.array(np.array([0, 2, 3], dtype=np.int32)),
        ),
        shape=(2, 3),
    )
    rhs_np = np.array([[3.0, 1.0], [10.0, -2.0], [7.0, 4.0]], dtype=np.float32)

    out = csr @ mx.array(rhs_np, dtype=mlx_dtype)

    expected = np.array([[-1.0, -2.0], [40.0, -8.0]], dtype=np.float32)
    assert out.dtype == mlx_dtype
    np.testing.assert_allclose(
        to_numpy(out).astype(np.float32),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


def test_dtype_mismatch_is_rejected(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            mx.array(np.array([0], dtype=np.int32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
        ),
        shape=(1, 1),
    )

    with pytest.raises(TypeError, match="same dtype"):
        csr @ mx.array(np.array([1.0], dtype=np.float16))
