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


def _available_devices(mx):
    devices = []
    for name, kind in (("cpu", mx.cpu), ("gpu", mx.gpu)):
        try:
            device = mx.Device(kind, 0)
            if not mx.is_available(device):
                continue
            mx.set_default_device(device)
            probe = mx.array(np.array([0.0], dtype=np.float32))
            mx.eval(probe)
            devices.append((name, device))
        except Exception:
            continue
    if not devices:
        pytest.skip("No usable MLX CPU or GPU device found.")
    return devices


def _sample_csr(mx):
    data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
    return ms.csr_array(
        (data, indices, indptr),
        shape=(3, 4),
        sorted_indices=True,
        canonical=True,
    )


@pytest.mark.gpu
def test_sparse_dense_ops_match_dense_mlx_on_available_devices():
    mx = pytest.importorskip("mlx.core")
    x_np = np.array([3.0, 10.0, 7.0, -2.0], dtype=np.float32)
    rhs_np = np.arange(20, dtype=np.float32).reshape(4, 5) / 10.0

    for _, device in _available_devices(mx):
        mx.set_default_device(device)
        csr = _sample_csr(mx)
        dense = csr.todense()
        x = mx.array(x_np)
        rhs = mx.array(rhs_np)

        np.testing.assert_allclose(np.array(csr @ x), np.array(dense @ x))
        np.testing.assert_allclose(np.array(csr @ rhs), np.array(dense @ rhs))
        np.testing.assert_allclose(np.array(csr.T.todense()), np.array(dense.T))


@pytest.mark.gpu
def test_coo_conversion_matches_dense_on_available_devices():
    mx = pytest.importorskip("mlx.core")
    for _, device in _available_devices(mx):
        mx.set_default_device(device)
        coo = ms.coo_array(
            (
                mx.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
                (
                    mx.array(np.array([0, 0, 2, 1], dtype=np.int32)),
                    mx.array(np.array([2, 0, 1, 3], dtype=np.int32)),
                ),
            ),
            shape=(3, 4),
        )
        csr = coo.tocsr()
        expected = np.array(
            [[2.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 4.0], [0.0, 3.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.array(csr.todense()), expected)
