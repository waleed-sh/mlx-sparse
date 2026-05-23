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
from mlx_sparse import linalg


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


@pytest.mark.gpu
def test_fromdense_and_sum_duplicates_match_on_available_devices():
    mx = pytest.importorskip("mlx.core")
    dense_np = np.array(
        [[0.0, 2.0, 1e-4], [3.0, 0.0, -4.0], [0.0, 5e-4, 0.0]],
        dtype=np.float32,
    )
    reference = None

    for _, device in _available_devices(mx):
        mx.set_default_device(device)
        from_dense = ms.fromdense(mx.array(dense_np), threshold=1e-3)
        duplicate = ms.csr_array(
            (
                mx.array(np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32)),
                mx.array(np.array([2, 0, 2, 0], dtype=np.int32)),
                mx.array(np.array([0, 4], dtype=np.int32)),
            ),
            shape=(1, 3),
        ).canonicalize()
        result = {
            "fromdense": np.array(from_dense.todense()),
            "duplicate_data": np.array(duplicate.data),
            "duplicate_indices": np.array(duplicate.indices),
            "duplicate_indptr": np.array(duplicate.indptr),
        }
        if reference is None:
            reference = result
            continue
        np.testing.assert_allclose(result["fromdense"], reference["fromdense"])
        np.testing.assert_allclose(
            result["duplicate_data"], reference["duplicate_data"]
        )
        np.testing.assert_array_equal(
            result["duplicate_indices"], reference["duplicate_indices"]
        )
        np.testing.assert_array_equal(
            result["duplicate_indptr"], reference["duplicate_indptr"]
        )


@pytest.mark.gpu
def test_sparse_linalg_native_ops_match_on_available_devices():
    mx = pytest.importorskip("mlx.core")
    reference = None
    for _, device in _available_devices(mx):
        mx.set_default_device(device)
        spd = ms.csr_array(
            (
                mx.array(np.array([4.0, -1.0, -1.0, 4.0], dtype=np.float32)),
                mx.array(np.array([0, 1, 0, 1], dtype=np.int32)),
                mx.array(np.array([0, 2, 4], dtype=np.int32)),
            ),
            shape=(2, 2),
            validate="full",
            canonical=True,
        )
        rhs = mx.array(np.array([1.0, 2.0], dtype=np.float32))
        x, info = linalg.cg(spd, rhs, rtol=1e-6, maxiter=32)
        result = {
            "dot": np.array(spd.dot(spd)),
            "vdot": np.array(spd.vdot(spd)),
            "cg_info": info,
            "cg": np.array(x),
        }
        if reference is None:
            reference = result
            continue
        assert result["cg_info"] == reference["cg_info"] == 0
        np.testing.assert_allclose(result["dot"], reference["dot"], rtol=1e-6)
        np.testing.assert_allclose(result["vdot"], reference["vdot"], rtol=1e-6)
        np.testing.assert_allclose(result["cg"], reference["cg"], rtol=1e-5, atol=1e-5)
