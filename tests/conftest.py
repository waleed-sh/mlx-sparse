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

import os

import numpy as np
import pytest


@pytest.fixture
def mlx_test_device(request):
    mx = pytest.importorskip("mlx.core")
    requested = os.environ.get("MLX_SPARSE_TEST_DEVICE", "auto").lower()
    if request.node.get_closest_marker("cpu_only"):
        candidates = [("cpu", mx.cpu)]
    elif request.node.get_closest_marker("gpu"):
        candidates = [("gpu", mx.gpu)]
    elif requested == "auto":
        candidates = [("gpu", mx.gpu), ("cpu", mx.cpu)]
    elif requested == "gpu":
        candidates = [("gpu", mx.gpu)]
    elif requested == "cpu":
        candidates = [("cpu", mx.cpu)]
    else:
        raise ValueError(
            "MLX_SPARSE_TEST_DEVICE must be 'auto', 'gpu', or 'cpu', "
            f"got {requested!r}."
        )

    failures: list[str] = []
    for name, kind in candidates:
        try:
            device = mx.Device(kind, 0)
            if not mx.is_available(device):
                failures.append(f"{name}: mx.is_available returned False")
                continue
            mx.set_default_device(device)
            probe = mx.array(np.array([0], dtype=np.float32))
            mx.eval(probe)
            return device
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    pytest.skip("No usable MLX device found. " + " | ".join(failures))


@pytest.fixture
def mx(mlx_test_device):
    import mlx.core as mx

    mx.set_default_device(mlx_test_device)
    return mx


@pytest.fixture
def scipy_sparse():
    return pytest.importorskip("scipy.sparse")


@pytest.fixture
def to_numpy():
    from mlx_sparse._host import to_numpy as host_to_numpy

    return host_to_numpy
