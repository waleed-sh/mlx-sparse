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

import pytest

import mlx_sparse._device as device_api


class _FakeDevice:
    def __init__(self, kind, index):
        self.kind = kind
        self.index = index


class _FakeMx:
    cpu = "cpu"
    gpu = "gpu"
    Device = _FakeDevice

    def __init__(self, *, available=True, fail_eval=False):
        self.available = available
        self.fail_eval = fail_eval
        self.default_device = None

    def set_default_device(self, device):
        self.default_device = device

    def is_available(self, device):
        return self.available

    def array(self, values):
        return list(values)

    def eval(self, array):
        if self.fail_eval:
            raise RuntimeError("probe failed")


def test_use_cpu_sets_default_device_and_optionally_probes(monkeypatch):
    fake_mx = _FakeMx()
    monkeypatch.setattr(device_api, "mx", fake_mx)

    device = device_api.use_cpu(index=1)

    assert device.kind == "cpu"
    assert device.index == 1
    assert fake_mx.default_device is device

    unprobed_mx = _FakeMx(fail_eval=True)
    monkeypatch.setattr(device_api, "mx", unprobed_mx)
    unprobed = device_api.use_cpu(require_available=False)

    assert unprobed.kind == "cpu"
    assert unprobed_mx.default_device is unprobed


def test_use_cpu_wraps_probe_failures(monkeypatch):
    monkeypatch.setattr(device_api, "mx", _FakeMx(fail_eval=True))

    with pytest.raises(RuntimeError, match="CPU device 0 is not available"):
        device_api.use_cpu()


def test_use_gpu_checks_availability_and_probe(monkeypatch):
    fake_mx = _FakeMx()
    monkeypatch.setattr(device_api, "mx", fake_mx)

    device = device_api.use_gpu(index=2)

    assert device.kind == "gpu"
    assert device.index == 2
    assert fake_mx.default_device is device

    unprobed_mx = _FakeMx(available=False, fail_eval=True)
    monkeypatch.setattr(device_api, "mx", unprobed_mx)
    unprobed = device_api.use_gpu(require_available=False)

    assert unprobed.kind == "gpu"
    assert unprobed_mx.default_device is unprobed


def test_use_gpu_rejects_unavailable_or_unprobeable_devices(monkeypatch):
    monkeypatch.setattr(device_api, "mx", _FakeMx(available=False))
    with pytest.raises(RuntimeError, match="GPU device 0 is not available"):
        device_api.use_gpu()

    monkeypatch.setattr(device_api, "mx", _FakeMx(fail_eval=True))
    with pytest.raises(RuntimeError, match="GPU device 0 is not available"):
        device_api.use_gpu()


def test_use_device_dispatches_by_name(monkeypatch):
    calls = []

    def fake_cpu(index):
        calls.append(("cpu", index))
        return "cpu-device"

    def fake_gpu(index):
        calls.append(("gpu", index))
        return "gpu-device"

    monkeypatch.setattr(device_api, "use_cpu", fake_cpu)
    monkeypatch.setattr(device_api, "use_gpu", fake_gpu)

    assert device_api.use_device("CPU", index=3) == "cpu-device"
    assert device_api.use_device("gpu", index=4) == "gpu-device"
    assert calls == [("cpu", 3), ("gpu", 4)]

    with pytest.raises(ValueError, match="device must be 'cpu' or 'gpu'"):
        device_api.use_device("tpu")
