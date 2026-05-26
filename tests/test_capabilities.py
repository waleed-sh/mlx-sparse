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

import sys
from types import SimpleNamespace

import pytest

import mlx_sparse as ms
import mlx_sparse._capabilities as _capabilities
from mlx_sparse._ext_loader import extension


def test_public_capabilities_are_simple_booleans_and_strings():
    caps = ms.capabilities

    assert isinstance(caps.platform, str)
    assert isinstance(caps.architecture, str)
    assert caps.names == ("extension", "cpu", "metal", "accelerate", "cuda", "rocm")

    for name in caps.names:
        assert isinstance(caps.has(name), bool)
        assert caps.status(name) in {"available", "unavailable", "not_built"}
        assert isinstance(caps.reason(name), str)
        assert isinstance(caps.built(name), bool)
        assert isinstance(caps.runtime_available(name), bool)


def test_capability_uppercase_properties_match_has_checks():
    caps = ms.capabilities

    assert caps.extension is caps.EXTENSION
    assert caps.cpu is caps.CPU
    assert caps.metal is caps.METAL
    assert caps.accelerate is caps.ACCELERATE
    assert caps.cuda is caps.CUDA
    assert caps.rocm is caps.ROCM
    assert caps.EXTENSION is caps.has("extension")
    assert caps.CPU is caps.has("cpu")
    assert caps.METAL is caps.has("metal")
    assert caps.ACCELERATE is caps.has("accelerate")
    assert caps.CUDA is caps.has("cuda")
    assert caps.ROCM is caps.has("rocm")


def test_has_capability_matches_capability_view():
    for name in ms.capabilities.names:
        assert ms.has_capability(name) is ms.capabilities.has(name)


def test_capability_view_repr_is_informative():
    rendered = repr(ms.capabilities)

    assert rendered.startswith("mlx_sparse.capabilities(")
    for name in ms.capabilities.names:
        assert f"{name}=" in rendered


def test_top_level_exports_do_not_include_capability_implementation_classes():
    assert "capabilities" in ms.__all__
    assert "has_capability" in ms.__all__

    for name in (
        "NativeBackend",
        "NativeCapabilities",
        "NativeCapability",
        "NativeCapabilityRecord",
        "NativeCapabilityStatus",
        "native_capabilities",
        "has_native_capability",
    ):
        assert name not in ms.__all__
        assert not hasattr(ms, name)


def test_capability_aliases_are_supported():
    caps = ms.capabilities

    assert caps.has("gpu") is caps.has("metal")
    assert caps.has("metal-kernels") is caps.has("metal")
    assert caps.has(_capabilities.NativeCapability.CPU_KERNELS) is caps.has("cpu")
    assert caps.status("cpu_kernels") == caps.status("cpu")
    assert caps.status("native_extension") == caps.status("extension")
    assert caps.status("accelerate_solvers") == caps.status("accelerate")
    assert caps.status("hip") == caps.status("rocm")


def test_invalid_capability_names_fail_loudly():
    with pytest.raises(ValueError):
        ms.capabilities.has("opencl")

    with pytest.raises(ValueError):
        ms.capabilities.status("not-a-backend")

    with pytest.raises(ValueError):
        ms.has_capability("made_up_backend")


def test_internal_enum_snapshot_remains_consistent():
    snapshot = _capabilities._native_capabilities()

    assert {record.capability for record in snapshot} == set(
        _capabilities.NativeCapability
    )
    assert all(
        isinstance(record.backend, (_capabilities.NativeBackend, type(None)))
        for record in snapshot
    )


def test_internal_snapshot_lookup_backend_and_available_invariants():
    snapshot = _capabilities._native_capabilities()

    cpu_record = snapshot[_capabilities.NativeCapability.CPU_KERNELS]
    assert snapshot.get("cpu") == cpu_record
    assert ("cpu" in snapshot) is cpu_record.available
    assert snapshot.status("cpu") is cpu_record.status
    assert snapshot.by_backend("cpu") == snapshot.by_backend(
        _capabilities.NativeBackend.CPU
    )
    assert snapshot.available == frozenset(
        record.capability for record in snapshot if record.available
    )

    empty = _capabilities.NativeCapabilities(
        records=(), platform="test", architecture="test"
    )
    with pytest.raises(KeyError):
        empty.get(_capabilities.NativeCapability.CPU_KERNELS)


def test_capabilities_without_loaded_extension(monkeypatch):
    monkeypatch.setattr(_capabilities, "extension", lambda: None)

    caps = _capabilities.capabilities

    assert caps.status("extension") == "unavailable"
    assert not caps.EXTENSION
    assert not caps.CPU
    assert not caps.METAL
    assert caps.status("accelerate") == "unavailable"
    assert caps.status("cuda") == "unavailable"
    assert caps.status("rocm") == "unavailable"


def test_older_extension_without_compiled_fact_hook_uses_resource_fallback(monkeypatch):
    monkeypatch.setattr(_capabilities, "extension", lambda: object())
    monkeypatch.setattr(_capabilities, "_metallib_present", lambda: True)

    facts = _capabilities._compiled_facts()

    assert facts["extension"] is True
    assert facts["cpu"] is True
    assert facts["metal"] is True
    assert facts["accelerate"] is False
    assert facts["cuda"] is False
    assert facts["rocm"] is False


def test_not_built_compile_facts_surface_as_not_built(monkeypatch):
    monkeypatch.setattr(
        _capabilities,
        "_compiled_facts",
        lambda: {
            "extension": True,
            "cpu": False,
            "metal": False,
            "accelerate": False,
            "cuda": False,
            "rocm": False,
            "platform": "darwin",
            "architecture": "arm64",
        },
    )

    assert _capabilities.capabilities.status("cpu") == "not_built"
    assert _capabilities.capabilities.status("metal") == "not_built"
    assert _capabilities.capabilities.status("accelerate") == "not_built"


def test_future_compiled_backends_keep_distinct_runtime_status(monkeypatch):
    monkeypatch.setattr(
        _capabilities,
        "_compiled_facts",
        lambda: {
            "extension": True,
            "cpu": False,
            "metal": False,
            "accelerate": True,
            "cuda": True,
            "rocm": True,
            "platform": "darwin",
            "architecture": "arm64",
        },
    )

    assert _capabilities.capabilities.status("accelerate") == "available"
    assert _capabilities.capabilities.runtime_available("accelerate")
    assert _capabilities.capabilities.status("cuda") == "unavailable"
    assert _capabilities.capabilities.status("rocm") == "unavailable"
    assert "compiled but not available" in _capabilities.capabilities.reason("cuda")


def test_probe_mlx_device_reports_each_runtime_failure(monkeypatch):
    def raising_device(kind, index):
        raise RuntimeError("bad device")

    monkeypatch.setattr(
        _capabilities,
        "mx",
        SimpleNamespace(Device=raising_device, is_available=None, device_info=None),
    )
    available, reason = _capabilities._probe_mlx_device("kind", "Test")
    assert not available
    assert "Could not construct MLX Test device" in reason

    monkeypatch.setattr(
        _capabilities,
        "mx",
        SimpleNamespace(
            Device=lambda kind, index: "device",
            is_available=lambda device: False,
            device_info=lambda device: {},
        ),
    )
    available, reason = _capabilities._probe_mlx_device("kind", "Test")
    assert not available
    assert "reports that Test device 0 is unavailable" in reason

    def raising_is_available(device):
        raise RuntimeError("availability failed")

    monkeypatch.setattr(
        _capabilities,
        "mx",
        SimpleNamespace(
            Device=lambda kind, index: "device",
            is_available=raising_is_available,
            device_info=lambda device: {},
        ),
    )
    available, reason = _capabilities._probe_mlx_device("kind", "Test")
    assert not available
    assert "Could not query MLX Test availability" in reason

    def raising_device_info(device):
        raise RuntimeError("device info failed")

    monkeypatch.setattr(
        _capabilities,
        "mx",
        SimpleNamespace(
            Device=lambda kind, index: "device",
            is_available=lambda device: True,
            device_info=raising_device_info,
        ),
    )
    available, reason = _capabilities._probe_mlx_device("kind", "Test")
    assert not available
    assert "could not initialize Test device 0" in reason

    monkeypatch.setattr(
        _capabilities,
        "mx",
        SimpleNamespace(
            Device=lambda kind, index: "device",
            is_available=lambda device: True,
            device_info=lambda device: {"device_name": "test"},
        ),
    )
    assert _capabilities._probe_mlx_device("kind", "Test") == (True, "")


def test_metallib_resource_probe_handles_importlib_errors(monkeypatch):
    def raising_files(package):
        raise RuntimeError("resource unavailable")

    monkeypatch.setattr(_capabilities.resources, "files", raising_files)

    assert not _capabilities._metallib_present()


def test_python_platform_normalization(monkeypatch):
    for raw, expected in (
        ("darwin", "darwin"),
        ("linux", "linux"),
        ("linux2", "linux"),
        ("win32", "windows"),
        ("freebsd14", "freebsd14"),
    ):
        monkeypatch.setattr(sys, "platform", raw)
        assert _capabilities._python_platform() == expected


@pytest.mark.native
def test_native_extension_reports_compiled_capability_facts():
    ext = extension()
    if ext is None:
        pytest.skip("native extension is not built")

    facts = ext._compiled_capabilities()

    assert facts["extension"] is True
    assert facts["cpu"] is True
    assert facts["accelerate"] is False
    assert facts["cuda"] is False
    assert facts["rocm"] is False
    assert facts["platform"] in {"darwin", "linux", "windows", "unknown"}
    assert isinstance(facts["architecture"], str)
