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

    assert caps.EXTENSION is caps.has("extension")
    assert caps.CPU is caps.has("cpu")
    assert caps.METAL is caps.has("metal")
    assert caps.ACCELERATE is caps.has("accelerate")
    assert caps.CUDA is caps.has("cuda")
    assert caps.ROCM is caps.has("rocm")


def test_has_capability_matches_capability_view():
    for name in ms.capabilities.names:
        assert ms.has_capability(name) is ms.capabilities.has(name)


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
    assert caps.status("cpu_kernels") == caps.status("cpu")
    assert caps.status("native_extension") == caps.status("extension")
    assert caps.status("accelerate_solvers") == caps.status("accelerate")
    assert caps.status("hip") == caps.status("rocm")


def test_internal_enum_snapshot_remains_consistent():
    snapshot = _capabilities._native_capabilities()

    assert {record.capability for record in snapshot} == set(
        _capabilities.NativeCapability
    )
    assert all(
        isinstance(record.backend, (_capabilities.NativeBackend, type(None)))
        for record in snapshot
    )


def test_capabilities_without_loaded_extension(monkeypatch):
    monkeypatch.setattr(_capabilities, "extension", lambda: None)

    caps = _capabilities.capabilities

    assert caps.status("extension") == "unavailable"
    assert not caps.EXTENSION
    assert not caps.CPU
    assert not caps.METAL


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
