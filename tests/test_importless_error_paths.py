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

import builtins
from pathlib import Path

import pytest

import mlx_sparse as ms
from mlx_sparse import _ext_loader

ROOT = Path(__file__).resolve().parents[1]


def test_shape_and_validation_errors_fail_before_array_materialization():
    with pytest.raises(ValueError, match="non-negative"):
        ms.eye(-1)

    with pytest.raises(ValueError, match="rank-2"):
        ms.csr_array(object(), shape=(1,))

    with pytest.raises(ValueError, match="validate must be"):
        ms.csr_array(object(), shape=(1, 1), validate="deep")

    with pytest.raises(ValueError, match="validate must be"):
        ms.coo_array(object(), shape=(1, 1), validate="deep")


def test_from_scipy_reports_missing_scipy_dependency(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.sparse":
            raise ImportError("scipy intentionally hidden")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(TypeError, match="requires scipy"):
        ms.from_scipy(object())


def test_extension_availability_helper_returns_a_boolean():
    assert isinstance(_ext_loader.extension_available(), bool)


def test_readthedocs_build_installs_docs_requirements_only():
    rtd_config = (ROOT / ".readthedocs.yaml").read_text()
    docs_requirements = (ROOT / "docs" / "requirements.txt").read_text()

    assert "configuration: docs/conf.py" in rtd_config
    assert "requirements: docs/requirements.txt" in rtd_config
    assert "myst-nb>=1.4" in docs_requirements
    assert "mlx>=" not in docs_requirements
