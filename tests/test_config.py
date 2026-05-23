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
import subprocess
import sys

import pytest

import mlx_sparse as ms

OPTION = "EXPERIMENTAL_METAL_SPGEMM"
ENV_KEY = "MLX_SPARSE_EXPERIMENTAL_METAL_SPGEMM"


@pytest.fixture(autouse=True)
def restore_config_state():
    overrides = ms.config.user_overrides()
    had_override = OPTION in overrides
    old_override = overrides.get(OPTION)
    old_env = os.environ.get(ENV_KEY)

    yield

    if had_override:
        ms.config.set(OPTION, old_override)
    else:
        ms.config.clear_override(OPTION)

    if old_env is None:
        os.environ.pop(ENV_KEY, None)
    else:
        os.environ[ENV_KEY] = old_env


def test_config_public_exports_and_default_sync():
    assert OPTION in ms.config.list_options()
    assert OPTION in ms.config.options_by_role()["sparse"]

    assert ms.get_config(OPTION) is False
    assert os.environ[ENV_KEY] == "0"
    for name in (
        "config",
        "config_context",
        "get_config",
        "set_config",
    ):
        assert name in ms.__all__


def test_config_does_not_export_internal_manager_api():
    namespace: dict[str, object] = {}
    exec("from mlx_sparse import *", namespace)

    for name in (
        "cfg",
        "ConfigManager",
        "ConfigError",
        "UnknownOptionError",
        "ConfigValidationError",
        "ConfigMutability",
        "ConfigSource",
        "ConfigMutation",
        "ConfigOption",
    ):
        assert name not in ms.__all__
        assert not hasattr(ms, name)
        assert name not in namespace


def test_set_config_updates_attribute_and_native_env_flag():
    assert ms.set_config(OPTION, True) is True
    assert ms.config.EXPERIMENTAL_METAL_SPGEMM is True
    assert os.environ[ENV_KEY] == "1"

    ms.config.EXPERIMENTAL_METAL_SPGEMM = False
    assert ms.get_config(OPTION) is False
    assert os.environ[ENV_KEY] == "0"


def test_config_patch_restores_value_and_env():
    ms.config.EXPERIMENTAL_METAL_SPGEMM = False

    with ms.config.patch(EXPERIMENTAL_METAL_SPGEMM=True):
        assert ms.config.EXPERIMENTAL_METAL_SPGEMM is True
        assert os.environ[ENV_KEY] == "1"

    assert ms.config.EXPERIMENTAL_METAL_SPGEMM is False
    assert os.environ[ENV_KEY] == "0"

    with ms.config_context(OPTION, True):
        assert ms.config.EXPERIMENTAL_METAL_SPGEMM is True

    assert ms.config.EXPERIMENTAL_METAL_SPGEMM is False


def test_config_metadata_snapshot_show_and_fingerprint():
    ms.config.EXPERIMENTAL_METAL_SPGEMM = True

    snapshot = ms.config.snapshot()
    assert snapshot[OPTION] is True

    description = ms.config.describe_option(OPTION)
    assert description["name"] == OPTION
    assert description["effective_value"] is True
    assert description["source"] == "user"
    assert description["env_default"] == (ENV_KEY,)

    table = ms.config.show()
    assert OPTION in table
    assert "effective" in table

    first = ms.config.fingerprint()
    ms.config.EXPERIMENTAL_METAL_SPGEMM = False
    assert ms.config.fingerprint() != first
