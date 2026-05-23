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

import pytest

import mlx_sparse as ms
from mlx_sparse._config import (
    ConfigError,
    ConfigMutability,
    ConfigOption,
    ConfigSource,
    ConfigValidationError,
    UnknownOptionError,
    config,
    parse_bool,
)

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


def test_public_issparse_rejects_plain_python_objects():
    assert ms.issparse(object()) is False


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


@pytest.mark.parametrize(
    "value",
    [True, 1, "1", "true", "T", "yes", "Y", "on"],
)
def test_parse_bool_truthy_values(value):
    assert parse_bool(value) is True


@pytest.mark.parametrize(
    "value",
    [False, 0, "0", "false", "F", "no", "N", "off"],
)
def test_parse_bool_falsey_values(value):
    assert parse_bool(value) is False


def test_parse_bool_rejects_unknown_value():
    with pytest.raises(ConfigValidationError, match="Cannot parse boolean"):
        parse_bool("absolutely")


def test_config_unknown_attribute_and_option_errors():
    with pytest.raises(AttributeError, match="no option"):
        _ = ms.config.DOES_NOT_EXIST

    with pytest.raises(AttributeError, match="no option"):
        ms.config.DOES_NOT_EXIST = True

    with pytest.raises(UnknownOptionError, match="Unknown option"):
        ms.config.get("DOES_NOT_EXIST")


def test_config_register_option_env_defaults_hooks_and_sources(monkeypatch):
    name = "TEST_COVERAGE_ENV_DEFAULT"
    env_key = "MLX_SPARSE_TEST_COVERAGE_ENV_DEFAULT_VALUE"
    monkeypatch.setenv(env_key.lower(), "yes")

    config.register_option(
        ConfigOption(
            name=name.lower(),
            default=False,
            doc="coverage-only env default option",
            value_type=bool,
            parser=parse_bool,
            env_default=env_key,
            role="coverage",
        )
    )

    assert config.get(name) is True
    assert config.read(name) is True
    assert config.value_source(name) is ConfigSource.ENV_DEFAULT
    assert config.TEST_COVERAGE_ENV_DEFAULT is True
    assert os.environ["MLX_SPARSE_" + name] == "1"
    assert name in config.options_by_role()["coverage"]
    assert "runtime_locked" in repr(config)

    seen = []
    config.add_hook(name, seen.append, run_immediately=True)
    config.add_global_hook(seen.append)
    assert seen[-1].name == name
    assert seen[-1].old_value is True
    assert seen[-1].new_value is True

    assert config.set(name, False) is False
    assert config.value_source(name) is ConfigSource.USER
    assert seen[-2].source is ConfigSource.USER
    assert seen[-1].source is ConfigSource.USER

    assert config.clear_override(name) is True
    assert config.user_overrides().get(name) is None


def test_config_forced_env_and_mutability_rules(monkeypatch):
    forced_name = "TEST_COVERAGE_FORCED"
    forced_env = "MLX_SPARSE_TEST_COVERAGE_FORCED_VALUE"
    monkeypatch.setenv(forced_env, "1")

    config.define_bool(
        forced_name,
        default=False,
        doc="coverage-only forced option",
        env_force=forced_env,
        role="coverage",
    )

    assert config.get(forced_name) is True
    assert config.value_source(forced_name) is ConfigSource.ENV_FORCE
    with pytest.raises(ConfigError, match="forced by environment"):
        config.set(forced_name, False)

    immutable_name = "TEST_COVERAGE_IMMUTABLE"
    config.define_bool(
        immutable_name,
        default=False,
        doc="coverage-only immutable option",
        mutability=ConfigMutability.IMMUTABLE,
        role="coverage",
    )
    with pytest.raises(ConfigError, match="immutable"):
        config.set(immutable_name, True)

    startup_name = "TEST_COVERAGE_STARTUP"
    config.define_bool(
        startup_name,
        default=False,
        doc="coverage-only startup option",
        mutability=ConfigMutability.STARTUP,
        role="coverage",
    )
    assert config.runtime_locked is False
    config.lock_runtime()
    try:
        assert config.runtime_locked is True
        with pytest.raises(ConfigError, match="startup-only"):
            config.set(startup_name, True)
    finally:
        config.unlock_runtime_for_testing()
    assert config.runtime_locked is False


def test_config_register_option_validation_errors():
    with pytest.raises(ConfigValidationError, match="non-empty"):
        config.define_bool("", default=False, doc="bad")

    with pytest.raises(ConfigValidationError, match="Environment names"):
        config.define_bool(
            "TEST_COVERAGE_BAD_ENV",
            default=False,
            doc="bad env",
            env_default=[""],
        )

    with pytest.raises(ConfigValidationError, match="expects"):
        config.register_option(
            ConfigOption(
                name="TEST_COVERAGE_BAD_DEFAULT",
                default="not-an-int",
                doc="bad default",
                value_type=int,
            )
        )

    with pytest.raises(ConfigValidationError, match="already registered"):
        config.define_bool(
            "EXPERIMENTAL_METAL_SPGEMM",
            default=False,
            doc="duplicate",
        )


def test_config_patch_call_forms_restore_previous_values():
    name = "TEST_COVERAGE_PATCH_FORMS"
    config.define_bool(
        name,
        default=False,
        doc="coverage-only patch option",
        role="coverage",
    )

    with config.patch(name, True):
        assert config.get(name) is True
    assert config.get(name) is False

    config.set(name, True)
    with config.patch({name: False}):
        assert config.get(name) is False
    assert config.get(name) is True
    config.clear_override(name)

    with config.patch(**{name: True}):
        assert config.get(name) is True
    assert config.get(name) is False


@pytest.mark.parametrize(
    "call",
    [
        lambda: config.patch(None, True),
        lambda: config.patch("EXPERIMENTAL_METAL_SPGEMM"),
        lambda: config.patch("EXPERIMENTAL_METAL_SPGEMM", True, other=False),
        lambda: config.patch({"EXPERIMENTAL_METAL_SPGEMM": True}, False),
        lambda: config.patch({"EXPERIMENTAL_METAL_SPGEMM": True}, other=False),
        lambda: config.patch(123),
    ],
)
def test_config_patch_rejects_invalid_call_forms(call):
    with pytest.raises(TypeError):
        with call():
            pass
