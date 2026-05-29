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
from mlx_sparse._config import ConfigValidationError

RUNTIME_OPTIONS = (
    "CPU_THREADS",
    "SPGEMM_PARALLEL",
    "SPGEMM_THREADS",
    "SOLVER_PARALLEL",
    "SOLVER_THREADS",
)
RUNTIME_ENV = (
    "MLX_SPARSE_CPU_THREADS",
    "MLX_SPARSE_N_THREADS",
    "MLX_SPARSE_SPGEMM_PARALLEL",
    "MLX_SPARSE_SPGEMM_THREADS",
    "MLX_SPARSE_SOLVER_PARALLEL",
    "MLX_SPARSE_SOLVER_THREADS",
    "OMP_NUM_THREADS",
    "SLURM_CPUS_PER_TASK",
    "PBS_NP",
    "LSB_DJOB_NUMPROC",
    "NSLOTS",
)


@pytest.fixture(autouse=True)
def restore_runtime_state():
    overrides = ms.config.user_overrides()
    saved_overrides = {
        name: overrides[name] for name in RUNTIME_OPTIONS if name in overrides
    }
    saved_env = {name: os.environ.get(name) for name in RUNTIME_ENV}

    ms.runtime.N_THREADS = "auto"
    ms.runtime.SPGEMM_PARALLEL = True
    ms.runtime.SPGEMM_THREADS = "inherit"
    ms.runtime.SOLVER_PARALLEL = False
    ms.runtime.SOLVER_THREADS = "inherit"
    for name in RUNTIME_ENV:
        if not name.startswith("MLX_SPARSE_"):
            os.environ.pop(name, None)

    yield

    for name in RUNTIME_OPTIONS:
        if name in saved_overrides:
            ms.config.set(name, saved_overrides[name])
        else:
            ms.config.clear_override(name)

    for name, value in saved_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_runtime_public_exports_and_option_roles():
    assert ms.runtime.RuntimeOption.N_THREADS.value == "CPU_THREADS"
    assert ms.runtime.RuntimeOption.SPGEMM_PARALLEL.value == "SPGEMM_PARALLEL"
    assert ms.runtime.RuntimeOption.SPGEMM_THREADS.value == "SPGEMM_THREADS"
    assert ms.runtime.RuntimeOption.SOLVER_PARALLEL.value == "SOLVER_PARALLEL"
    assert ms.runtime.RuntimeOption.SOLVER_THREADS.value == "SOLVER_THREADS"

    assert ms.runtime.N_THREADS == ms.runtime.resolve_n_threads()[0]
    assert ms.runtime.SPGEMM_PARALLEL is True
    assert ms.runtime.SPGEMM_THREADS == ms.runtime.resolve_spgemm_threads()[0]
    assert ms.runtime.SOLVER_PARALLEL is False
    assert ms.runtime.SOLVER_THREADS == ms.runtime.resolve_solver_threads()[0]
    assert ms.config.CPU_THREADS == "auto"
    assert ms.config.SPGEMM_THREADS == "inherit"
    assert ms.config.SOLVER_THREADS == "inherit"

    roles = ms.config.options_by_role()
    assert set(RUNTIME_OPTIONS).issubset(roles["runtime"])
    assert "runtime" in ms.__all__

    namespace: dict[str, object] = {}
    exec("from mlx_sparse import *", namespace)
    assert namespace["runtime"] is ms.runtime

    runtime_namespace: dict[str, object] = {}
    exec("from mlx_sparse.runtime import *", runtime_namespace)
    assert runtime_namespace["N_THREADS"] == ms.runtime.N_THREADS
    assert runtime_namespace["RuntimeOption"] is ms.runtime.RuntimeOption


def test_runtime_direct_attribute_assignment_syncs_config_and_environment():
    ms.runtime.N_THREADS = 4
    ms.runtime.SPGEMM_PARALLEL = False
    ms.runtime.SPGEMM_THREADS = 6
    ms.runtime.SOLVER_PARALLEL = True
    ms.runtime.SOLVER_THREADS = 2

    assert ms.runtime.N_THREADS == 4
    assert ms.runtime.SPGEMM_PARALLEL is False
    assert ms.runtime.SPGEMM_THREADS == 1
    assert ms.runtime.SOLVER_PARALLEL is True
    assert ms.runtime.SOLVER_THREADS == 2

    assert ms.config.CPU_THREADS == 4
    assert ms.config.SPGEMM_PARALLEL is False
    assert ms.config.SPGEMM_THREADS == 6
    assert ms.config.SOLVER_PARALLEL is True
    assert ms.config.SOLVER_THREADS == 2

    assert os.environ["MLX_SPARSE_CPU_THREADS"] == "4"
    assert os.environ["MLX_SPARSE_SPGEMM_PARALLEL"] == "0"
    assert os.environ["MLX_SPARSE_SPGEMM_THREADS"] == "6"
    assert os.environ["MLX_SPARSE_SOLVER_PARALLEL"] == "1"
    assert os.environ["MLX_SPARSE_SOLVER_THREADS"] == "2"

    ms.runtime.SPGEMM_PARALLEL = True
    ms.runtime.SPGEMM_THREADS = "inherit"
    assert ms.runtime.SPGEMM_THREADS == 4


def test_runtime_context_call_forms_restore_values_and_environment():
    ms.runtime.N_THREADS = 2

    with ms.runtime.context(ms.runtime.RuntimeOption.N_THREADS, 4):
        assert ms.runtime.N_THREADS == 4
        assert os.environ["MLX_SPARSE_CPU_THREADS"] == "4"
    assert ms.runtime.N_THREADS == 2
    assert os.environ["MLX_SPARSE_CPU_THREADS"] == "2"

    with ms.runtime.context(
        {
            ms.runtime.RuntimeOption.N_THREADS: 6,
            ms.runtime.RuntimeOption.SPGEMM_PARALLEL: False,
            ms.runtime.RuntimeOption.SPGEMM_THREADS: 3,
            "solver_parallel": True,
            "solver_threads": 4,
        }
    ):
        assert ms.runtime.N_THREADS == 6
        assert ms.runtime.SPGEMM_THREADS == 1
        assert ms.runtime.SOLVER_THREADS == 4

    assert ms.runtime.N_THREADS == 2
    assert ms.runtime.SPGEMM_PARALLEL is True
    assert ms.config.SPGEMM_THREADS == "inherit"
    assert ms.runtime.SOLVER_PARALLEL is False
    assert ms.config.SOLVER_THREADS == "inherit"

    with ms.runtime.patch(n_threads="auto", spgemm_parallel=False, spgemm_threads=8):
        assert ms.config.CPU_THREADS == "auto"
        assert ms.runtime.SPGEMM_PARALLEL is False
        assert ms.config.SPGEMM_THREADS == 8

    assert ms.runtime.N_THREADS == 2
    assert ms.runtime.SPGEMM_PARALLEL is True
    assert ms.config.SPGEMM_THREADS == "inherit"


@pytest.mark.parametrize("value", [True, 0, -1, "0", "-2", "many"])
def test_runtime_rejects_invalid_thread_counts(value):
    with pytest.raises(ConfigValidationError, match="Thread count"):
        ms.runtime.N_THREADS = value
    with pytest.raises(ConfigValidationError, match="Thread count"):
        ms.runtime.SPGEMM_THREADS = value
    with pytest.raises(ConfigValidationError, match="Thread count"):
        ms.runtime.SOLVER_THREADS = value


def test_runtime_direct_attribute_assignment_validates_values():
    with pytest.raises(ConfigValidationError, match="Thread count"):
        ms.runtime.N_THREADS = 0
    with pytest.raises(ConfigValidationError, match="Thread count"):
        ms.runtime.SPGEMM_THREADS = "many"
    with pytest.raises(ConfigValidationError, match="Cannot parse boolean"):
        ms.runtime.SOLVER_PARALLEL = "sometimes"


def test_runtime_accepts_inherited_family_thread_counts():
    ms.runtime.SPGEMM_THREADS = "inherit"
    ms.runtime.SOLVER_THREADS = "inherit"
    assert ms.config.SPGEMM_THREADS == "inherit"
    assert ms.config.SOLVER_THREADS == "inherit"


@pytest.mark.parametrize(
    "call",
    [
        lambda: ms.runtime.context(None, 1),
        lambda: ms.runtime.context(ms.runtime.RuntimeOption.N_THREADS),
        lambda: ms.runtime.context(
            ms.runtime.RuntimeOption.N_THREADS, 1, spgemm_parallel=False
        ),
        lambda: ms.runtime.context({ms.runtime.RuntimeOption.N_THREADS: 1}, 2),
        lambda: ms.runtime.context(
            {ms.runtime.RuntimeOption.N_THREADS: 1}, spgemm_parallel=False
        ),
    ],
)
def test_runtime_context_rejects_invalid_call_forms(call):
    with pytest.raises(TypeError):
        with call():
            pass


def test_runtime_resolves_explicit_config_before_external_hints(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    ms.runtime.N_THREADS = 4

    assert ms.runtime.resolve_n_threads() == (4, "configured")
    assert ms.runtime.N_THREADS == 4


def test_runtime_auto_resolution_precedence(monkeypatch):
    ms.runtime.N_THREADS = "auto"
    monkeypatch.setenv("OMP_NUM_THREADS", "5,3")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    monkeypatch.setattr(ms.runtime, "_affinity_count", lambda: 6)
    monkeypatch.setattr(ms.runtime, "_hardware_concurrency", lambda: 8)

    assert ms.runtime.resolve_n_threads() == (5, "OMP_NUM_THREADS")

    monkeypatch.delenv("OMP_NUM_THREADS")
    assert ms.runtime.resolve_n_threads() == (12, "SLURM_CPUS_PER_TASK")

    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert ms.runtime.resolve_n_threads() == (6, "process_affinity")

    monkeypatch.setattr(ms.runtime, "_affinity_count", lambda: None)
    assert ms.runtime.resolve_n_threads() == (8, "hardware_concurrency")

    monkeypatch.setattr(ms.runtime, "_hardware_concurrency", lambda: None)
    assert ms.runtime.resolve_n_threads() == (1, "fallback")


def test_runtime_parallel_family_thread_gates():
    ms.runtime.N_THREADS = 7
    ms.runtime.SPGEMM_PARALLEL = True
    ms.runtime.SOLVER_PARALLEL = False

    assert ms.runtime.SPGEMM_THREADS == 7
    assert ms.runtime.SOLVER_THREADS == 1

    ms.runtime.SPGEMM_PARALLEL = False
    ms.runtime.SOLVER_PARALLEL = True
    assert ms.runtime.SPGEMM_THREADS == 1
    assert ms.runtime.SOLVER_THREADS == 7

    ms.runtime.SPGEMM_PARALLEL = True
    ms.runtime.SPGEMM_THREADS = 3
    ms.runtime.SOLVER_THREADS = 4
    assert ms.runtime.resolve_spgemm_threads() == (3, "configured")
    assert ms.runtime.resolve_solver_threads() == (4, "configured")


def test_runtime_info_is_structured(monkeypatch):
    monkeypatch.setattr(ms.runtime, "_affinity_count", lambda: 4)
    monkeypatch.setattr(ms.runtime, "_hardware_concurrency", lambda: 8)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    ms.runtime.N_THREADS = 3
    ms.runtime.SPGEMM_PARALLEL = True
    ms.runtime.SPGEMM_THREADS = "inherit"
    ms.runtime.SOLVER_PARALLEL = False
    ms.runtime.SOLVER_THREADS = 2

    info = ms.runtime.info()

    assert info["runtime_options"] == {
        "n_threads": 3,
        "spgemm_parallel": True,
        "spgemm_threads": "inherit",
        "solver_parallel": False,
        "solver_threads": 2,
    }
    assert info["config_sources"]["n_threads"] == "user"
    assert info["n_threads"] == 3
    assert info["n_threads_source"] == "configured"
    assert info["spgemm_n_threads"] == 3
    assert info["spgemm_n_threads_source"] == "configured"
    assert info["solver_n_threads"] == 1
    assert info["solver_n_threads_source"] == "disabled"
    assert info["native_extension_available"] in {True, False}
    assert info["hardware_concurrency"] == 8
    assert info["process_affinity"] == 4
    assert info["scheduler"] == {"SLURM_CPUS_PER_TASK": 12}
    assert info["environment"]["MLX_SPARSE_CPU_THREADS"] == "3"
    assert info["mlx"]["available"] is True
    assert isinstance(info["config_fingerprint"], str)
