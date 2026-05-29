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

"""Runtime controls for mlx-sparse CPU execution.

The public API exposes direct module attributes for common interactive use, and
an enum for structured programmatic use:

.. code-block:: python

   import mlx_sparse as ms

   print(ms.runtime.N_THREADS)
   ms.runtime.N_THREADS = 8
   ms.runtime.SPGEMM_PARALLEL = True

   with ms.runtime.context(n_threads=1, spgemm_parallel=False):
       C = A @ B

``N_THREADS`` is the resolved package-wide CPU worker count. It is
intentionally separate from operation-family switches and per-family thread
overrides so future kernels can share a common worker budget by default while
still allowing users to disable or tune one family independently.
"""

from __future__ import annotations

import contextlib
import enum
import os
import sys
import types
from collections.abc import Iterator, Mapping
from typing import Any

from mlx_sparse._config import config
from mlx_sparse._typing import is_available


class RuntimeOption(str, enum.Enum):
    """Runtime option identifiers accepted by :mod:`mlx_sparse.runtime`."""

    N_THREADS = "CPU_THREADS"
    SPGEMM_PARALLEL = "SPGEMM_PARALLEL"
    SPGEMM_THREADS = "SPGEMM_THREADS"
    SOLVER_PARALLEL = "SOLVER_PARALLEL"
    SOLVER_THREADS = "SOLVER_THREADS"


N_THREADS = RuntimeOption.N_THREADS
SPGEMM_PARALLEL = RuntimeOption.SPGEMM_PARALLEL
SPGEMM_THREADS = RuntimeOption.SPGEMM_THREADS
SOLVER_PARALLEL = RuntimeOption.SOLVER_PARALLEL
SOLVER_THREADS = RuntimeOption.SOLVER_THREADS

_MISSING = object()
_OPTION_ALIASES = {
    "N_THREADS": RuntimeOption.N_THREADS.value,
    "CPU_THREADS": RuntimeOption.N_THREADS.value,
    "SPGEMM_PARALLEL": RuntimeOption.SPGEMM_PARALLEL.value,
    "SPGEMM_THREADS": RuntimeOption.SPGEMM_THREADS.value,
    "SOLVER_PARALLEL": RuntimeOption.SOLVER_PARALLEL.value,
    "SOLVER_THREADS": RuntimeOption.SOLVER_THREADS.value,
}
_SCHEDULER_THREAD_ENV_VARS = (
    "SLURM_CPUS_PER_TASK",
    "PBS_NP",
    "LSB_DJOB_NUMPROC",
    "NSLOTS",
)
_THREAD_HINT_ENV_VARS = ("OMP_NUM_THREADS",)
_RUNTIME_ENV_VARS = (
    "MLX_SPARSE_CPU_THREADS",
    "MLX_SPARSE_N_THREADS",
    "MLX_SPARSE_SPGEMM_PARALLEL",
    "MLX_SPARSE_SPGEMM_THREADS",
    "MLX_SPARSE_SOLVER_PARALLEL",
    "MLX_SPARSE_SOLVER_THREADS",
    *_THREAD_HINT_ENV_VARS,
    *_SCHEDULER_THREAD_ENV_VARS,
)


def _normalize_option(option: RuntimeOption | str) -> str:
    if isinstance(option, RuntimeOption):
        return option.value
    if not isinstance(option, str):
        raise TypeError("runtime option must be a RuntimeOption or string.")
    key = option.strip().upper()
    try:
        return _OPTION_ALIASES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown runtime option {option!r}.") from exc


def _normalize_updates(updates: Mapping[Any, Any]) -> dict[str, Any]:
    return {_normalize_option(option): value for option, value in updates.items()}


def _normalize_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {_normalize_option(name): value for name, value in kwargs.items()}


def _parse_positive_env_int(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    token = raw_value.strip().split(",", 1)[0]
    if not token:
        return None
    try:
        value = int(token, 10)
    except ValueError:
        return None
    if value >= 1:
        return value
    return None


def _hardware_concurrency() -> int | None:
    value = os.cpu_count()
    if value is None or value < 1:
        return None
    return value


def _affinity_count() -> int | None:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        return None
    try:
        value = len(get_affinity(0))
    except OSError:
        return None
    if value < 1:
        return None
    return value


def _detected_scheduler_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for name in _SCHEDULER_THREAD_ENV_VARS:
        value = _parse_positive_env_int(name)
        if value is not None:
            out[name] = value
    return out


def _detected_thread_hints() -> dict[str, int]:
    out: dict[str, int] = {}
    for name in _THREAD_HINT_ENV_VARS:
        value = _parse_positive_env_int(name)
        if value is not None:
            out[name] = value
    return out


def _mlx_info() -> dict[str, Any]:
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - MLX is a hard dependency here.
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "default_device": str(mx.default_device()),
        "metal_available": bool(mx.metal.is_available()),
    }


def _resolve_auto_threads() -> tuple[int, str]:
    for name in _THREAD_HINT_ENV_VARS:
        value = _parse_positive_env_int(name)
        if value is not None:
            return value, name

    for name in _SCHEDULER_THREAD_ENV_VARS:
        value = _parse_positive_env_int(name)
        if value is not None:
            return value, name

    affinity = _affinity_count()
    if affinity is not None:
        return affinity, "process_affinity"

    hardware = _hardware_concurrency()
    if hardware is not None:
        return hardware, "hardware_concurrency"

    return 1, "fallback"


def resolve_n_threads() -> tuple[int, str]:
    """Resolve the effective CPU worker count and the source used.

    Explicit ``MLX_SPARSE_CPU_THREADS`` / ``ms.runtime.N_THREADS = ...`` values
    win first. In ``"auto"`` mode, standard thread hints are consulted before
    scheduler allocations, then process affinity, then hardware concurrency.
    """

    configured = config.get(RuntimeOption.N_THREADS.value)
    if isinstance(configured, int):
        return configured, "configured"

    return _resolve_auto_threads()


def _resolve_family_threads(option: RuntimeOption) -> tuple[int, str]:
    configured = config.get(option.value)
    if isinstance(configured, int):
        return configured, "configured"
    if configured == "inherit":
        return resolve_n_threads()
    return _resolve_auto_threads()


def resolve_spgemm_threads() -> tuple[int, str]:
    """Resolve the CPU worker count and source for sparse-sparse products."""

    if not bool(config.get(RuntimeOption.SPGEMM_PARALLEL.value)):
        return 1, "disabled"
    return _resolve_family_threads(SPGEMM_THREADS)


def resolve_solver_threads() -> tuple[int, str]:
    """Resolve the CPU worker count and source for solver routines."""

    if not bool(config.get(RuntimeOption.SOLVER_PARALLEL.value)):
        return 1, "disabled"
    return _resolve_family_threads(SOLVER_THREADS)


@contextlib.contextmanager
def context(
    arg1: RuntimeOption | str | Mapping[Any, Any] | None = None,
    arg2: Any = _MISSING,
    **kwargs: Any,
) -> Iterator[None]:
    """Temporarily patch runtime options.

    Accepted forms mirror :meth:`mlx_sparse.config.patch`:

    ``context(ms.runtime.RuntimeOption.N_THREADS, 4)``
        Patch one enum option.

    ``context({ms.runtime.RuntimeOption.N_THREADS: 4})``
        Patch several enum options.

    ``context(n_threads=4, spgemm_parallel=False)``
        Patch options with readable keyword aliases.
    """

    if arg1 is None:
        if arg2 is not _MISSING:
            raise TypeError("context(None, value) is not a valid call form.")
        updates = _normalize_kwargs(kwargs)
    elif isinstance(arg1, Mapping):
        if arg2 is not _MISSING:
            raise TypeError("Mapping context form does not accept a second value.")
        if kwargs:
            raise TypeError("Cannot combine mapping context with keywords.")
        updates = _normalize_updates(arg1)
    else:
        if arg2 is _MISSING:
            raise TypeError("context(option, value) requires a value.")
        if kwargs:
            raise TypeError("Cannot combine two-argument context with keywords.")
        updates = {_normalize_option(arg1): arg2}

    with config.patch(updates):
        yield


patch = context


def info() -> dict[str, Any]:
    """Return structured runtime information for reports and diagnostics."""

    resolved_threads, source = resolve_n_threads()
    spgemm_threads, spgemm_source = resolve_spgemm_threads()
    solver_threads, solver_source = resolve_solver_threads()
    spgemm_parallel = bool(config.get(RuntimeOption.SPGEMM_PARALLEL.value))
    solver_parallel = bool(config.get(RuntimeOption.SOLVER_PARALLEL.value))
    option_names = {option.name.lower(): option.value for option in RuntimeOption}
    return {
        "runtime_options": {
            key: config.get(name) for key, name in option_names.items()
        },
        "config_sources": {
            key: config.value_source(name).value for key, name in option_names.items()
        },
        "n_threads": resolved_threads,
        "n_threads_source": source,
        "spgemm_parallel": spgemm_parallel,
        "spgemm_n_threads": spgemm_threads,
        "spgemm_n_threads_source": spgemm_source,
        "solver_parallel": solver_parallel,
        "solver_n_threads": solver_threads,
        "solver_n_threads_source": solver_source,
        "native_extension_available": is_available(),
        "hardware_concurrency": _hardware_concurrency(),
        "process_affinity": _affinity_count(),
        "thread_hints": _detected_thread_hints(),
        "scheduler": _detected_scheduler_counts(),
        "environment": {
            name: os.environ[name] for name in _RUNTIME_ENV_VARS if name in os.environ
        },
        "mlx": _mlx_info(),
        "config_fingerprint": config.fingerprint(),
    }


_OPTION_ATTRIBUTE_OPTIONS = {
    "N_THREADS": RuntimeOption.N_THREADS,
    "SPGEMM_PARALLEL": RuntimeOption.SPGEMM_PARALLEL,
    "SPGEMM_THREADS": RuntimeOption.SPGEMM_THREADS,
    "SOLVER_PARALLEL": RuntimeOption.SOLVER_PARALLEL,
    "SOLVER_THREADS": RuntimeOption.SOLVER_THREADS,
}


def _read_option_attribute(name: str) -> Any:
    if name == "N_THREADS":
        return resolve_n_threads()[0]
    if name == "SPGEMM_THREADS":
        return resolve_spgemm_threads()[0]
    if name == "SOLVER_THREADS":
        return resolve_solver_threads()[0]
    return config.get(_OPTION_ATTRIBUTE_OPTIONS[name].value)


class _RuntimeModule(types.ModuleType):
    """Module subclass that turns public runtime knobs into descriptors."""

    def __getattribute__(self, name: str) -> Any:
        if name in _OPTION_ATTRIBUTE_OPTIONS:
            return _read_option_attribute(name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        option = _OPTION_ATTRIBUTE_OPTIONS.get(name)
        if option is not None:
            config.set(option.value, value)
            return
        super().__setattr__(name, value)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(_OPTION_ATTRIBUTE_OPTIONS))


sys.modules[__name__].__class__ = _RuntimeModule


__all__ = [
    "N_THREADS",
    "SOLVER_PARALLEL",
    "SOLVER_THREADS",
    "SPGEMM_PARALLEL",
    "SPGEMM_THREADS",
    "RuntimeOption",
    "context",
    "info",
    "patch",
    "resolve_n_threads",
    "resolve_solver_threads",
    "resolve_spgemm_threads",
]
