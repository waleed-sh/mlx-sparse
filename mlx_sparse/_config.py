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

"""Typed runtime configuration for mlx-sparse.

The design follows the same broad shape as neuraLQX's configuration manager:
options are declared in one schema, read from environment variables, validated
on mutation, observable through hooks, and temporarily patchable from user code.
The implementation here is deliberately smaller because mlx-sparse only needs a
few package-level knobs today.
"""

from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")


class ConfigError(Exception):
    """Base class for configuration errors."""


class UnknownOptionError(ConfigError):
    """Raised when a configuration option name is not registered."""


class ConfigValidationError(ConfigError):
    """Raised when a configuration value cannot be parsed or validated."""


class ConfigMutability(str, enum.Enum):
    """When an option may be changed through the Python API."""

    IMMUTABLE = "immutable"
    STARTUP = "startup"
    RUNTIME = "runtime"


class ConfigSource(str, enum.Enum):
    """Where an effective configuration value came from."""

    DEFAULT = "default"
    ENV_DEFAULT = "env_default"
    ENV_FORCE = "env_force"
    USER = "user"
    PATCH = "patch"
    RESET = "reset"


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<UNSET>"


UNSET = _UnsetType()
_MISSING = _UnsetType()

Parser = Callable[[Any], Any]
Validator = Callable[[Any], None]
MutationHook = Callable[["ConfigMutation"], None]


@dataclass(frozen=True, slots=True)
class ConfigMutation:
    """A single effective configuration change."""

    name: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    mutability: ConfigMutability


@dataclass(frozen=True, slots=True)
class ConfigOption(Generic[T]):
    """Static declaration for one configuration option."""

    name: str
    default: T
    doc: str
    value_type: type[Any] | tuple[type[Any], ...] | None = None
    parser: Parser | None = None
    validator: Validator | None = None
    env_default: tuple[str, ...] = ()
    env_force: tuple[str, ...] = ()
    role: str = "general"
    mutability: ConfigMutability = ConfigMutability.RUNTIME
    include_in_fingerprint: bool = True

    def parse(self, raw_value: Any) -> T:
        parsed = self.parser(raw_value) if self.parser is not None else raw_value
        if self.value_type is not None and not isinstance(parsed, self.value_type):
            raise ConfigValidationError(
                f"Option {self.name!r} expects {self.value_type}, "
                f"got {type(parsed).__name__} with value {parsed!r}."
            )
        if self.validator is not None:
            self.validator(parsed)
        return cast(T, parsed)


@dataclass(slots=True)
class _OptionState(Generic[T]):
    spec: ConfigOption[T]
    env_default_value: T | _UnsetType = UNSET
    env_force_value: T | _UnsetType = UNSET
    user_override: T | _UnsetType = UNSET


def parse_bool(value: Any) -> bool:
    """Parse a permissive boolean value."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ConfigValidationError(f"Cannot parse boolean from value {value!r}.")


def parse_thread_count(value: Any) -> int | str:
    """Parse a positive thread count or the ``"auto"`` sentinel."""

    if isinstance(value, bool):
        raise ConfigValidationError(
            f"Thread count must be a positive integer or 'auto', got {value!r}."
        )
    if isinstance(value, int):
        if value >= 1:
            return value
        raise ConfigValidationError(
            f"Thread count must be a positive integer or 'auto', got {value!r}."
        )
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        try:
            parsed = int(normalized, 10)
        except ValueError as exc:
            raise ConfigValidationError(
                f"Thread count must be a positive integer or 'auto', got {value!r}."
            ) from exc
        if parsed >= 1:
            return parsed
    raise ConfigValidationError(
        f"Thread count must be a positive integer or 'auto', got {value!r}."
    )


def parse_thread_count_or_inherit(value: Any) -> int | str:
    """Parse a positive thread count, ``"auto"``, or ``"inherit"``."""

    if isinstance(value, str) and value.strip().lower() == "inherit":
        return "inherit"
    return parse_thread_count(value)


def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


class ConfigManager:
    """Typed, hookable singleton configuration manager for mlx-sparse.

    Effective value precedence is:

    1. forced environment variable (``env_force``)
    2. programmatic override
    3. default environment variable (``env_default``)
    4. built-in default
    """

    _instance: "ConfigManager | None" = None
    _class_lock = threading.Lock()

    PREFIX = "MLX_SPARSE_"

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        object.__setattr__(self, "_options", {})
        object.__setattr__(self, "_hooks", {})
        object.__setattr__(self, "_global_hooks", [])
        object.__setattr__(self, "_runtime_locked", False)
        object.__setattr__(self, "_lock", threading.RLock())
        self._register_default_options()
        self._initialized = True

    def __repr__(self) -> str:
        return (
            "ConfigManager("
            f"options={len(self._options)}, "
            f"runtime_locked={self._runtime_locked})"
        )

    def __getattr__(self, name: str) -> Any:
        if "_options" in self.__dict__ and name in self._options:
            return self.get(name)
        raise AttributeError(f"{self.__class__.__name__} has no option {name!r}.")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if "_options" in self.__dict__ and name in self._options:
            self.set(name, value)
            return
        raise AttributeError(f"{self.__class__.__name__} has no option {name!r}.")

    @staticmethod
    def _normalize_env_names(names: str | Sequence[str] | None) -> tuple[str, ...]:
        if names is None:
            return ()
        if isinstance(names, str):
            names = (names,)
        out: list[str] = []
        for name in names:
            if not isinstance(name, str) or not name:
                raise ConfigValidationError(
                    "Environment names must be non-empty strings."
                )
            out.append(name.upper())
        return tuple(out)

    def _lookup_env(self, key: str) -> str | None:
        key_upper = key.upper()
        for env_key, env_value in os.environ.items():
            if env_key.upper() == key_upper:
                return env_value
        return None

    def _read_env_value(
        self, option: ConfigOption[Any], env_names: Sequence[str]
    ) -> Any | _UnsetType:
        for env_name in env_names:
            raw_value = self._lookup_env(env_name)
            if raw_value is None:
                continue
            try:
                return option.parse(raw_value)
            except ConfigValidationError as exc:
                raise ConfigValidationError(
                    f"Invalid value from environment variable {env_name!r} "
                    f"for option {option.name!r}: {raw_value!r}."
                ) from exc
        return UNSET

    def _direct_effective_value(self, state: _OptionState[Any]) -> Any:
        if state.env_force_value is not UNSET:
            return state.env_force_value
        if state.user_override is not UNSET:
            return state.user_override
        if state.env_default_value is not UNSET:
            return state.env_default_value
        return state.spec.default

    def _sync_env(self, state: _OptionState[Any]) -> None:
        os.environ[self.PREFIX + state.spec.name] = _format_env_value(
            self._direct_effective_value(state)
        )

    def register_option(self, option: ConfigOption[Any]) -> None:
        with self._lock:
            if not option.name or not isinstance(option.name, str):
                raise ConfigValidationError("Option name must be a non-empty string.")
            name = option.name.upper()
            if name in self._options:
                raise ConfigValidationError(f"Option {name!r} is already registered.")

            default = option.parse(option.default)
            option = replace(
                option,
                name=name,
                default=default,
                env_default=self._normalize_env_names(option.env_default),
                env_force=self._normalize_env_names(option.env_force),
            )
            state = _OptionState(spec=option)
            state.env_default_value = self._read_env_value(option, option.env_default)
            state.env_force_value = self._read_env_value(option, option.env_force)
            self._options[name] = state
            self._hooks[name] = []
            self._sync_env(state)

    def define_option(
        self,
        name: str,
        *,
        default: Any,
        doc: str,
        value_type: type[Any] | tuple[type[Any], ...] | None = None,
        parser: Parser | None = None,
        validator: Validator | None = None,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.RUNTIME,
        include_in_fingerprint: bool = True,
    ) -> None:
        self.register_option(
            ConfigOption(
                name=name,
                default=default,
                doc=doc,
                value_type=value_type,
                parser=parser,
                validator=validator,
                env_default=self._normalize_env_names(env_default),
                env_force=self._normalize_env_names(env_force),
                role=role,
                mutability=mutability,
                include_in_fingerprint=include_in_fingerprint,
            )
        )

    def define_bool(
        self,
        name: str,
        *,
        default: bool,
        doc: str,
        env_default: str | Sequence[str] | None = None,
        env_force: str | Sequence[str] | None = None,
        role: str = "general",
        mutability: ConfigMutability = ConfigMutability.RUNTIME,
        include_in_fingerprint: bool = True,
    ) -> None:
        self.define_option(
            name,
            default=default,
            doc=doc,
            value_type=bool,
            parser=parse_bool,
            env_default=env_default,
            env_force=env_force,
            role=role,
            mutability=mutability,
            include_in_fingerprint=include_in_fingerprint,
        )

    def _get_state(self, name: str) -> _OptionState[Any]:
        key = name.upper()
        try:
            return self._options[key]
        except KeyError as exc:
            raise UnknownOptionError(f"Unknown option {name!r}.") from exc

    def _assert_can_mutate(self, state: _OptionState[Any]) -> None:
        if state.env_force_value is not UNSET:
            raise ConfigError(
                f"Option {state.spec.name!r} is forced by environment "
                f"{state.spec.env_force} and cannot be changed from Python."
            )
        if state.spec.mutability is ConfigMutability.IMMUTABLE:
            raise ConfigError(f"Option {state.spec.name!r} is immutable.")
        if state.spec.mutability is ConfigMutability.STARTUP and self._runtime_locked:
            raise ConfigError(
                f"Option {state.spec.name!r} is startup-only and runtime is locked."
            )

    def get(self, name: str) -> Any:
        return self._direct_effective_value(self._get_state(name))

    def read(self, name: str) -> Any:
        return self.get(name)

    def set(
        self,
        name: str,
        value: Any,
        *,
        source: ConfigSource = ConfigSource.USER,
    ) -> Any:
        state = self._get_state(name)
        parsed = state.spec.parse(value)
        with self._lock:
            self._assert_can_mutate(state)
            old_value = self.get(state.spec.name)
            state.user_override = parsed
            self._sync_env(state)
            new_value = self.get(state.spec.name)
            self._maybe_emit(state, old_value, new_value, source)
            return new_value

    def update(
        self,
        name: str,
        value: Any,
        *,
        source: ConfigSource = ConfigSource.USER,
    ) -> Any:
        return self.set(name, value, source=source)

    def clear_override(
        self,
        name: str,
        *,
        source: ConfigSource = ConfigSource.RESET,
    ) -> Any:
        state = self._get_state(name)
        with self._lock:
            self._assert_can_mutate(state)
            old_value = self.get(state.spec.name)
            state.user_override = UNSET
            self._sync_env(state)
            new_value = self.get(state.spec.name)
            self._maybe_emit(state, old_value, new_value, source)
            return new_value

    def set_many(
        self,
        updates: Mapping[str, Any],
        *,
        source: ConfigSource = ConfigSource.USER,
    ) -> None:
        for name, value in updates.items():
            self.set(name, value, source=source)

    @contextlib.contextmanager
    def patch(
        self,
        arg1: str | Mapping[str, Any] | None = None,
        arg2: Any = _MISSING,
        **kwargs: Any,
    ) -> Iterator[None]:
        updates = self._normalize_patch_args(arg1, arg2, kwargs)
        previous: dict[str, Any] = {}
        for name in updates:
            state = self._get_state(name)
            previous[state.spec.name] = state.user_override

        try:
            self.set_many(updates, source=ConfigSource.PATCH)
            yield
        finally:
            for name, old_value in previous.items():
                if old_value is UNSET:
                    self.clear_override(name, source=ConfigSource.PATCH)
                else:
                    self.set(name, old_value, source=ConfigSource.PATCH)

    def list_options(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._options))

    def snapshot(self) -> dict[str, Any]:
        return {name: self.get(name) for name in self.list_options()}

    @property
    def values(self) -> dict[str, Any]:
        return self.snapshot()

    def user_overrides(self) -> dict[str, Any]:
        return {
            name: state.user_override
            for name, state in self._options.items()
            if state.user_override is not UNSET
        }

    def describe_option(self, name: str) -> dict[str, Any]:
        state = self._get_state(name)
        return {
            "name": state.spec.name,
            "doc": state.spec.doc,
            "role": state.spec.role,
            "default": state.spec.default,
            "mutability": state.spec.mutability.value,
            "env_default": state.spec.env_default,
            "env_force": state.spec.env_force,
            "effective_value": self.get(state.spec.name),
            "source": self.value_source(state.spec.name).value,
            "runtime_locked": self._runtime_locked,
            "include_in_fingerprint": state.spec.include_in_fingerprint,
        }

    def options_by_role(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for name, state in self._options.items():
            grouped.setdefault(state.spec.role, []).append(name)
        return {role: tuple(sorted(names)) for role, names in grouped.items()}

    def value_source(self, name: str) -> ConfigSource:
        state = self._get_state(name)
        if state.env_force_value is not UNSET:
            return ConfigSource.ENV_FORCE
        if state.user_override is not UNSET:
            return ConfigSource.USER
        if state.env_default_value is not UNSET:
            return ConfigSource.ENV_DEFAULT
        return ConfigSource.DEFAULT

    def fingerprint(self) -> str:
        values = {
            name: self.get(name)
            for name, state in self._options.items()
            if state.spec.include_in_fingerprint
        }
        payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def runtime_locked(self) -> bool:
        return self._runtime_locked

    def lock_runtime(self) -> None:
        with self._lock:
            self._runtime_locked = True

    def unlock_runtime_for_testing(self) -> None:
        with self._lock:
            self._runtime_locked = False

    def add_hook(
        self,
        name: str,
        hook: MutationHook,
        *,
        run_immediately: bool = False,
    ) -> None:
        state = self._get_state(name)
        with self._lock:
            self._hooks[state.spec.name].append(hook)
        if run_immediately:
            current = self.get(state.spec.name)
            hook(
                ConfigMutation(
                    name=state.spec.name,
                    old_value=current,
                    new_value=current,
                    source=self.value_source(state.spec.name),
                    mutability=state.spec.mutability,
                )
            )

    def add_global_hook(self, hook: MutationHook) -> None:
        with self._lock:
            self._global_hooks.append(hook)

    def show(self) -> str:
        rows = [
            (
                name,
                state.spec.role,
                repr(state.spec.default),
                state.spec.mutability.value,
                repr(self.get(name)),
                self.value_source(name).value,
            )
            for name, state in sorted(self._options.items())
        ]
        headers = ("name", "role", "default", "mutability", "effective", "source")
        widths = [
            max(len(str(cell)) for cell in (header, *(row[i] for row in rows)))
            for i, header in enumerate(headers)
        ]
        lines = [
            "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
            "  ".join("-" * width for width in widths),
        ]
        lines.extend(
            "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            for row in rows
        )
        return "\n".join(lines)

    def _maybe_emit(
        self,
        state: _OptionState[Any],
        old_value: Any,
        new_value: Any,
        source: ConfigSource,
    ) -> None:
        if old_value == new_value:
            return
        event = ConfigMutation(
            name=state.spec.name,
            old_value=old_value,
            new_value=new_value,
            source=source,
            mutability=state.spec.mutability,
        )
        for hook in tuple(self._hooks[state.spec.name]):
            hook(event)
        for hook in tuple(self._global_hooks):
            hook(event)

    @staticmethod
    def _normalize_patch_args(
        arg1: str | Mapping[str, Any] | None,
        arg2: Any,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if arg1 is None:
            if arg2 is not _MISSING:
                raise TypeError("patch(None, value) is not a valid call form.")
            return dict(kwargs)
        if isinstance(arg1, str):
            if arg2 is _MISSING:
                raise TypeError("patch('NAME', value) requires a value.")
            if kwargs:
                raise TypeError("Cannot combine two-argument patch with keywords.")
            return {arg1: arg2}
        if arg2 is not _MISSING:
            raise TypeError("Mapping patch form does not accept a second value.")
        if kwargs:
            raise TypeError("Cannot combine mapping patch with keywords.")
        if not isinstance(arg1, Mapping):
            raise TypeError("patch expects a name or mapping.")
        return dict(arg1)

    def _register_default_options(self) -> None:
        self.define_option(
            "CPU_THREADS",
            default="auto",
            doc=(
                "Package-wide CPU worker setting. Use a positive integer for an "
                "explicit worker count, or 'auto' to resolve from standard "
                "threading and scheduler environment variables, process affinity, "
                "and hardware concurrency."
            ),
            value_type=(int, str),
            parser=parse_thread_count,
            env_default=("MLX_SPARSE_CPU_THREADS", "MLX_SPARSE_N_THREADS"),
            role="runtime",
            mutability=ConfigMutability.RUNTIME,
        )
        self.define_bool(
            "SPGEMM_PARALLEL",
            default=True,
            doc=(
                "Enable package-level CPU parallel execution for sparse-sparse "
                "matrix products when a parallel implementation is available."
            ),
            env_default="MLX_SPARSE_SPGEMM_PARALLEL",
            role="runtime",
            mutability=ConfigMutability.RUNTIME,
        )
        self.define_option(
            "SPGEMM_THREADS",
            default="inherit",
            doc=(
                "CPU worker setting for sparse-sparse matrix products. Use a "
                "positive integer for an explicit family-specific count, 'auto' "
                "for dynamic runtime resolution, or 'inherit' to use CPU_THREADS."
            ),
            value_type=(int, str),
            parser=parse_thread_count_or_inherit,
            env_default="MLX_SPARSE_SPGEMM_THREADS",
            role="runtime",
            mutability=ConfigMutability.RUNTIME,
        )
        self.define_bool(
            "SOLVER_PARALLEL",
            default=False,
            doc=(
                "Enable package-level CPU parallel execution for solver routines "
                "when a parallel implementation is available."
            ),
            env_default="MLX_SPARSE_SOLVER_PARALLEL",
            role="runtime",
            mutability=ConfigMutability.RUNTIME,
        )
        self.define_option(
            "SOLVER_THREADS",
            default="inherit",
            doc=(
                "CPU worker setting for solver routines. Use a positive integer "
                "for an explicit family-specific count, 'auto' for dynamic "
                "runtime resolution, or 'inherit' to use CPU_THREADS."
            ),
            value_type=(int, str),
            parser=parse_thread_count_or_inherit,
            env_default="MLX_SPARSE_SOLVER_THREADS",
            role="runtime",
            mutability=ConfigMutability.RUNTIME,
        )
        self.define_bool(
            "EXPERIMENTAL_METAL_SPGEMM",
            default=False,
            doc=(
                "Enable experimental staged Metal implementations for same-format "
                "CSR, COO, and CSC sparse-sparse products. "
                "The optimized native host SpGEMM path remains the default because "
                "it is faster on current small and medium benchmark cases."
            ),
            env_default="MLX_SPARSE_EXPERIMENTAL_METAL_SPGEMM",
            env_force="MLX_SPARSE_FORCE_EXPERIMENTAL_METAL_SPGEMM",
            role="sparse",
            mutability=ConfigMutability.RUNTIME,
        )


config = ConfigManager()


def get_config(name: str) -> Any:
    """Read a package configuration value."""

    return config.get(name)


def set_config(name: str, value: Any) -> Any:
    """Set a package configuration value."""

    return config.set(name, value)


def config_context(
    *args: Any, **kwargs: Any
) -> contextlib.AbstractContextManager[None]:
    """Temporarily patch package configuration values."""

    return config.patch(*args, **kwargs)


__all__ = [
    "config",
    "config_context",
    "get_config",
    "parse_bool",
    "parse_thread_count",
    "parse_thread_count_or_inherit",
    "set_config",
]
