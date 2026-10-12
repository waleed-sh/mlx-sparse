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

"""Registry and runtime warnings for experimental public APIs."""

from __future__ import annotations

import inspect
import sys
import textwrap
import threading
import warnings
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from types import FunctionType
from typing import ParamSpec, TypeVar

from mlx_sparse._config import config
from mlx_sparse._experimental_features import register_features as _register_features

__all__ = [
    "ExperimentalFeature",
    "ExperimentalRegistry",
    "ExperimentalWarning",
    "experimental",
    "registry",
]

_P = ParamSpec("_P")
_R = TypeVar("_R")
_active_call: ContextVar[bool] = ContextVar(
    "mlx_sparse_experimental_call", default=False
)
_wrapper_codes: set = set()


class ExperimentalWarning(RuntimeWarning):
    """An experimental API has documented limitations that need review.

    Visible under Python's default warning filter. Set
    ``mlx_sparse.config.EXPERIMENTAL = True`` to acknowledge the limitations
    and silence this category at its source. Other warning categories are
    unaffected. An explicit Python warning filter can turn warnings into errors.
    """


@dataclass(frozen=True, slots=True)
class ExperimentalFeature:
    """Immutable registry entry for a function or a particular signature.

    Args:
        target: Fully qualified implementation name, including its module.
        signature: Registered signature, or ``None`` for all overloads.
        message: Explanation of the limitation and any result checks needed.
        docs_url: Link to the documentation describing the limitation.
        display_name: Public name shown in warnings and generated docstrings.
    """

    target: str
    signature: inspect.Signature | None
    message: str
    docs_url: str
    display_name: str


def _name(function: Callable) -> str:
    return f"{function.__module__}.{function.__qualname__}"


def _signature_key(signature: inspect.Signature) -> tuple:
    # Defaults and return annotations do not select a dispatched implementation.
    parameters = []
    for parameter in signature.parameters.values():
        annotation = parameter.annotation
        try:
            hash(annotation)
        except TypeError:
            annotation = ("unhashable annotation", id(annotation))
        parameters.append((parameter.name, parameter.kind, annotation))
    return tuple(parameters)


class ExperimentalRegistry:
    """Store experimental notices by qualified name and implementation signature.

    Register entries before applying :func:`experimental`. An entry with a
    signature takes precedence over an entry covering all signatures. Wrappers
    resolve their notice once at decoration time, so calls do not inspect
    signatures or acquire a registry lock. Registration does not alter notices
    already attached to a wrapper.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple, ExperimentalFeature] = {}
        self._lock = threading.RLock()

    def register(
        self,
        target: str | Callable,
        *,
        message: str,
        docs_url: str,
        signature: inspect.Signature | None = None,
        display_name: str | None = None,
    ) -> None:
        """Register a notice for an implementation or a function family.

        Args:
            target: Qualified name or implementation function. A function
                supplies its name and signature automatically. A string without
                ``signature`` registers a notice for every overload of that name.
            message: Limitation and suggested checks, written as plain text.
            docs_url: HTTP or HTTPS documentation link.
            signature: Signature selecting a particular implementation. Parameter
                names, kinds, and annotations identify the signature. Defaults
                and return annotations are ignored. Postponed annotations are
                retained without evaluation. Use the same implementation's
                signature when registering an overload.
            display_name: Public API name to display, defaulting to ``target``.

        Raises:
            TypeError: An argument has an unsupported type.
            ValueError: An entry is incomplete or its key is already registered.
        """
        if callable(target):
            if signature is None:
                signature = inspect.signature(target)
            target = _name(target)
        if not isinstance(target, str):
            raise TypeError("target must be a qualified name or a function.")
        if "." not in target or not target.strip() or target.endswith("."):
            raise ValueError("target must include a module and function name.")
        if signature is not None and not isinstance(signature, inspect.Signature):
            raise TypeError("signature must be an inspect.Signature or None.")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be nonempty text.")
        if not isinstance(docs_url, str) or not docs_url.startswith(
            ("https://", "http://")
        ):
            raise ValueError("docs_url must be an HTTP or HTTPS link.")
        if display_name is None:
            display_name = target
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError("display_name must be nonempty text.")
        key = (target, None if signature is None else _signature_key(signature))
        feature = ExperimentalFeature(
            target, signature, message.strip(), docs_url, display_name
        )
        with self._lock:
            if key in self._entries:
                raise ValueError(
                    f"Experimental notice already registered for {target}."
                )
            self._entries[key] = feature

    def resolve(self, function: Callable) -> ExperimentalFeature:
        """Return the notice matching a function's name and signature.

        A signature-specific entry is preferred over a function-wide entry.
        Raises ``KeyError`` when neither exists. Resolution never evaluates
        annotations or invokes the function.
        """
        target = _name(function)
        key = (target, _signature_key(inspect.signature(function)))
        with self._lock:
            feature = self._entries.get(key) or self._entries.get((target, None))
        if feature is None:
            raise KeyError(f"No experimental notice registered for {target}.")
        return feature

    def features(self) -> tuple[ExperimentalFeature, ...]:
        """Return an immutable snapshot of entries in registration order."""
        with self._lock:
            return tuple(self._entries.values())


registry = ExperimentalRegistry()


def _package_frame(module: str) -> bool:
    return module == "mlx_sparse" or module.startswith("mlx_sparse.")


def _dispatch_frame(module: str) -> bool:
    return module == "plum" or module.startswith("plum.")


def _nested_internal_call() -> bool:
    if not _active_call.get():
        return False
    frame = sys._getframe(2)
    try:
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if not _dispatch_frame(module):
                return _package_frame(module)
            frame = frame.f_back
        return False
    finally:
        del frame


def _warn(message: str) -> None:
    # Python 3.10 has no skip_file_prefixes argument. Walk wrappers and dispatch
    # frames so default warning deduplication uses the external call site.
    frame = sys._getframe(1)
    stacklevel = 2
    try:
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if not (
                frame.f_code in _wrapper_codes
                or _package_frame(module)
                or _dispatch_frame(module)
            ):
                break
            frame = frame.f_back
            stacklevel += 1
    finally:
        del frame
    warnings.warn(message, ExperimentalWarning, stacklevel=stacklevel)


def experimental(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Warn when calling an experimental implementation registered in ``registry``.

    Use as ``@experimental`` on a Python function. Its qualified name and
    signature select the registry entry. Missing entries fail at decoration
    time. Metadata, annotations, the original globals, and the inspectable
    signature are preserved.
    The generated docstring includes the registry notice as a warning.

    With Plum, place ``@dispatch`` above ``@experimental`` and decorate each
    experimental implementation separately. Apply this decorator below
    ``@staticmethod`` or ``@classmethod`` when decorating a method.

    Warnings follow Python's filters and point to the external call site.
    Internal calls made by an experimental solver share its outer notice.
    Independent calls made from user callbacks still emit their own notice.
    ``config.EXPERIMENTAL=True`` skips notification without changing execution.
    Normal warning filters do not interrupt the call. An explicit ``error``
    filter can do so.

    Args:
        function: Registered implementation to wrap.

    Returns:
        A wrapper with the same calling convention and return behavior.

    Raises:
        TypeError: The target is not a Python function.
        KeyError: No notice matches the function in the registry.
    """
    if not inspect.isfunction(function):
        raise TypeError("Apply @experimental to each Python implementation function.")
    feature = registry.resolve(function)
    message = (
        f"{feature.display_name} is experimental.\n"
        + textwrap.fill(
            feature.message, width=80, initial_indent="  ", subsequent_indent="  "
        )
        + f"\n  Details: {feature.docs_url}\n"
        + "  To acknowledge these limitations and silence experimental warnings:\n"
        + "    mlx_sparse.config.EXPERIMENTAL = True"
    )

    # Capture wrapper dependencies in closure cells. The wrapper can then use
    # the implementation's globals, as Plum requires for method ownership and
    # annotation resolution, without adding names to the implementation module.
    settings = config
    nested_call = _nested_internal_call
    warn = _warn
    active_call = _active_call

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if settings.get("EXPERIMENTAL") or nested_call():
            return function(*args, **kwargs)
        warn(message)
        token = active_call.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            active_call.reset(token)

    async def async_wrapper(*args: _P.args, **kwargs: _P.kwargs):
        if settings.get("EXPERIMENTAL") or nested_call():
            return await function(*args, **kwargs)
        warn(message)
        token = active_call.set(True)
        try:
            return await function(*args, **kwargs)
        finally:
            active_call.reset(token)

    template = async_wrapper if inspect.iscoroutinefunction(function) else wrapper
    _wrapper_codes.add(template.__code__)
    result = wraps(function)(
        FunctionType(
            template.__code__, function.__globals__, closure=template.__closure__
        )
    )
    doc = inspect.getdoc(function) or ""
    result.__doc__ = (
        doc
        + "\n\n.. warning::\n\n"
        + f"   This API is experimental. {feature.message}\n\n"
        + f"   See the `documented limitations <{feature.docs_url}>`_.\n"
        + "   Set ``mlx_sparse.config.EXPERIMENTAL = True`` to acknowledge them\n"
        + "   and silence experimental warnings. This does not change the result.\n"
    )
    result.__experimental_feature__ = feature
    return result


_register_features(registry)
