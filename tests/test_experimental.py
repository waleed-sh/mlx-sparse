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

import asyncio
import inspect
import os
import subprocess
import sys
import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse._config import ConfigValidationError
from mlx_sparse.experimental import (
    ExperimentalRegistry,
    ExperimentalWarning,
    experimental,
)

DOCS = "https://example.com/limitations"


@pytest.fixture(autouse=True)
def experimental_defaults():
    with ms.config_context(EXPERIMENTAL=False):
        yield


@pytest.fixture
def notices(monkeypatch):
    registry = ExperimentalRegistry()
    monkeypatch.setattr(ms.experimental, "registry", registry)
    return registry


def registered(registry, function, message="Check the result."):
    registry.register(function, message=message, docs_url=DOCS)
    return experimental(function)


def make_function(source, module="example", **namespace):
    namespace.update(__name__=module)
    exec(compile(source, __file__, "exec", dont_inherit=True), namespace)
    return namespace["solve"]


def test_public_exports_and_configuration():
    assert ms.ExperimentalWarning is ExperimentalWarning
    assert "experimental" in ms.__all__
    assert "ExperimentalWarning" in ms.__all__
    assert issubclass(ExperimentalWarning, RuntimeWarning)
    assert ms.config.EXPERIMENTAL is False
    assert os.environ["MLX_SPARSE_EXPERIMENTAL"] == "0"
    assert "EXPERIMENTAL" in ms.config.options_by_role()["warnings"]
    fingerprint = ms.config.fingerprint()
    with ms.config_context(EXPERIMENTAL=True):
        assert ms.config.EXPERIMENTAL is True
        assert os.environ["MLX_SPARSE_EXPERIMENTAL"] == "1"
        assert ms.config.fingerprint() == fingerprint
    assert ms.config.EXPERIMENTAL is False
    assert os.environ["MLX_SPARSE_EXPERIMENTAL"] == "0"
    with pytest.raises(ConfigValidationError):
        ms.set_config("EXPERIMENTAL", "unknown")


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("0", False)])
def test_environment_default_at_import(value, expected):
    env = dict(os.environ, MLX_SPARSE_EXPERIMENTAL=value)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx_sparse as ms; print(ms.config.EXPERIMENTAL)",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(expected)


def test_environment_silences_warnings_in_fresh_process():
    env = dict(os.environ, MLX_SPARSE_EXPERIMENTAL="1")
    source = (
        "import warnings, mlx_sparse as ms; "
        "warnings.simplefilter('error', ms.ExperimentalWarning); "
        "assert ms.config.EXPERIMENTAL is True; "
        "assert ms.linalg.cg(ms.eye(1), [1.0])[1] == 0"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_metadata_signature_and_cached_resolution(notices, monkeypatch):
    def solve(x: float, /, *, scale: float = 2.0) -> float:
        """Scale the input."""
        return x * scale

    wrapped = registered(notices, solve)
    assert wrapped.__name__ == solve.__name__
    assert wrapped.__qualname__ == solve.__qualname__
    assert wrapped.__module__ == solve.__module__
    assert wrapped.__annotations__ == solve.__annotations__
    assert wrapped.__wrapped__ is solve
    assert wrapped.__globals__ is solve.__globals__
    assert inspect.signature(wrapped) == inspect.signature(solve)
    assert ".. warning::" in wrapped.__doc__
    assert "Scale the input." in wrapped.__doc__
    assert "Check the result." in wrapped.__doc__
    assert wrapped.__experimental_feature__ is notices.resolve(solve)

    def unexpected(*args, **kwargs):
        raise AssertionError("Signatures must only be resolved at decoration time")

    monkeypatch.setattr(ms.experimental.inspect, "signature", unexpected)
    with pytest.warns(ExperimentalWarning):
        assert wrapped(3.0, scale=4.0) == 12.0
    with (
        ms.config_context(EXPERIMENTAL=True),
        warnings.catch_warnings(record=True) as seen,
    ):
        warnings.simplefilter("always")
        assert wrapped(3.0) == 6.0
        assert not seen


def test_warning_format_category_location_and_runtime_toggle(notices):
    def solve(value):
        return value

    wrapped = registered(
        notices, solve, "A result needs an independent residual check."
    )
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        assert wrapped(7) == 7
    assert len(seen) == 1
    assert seen[0].filename == __file__
    assert seen[0].lineno == line
    assert seen[0].category is ExperimentalWarning
    text = str(seen[0].message)
    assert text.startswith(
        f"{solve.__module__}.{solve.__qualname__} is experimental.\n  "
    )
    assert "residual check" in text
    assert f"Details: {DOCS}" in text
    assert "mlx_sparse.config.EXPERIMENTAL = True" in text
    assert "\x1b" not in text
    with ms.config_context(EXPERIMENTAL=True), warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        assert wrapped(8) == 8
    with pytest.warns(ExperimentalWarning):
        assert wrapped(9) == 9


def test_default_filter_deduplicates_by_external_call_site(notices):
    solve = registered(notices, lambda: None)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("default", ExperimentalWarning)
        for _ in range(3):
            solve()
        solve()
    assert len(seen) == 2
    assert seen[0].lineno != seen[1].lineno


def test_other_warnings_and_exceptions_are_preserved(notices):
    def solve():
        warnings.warn("input issue", UserWarning)
        raise ValueError("failed")

    wrapped = registered(notices, solve)
    with (
        ms.config_context(EXPERIMENTAL=True),
        pytest.warns(UserWarning, match="input issue"),
    ):
        with pytest.raises(ValueError, match="failed"):
            wrapped()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="failed"):
            wrapped()
        with pytest.raises(ValueError, match="failed"):
            wrapped()
    assert sum(w.category is ExperimentalWarning for w in seen) == 2


def test_explicit_error_filter_remains_under_caller_control(notices):
    calls = []
    solve = registered(notices, lambda: calls.append(1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        with pytest.raises(ExperimentalWarning):
            solve()
    assert calls == []
    with pytest.warns(ExperimentalWarning):
        solve()
    assert calls == [1]


def test_registry_overloads_and_family_fallback():
    registry = ExperimentalRegistry()
    first = make_function("def solve(x: int): return x")
    second = make_function("def solve(x: str): return x")
    third = make_function("def solve(x: float): return x")
    registry.register("example.solve", message="Family", docs_url=DOCS)
    registry.register(first, message="Integer", docs_url=DOCS)
    registry.register(second, message="String", docs_url=DOCS)
    assert registry.resolve(first).message == "Integer"
    assert registry.resolve(second).message == "String"
    assert registry.resolve(third).message == "Family"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(first, message="Duplicate", docs_url=DOCS)
    with pytest.raises(FrozenInstanceError):
        registry.features()[0].message = "Changed"
    snapshot = registry.features()
    registry.register(third, message="Float", docs_url=DOCS)
    assert len(snapshot) == 3
    assert len(registry.features()) == 4


def test_registry_distinguishes_modules_parameters_and_type_identity():
    registry = ExperimentalRegistry()
    functions = [
        make_function("def solve(x): pass", module="first"),
        make_function("def solve(x): pass", module="second"),
        make_function("def solve(x, /): pass", module="first"),
        make_function("def solve(*, x): pass", module="first"),
        make_function("def solve(y): pass", module="first"),
    ]
    for i, function in enumerate(functions):
        registry.register(function, message=str(i), docs_url=DOCS)
    for i, function in enumerate(functions):
        assert registry.resolve(function).message == str(i)
    types = [type("Input", (), {}), type("Input", (), {})]
    for i, type_ in enumerate(types):
        function = make_function("def solve(x: Input): pass", Input=type_)
        registry.register(function, message=f"type {i}", docs_url=DOCS)
        assert registry.resolve(function).message == f"type {i}"


def test_registry_ignores_defaults_and_return_annotations_without_evaluation():
    registry = ExperimentalRegistry()
    first = make_function("def solve(x: 'Unavailable' = []): pass")
    second = make_function("def solve(x: 'Unavailable' = {})->int: pass")
    registry.register(first, message="Notice", docs_url=DOCS)
    assert registry.resolve(second).message == "Notice"
    assert registry.resolve(first).signature.parameters["x"].annotation == "Unavailable"
    annotation = []
    first.__annotations__["x"] = annotation
    registry.register(first, message="Unhashable", docs_url=DOCS)
    assert registry.resolve(first).message == "Unhashable"


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"target": 123}, TypeError),
        ({"target": "unqualified"}, ValueError),
        ({"target": "example."}, ValueError),
        ({"signature": "(x)"}, TypeError),
        ({"message": " "}, ValueError),
        ({"message": None}, ValueError),
        ({"docs_url": "file:///tmp/page"}, ValueError),
        ({"display_name": ""}, ValueError),
    ],
)
def test_registry_rejects_invalid_entries(kwargs, error):
    args = dict(target="example.solve", message="Notice", docs_url=DOCS)
    args.update(kwargs)
    with pytest.raises(error):
        ExperimentalRegistry().register(**args)


def test_missing_notice_and_wrong_decorator_order_fail_at_definition(notices):
    with pytest.raises(KeyError, match="No experimental notice"):
        experimental(lambda: None)
    with pytest.raises(TypeError, match="Python implementation"):
        experimental(classmethod(lambda cls: None))
    with pytest.raises(TypeError, match="Python implementation"):
        experimental(len)


def test_nested_internal_calls_and_user_callbacks(notices):
    inner = make_function("def solve(x): return x", module="mlx_sparse.testing.inner")
    inner = registered(notices, inner, "Inner")
    outer = make_function(
        "def solve(x, callback):\n    inner(x)\n    return callback(x)",
        module="mlx_sparse.testing.outer",
        inner=inner,
    )
    outer = registered(notices, outer, "Outer")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        assert outer(3, lambda x: x) == 3
    assert len(seen) == 1
    assert seen[0].lineno == line
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        assert outer(3, lambda x: inner(x)) == 3
    assert len(seen) == 2
    assert "Outer" in str(seen[0].message)
    assert "Inner" in str(seen[1].message)


def test_methods_and_descriptor_order(notices):
    class Factor:
        def solve(self, x):
            return x

        @staticmethod
        def static(x):
            return x

        @classmethod
        def create(cls, x):
            return x

    Factor.solve = registered(notices, Factor.solve)
    Factor.static = staticmethod(registered(notices, Factor.static))
    Factor.create = classmethod(registered(notices, Factor.create.__func__))
    for call in (Factor().solve, Factor.static, Factor.create):
        with pytest.warns(ExperimentalWarning):
            assert call(3) == 3


def test_async_task_local_notification_and_exception_cleanup(notices):
    async def solve(value):
        await asyncio.sleep(0)
        if value < 0:
            raise ValueError("negative")
        return value

    wrapped = registered(notices, solve)
    assert inspect.iscoroutinefunction(wrapped)
    assert inspect.signature(wrapped) == inspect.signature(solve)

    async def run():
        return await asyncio.gather(
            wrapped(1), wrapped(2), wrapped(-1), return_exceptions=True
        )

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        results = asyncio.run(run())
    assert results[:2] == [1, 2]
    assert isinstance(results[2], ValueError)
    assert len(seen) == 3
    with ms.config_context(EXPERIMENTAL=True), warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        assert asyncio.run(wrapped(4)) == 4


SOLVERS = [
    "cg",
    "eigsh",
    "eigs",
    "svds",
    "spsolve_triangular",
    "sparse_lu",
    "splu",
    "sparse_cholesky",
    "cholesky",
    "factorized",
    "spsolve",
    "SparseLU.solve",
    "SparseCholesky.solve",
    "FactorizedSolve.solve",
]


def public_solver(name):
    obj = ms.linalg
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("name", SOLVERS)
def test_audited_entry_points_have_registry_notices_and_docstrings(name):
    function = public_solver(name)
    feature = ms.experimental.registry.resolve(function)
    assert feature.display_name == f"mlx_sparse.linalg.{name}"
    assert function.__experimental_feature__ is feature
    assert "solver_limitations.rst" in feature.docs_url
    assert ".. warning::" in function.__doc__
    assert ";" not in feature.message
    assert "—" not in feature.message


@pytest.mark.parametrize("name", ["gmres", "minres", "bicgstab", "lanczos"])
def test_other_solvers_are_not_marked(name):
    assert not hasattr(public_solver(name), "__experimental_feature__")


@pytest.mark.parametrize("name", SOLVERS)
def test_solver_warnings_and_suppression_preserve_numerical_results(mx, to_numpy, name):
    A = ms.from_dense(mx.array([[4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]]))
    b = mx.array([4.0, 6.0, 8.0])
    function = public_solver(name)
    if name == "cg":
        args, kwargs = (A, b), {}
    elif name in {"eigsh", "eigs", "svds"}:
        args, kwargs = (A,), {"k": 1, "ncv": 3}
    elif name in {"spsolve", "spsolve_triangular"}:
        args, kwargs = (A, b), {}
    elif ".solve" in name:
        factories = {
            "SparseLU.solve": ms.linalg.sparse_lu,
            "SparseCholesky.solve": ms.linalg.sparse_cholesky,
            "FactorizedSolve.solve": ms.linalg.factorized,
        }
        with ms.config_context(EXPERIMENTAL=True):
            factor = factories[name](A)
        function, args, kwargs = factor.solve, (b,), {}
    else:
        args, kwargs = (A,), {}

    def arrays(value):
        if hasattr(value, "solve"):
            return arrays(value.solve(b))
        if isinstance(value, tuple):
            return tuple(arrays(item) for item in value)
        if isinstance(value, mx.array):
            return to_numpy(value)
        return value

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ExperimentalWarning)
        line = inspect.currentframe().f_lineno + 1
        result = function(*args, **kwargs)
    assert len(seen) == 1
    assert seen[0].filename == __file__
    assert seen[0].lineno == line
    assert f"mlx_sparse.linalg.{name} is experimental" in str(seen[0].message)
    with ms.config_context(EXPERIMENTAL=True), warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        before = arrays(result)
        after = arrays(function(*args, **kwargs))

    def equal(first, second):
        if isinstance(first, tuple):
            for left, right in zip(first, second):
                equal(left, right)
        else:
            np.testing.assert_array_equal(first, second)

    equal(before, after)


@pytest.mark.parametrize("factory", ["sparse_lu", "sparse_cholesky", "factorized"])
def test_factor_callable_alias_warns_at_external_caller(mx, factory):
    with ms.config_context(EXPERIMENTAL=True):
        factor = getattr(ms.linalg, factory)(ms.eye(3))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ExperimentalWarning)
        line = inspect.currentframe().f_lineno + 1
        result = factor(mx.ones((3,)))
        mx.eval(result)
    assert len(seen) == 1
    assert seen[0].lineno == line
    assert seen[0].filename == __file__


def test_plum_dispatch_selects_notice_for_each_implementation(notices):
    plum = pytest.importorskip("plum")
    dispatch = plum.Dispatcher()
    integer = make_function("def solve(x: int): return x + 1")
    string = make_function("def solve(x: str): return x.upper()")
    stable = make_function("def solve(x: float): return x * 2")
    dispatch(registered(notices, integer, "Integer overload"))
    dispatch(registered(notices, string, "String overload"))
    solve = dispatch(stable)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        assert solve(3) == 4
        assert solve("a") == "A"
        assert solve(1.5) == 3.0
    assert len(seen) == 2
    assert seen[0].filename == __file__
    assert seen[0].lineno == line
    assert "Integer overload" in str(seen[0].message)
    assert "String overload" in str(seen[1].message)
    with ms.config_context(EXPERIMENTAL=True), warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        assert solve(3) == 4
        assert solve("a") == "A"


def test_plum_methods_and_postponed_annotations_use_original_globals(notices):
    plum = pytest.importorskip("plum")
    namespace = dict(
        __name__="experimental_dispatch_test",
        dispatch=plum.Dispatcher(),
        experimental=experimental,
    )
    notices.register(
        "experimental_dispatch_test.Factor.solve",
        message="Factor method",
        docs_url=DOCS,
    )
    source = """from __future__ import annotations
class Input:
    pass

class Base:
    @dispatch
    def solve(self, x: float):
        return x * 2

class Factor(Base):
    @dispatch
    @experimental
    def solve(self, x: Input):
        return x

    @dispatch
    def solve(self, x: int):
        return x + 1
"""
    exec(compile(source, __file__, "exec", dont_inherit=True), namespace)
    value = namespace["Input"]()
    factor = namespace["Factor"]()
    assert namespace["Factor"].solve.owner is namespace["Factor"]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        assert factor.solve(value) is value
        assert factor.solve(2) == 3
        assert factor.solve(1.5) == 3.0
    assert len(seen) == 1
    assert seen[0].filename == __file__
    assert seen[0].lineno == line
