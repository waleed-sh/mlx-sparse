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

"""The API reference must not omit an export, or one member of a family.

Two checks, and they fail on different things.

The named-family cells below require a *directive-grade* entry — an ``auto*``
block that renders a signature and a docstring — for the constructors, the
containers, the solvers and the spectral routines. They are lists of names
written by hand, and that is what they are for: those families are the core of
the package and each member is meant to have a page of its own.

A hand-written list cannot report a name nobody thought of, though, and that is
the shape this file was missing. ``test_every_export_is_reachable_from_the_api_reference``
derives the names instead, from ``__all__`` of each public namespace, and
subtracts what the reference mentions. It is the weaker requirement per name —
a mention anywhere in ``docs/api`` counts, including a ``:func:`` cross-
reference or a name in a table — because whole groups (the format reductions,
for instance) are covered by a table rather than by per-function entries, and
that is a deliberate choice rather than a gap. What it will not tolerate is an
export the reference never names at all, which is indistinguishable from one
nobody remembered to document.

Anything genuinely meant to be absent goes in ``EXEMPT_FROM_THE_API_REFERENCE``
with a reason. An exemption is a claim a reviewer can read and disagree with; a
name quietly missing from a hand-written list is not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import mlx_sparse as ms
from mlx_sparse import linalg

DIRECTIVE = re.compile(
    r"^\.\.\s+auto(?:function|class|data|method|attribute|module)::\s+(\S+)", re.M
)
# A cross-reference role, ``:func:`csr_row_sums``` and friends.
ROLE = re.compile(r":(?:func|class|data|meth|attr|obj|mod):`~?([A-Za-z_][\w.]*)`")
# A bare literal, which is how the runtime options are listed in their table.
LITERAL = re.compile(r"``([A-Za-z_][\w.]*)``")

# Every namespace whose ``__all__`` is a public promise.
PUBLIC_NAMESPACES = (
    ("mlx_sparse", ms),
    ("mlx_sparse.linalg", linalg),
    ("mlx_sparse.random", ms.random),
    ("mlx_sparse.runtime", ms.runtime),
)

# Exported names deliberately absent from the API reference, each with the
# reason. Empty is the healthy state. An entry here is a decision on the
# record; a name missing from a hand-written checklist is an accident nobody
# can see.
EXEMPT_FROM_THE_API_REFERENCE: dict[str, str] = {}


def _api_dir():
    return Path(__file__).resolve().parents[1] / "docs" / "api"


def _api_text():
    return "\n".join(rst.read_text() for rst in sorted(_api_dir().glob("*.rst")))


def _documented_names():
    """Names carrying a directive-grade entry, for the named-family cells."""
    names = {m.split(".")[-1] for m in DIRECTIVE.findall(_api_text())}
    # Without this, a moved docs tree or a changed directive spelling would make
    # every check below pass against an empty set.
    assert len(names) > 50, f"parsed only {len(names)} directives from {_api_dir()}"
    return names


def _api_mentions():
    """Every name the API reference names at all, however it names it."""
    text = _api_text()
    names = _documented_names()
    names.update(m.split(".")[-1] for m in ROLE.findall(text))
    names.update(m.split(".")[-1] for m in LITERAL.findall(text))
    return names


def undocumented_exports(namespaces, mentioned, exempt):
    """``"namespace.name"`` for every export the reference never names.

    Kept separate from the cell that calls it so the rule can be exercised on
    inputs of our own making. A discovery check that has only ever run against
    the real tree has never been shown to report anything.
    """
    missing = []
    for label, namespace in namespaces:
        exported = getattr(namespace, "__all__", None)
        assert exported, f"{label} exports no __all__ to check"
        for name in sorted(exported):
            if name in mentioned or name in exempt:
                continue
            missing.append(f"{label}.{name}")
    return missing


@pytest.mark.parametrize("name", ["csr_array", "csc_array", "coo_array"])
def test_every_array_constructor_has_an_api_entry(name):
    assert name in ms.__all__
    assert name in _documented_names()


@pytest.mark.parametrize("name", ["CSRArray", "CSCArray", "COOArray"])
def test_every_array_container_has_an_api_entry(name):
    assert name in _documented_names()


@pytest.mark.parametrize("name", ["spsolve", "spsolve_triangular", "factorized"])
def test_every_sparse_solve_entrypoint_has_an_api_entry(name):
    assert name in linalg.__all__
    assert name in _documented_names()


@pytest.mark.parametrize("name", ["cg", "bicgstab", "gmres", "minres"])
def test_every_iterative_solver_has_an_api_entry(name):
    assert name in linalg.__all__
    assert name in _documented_names()


@pytest.mark.parametrize("name", ["lanczos", "eigsh", "eigs", "svds"])
def test_every_spectral_routine_has_an_api_entry(name):
    assert name in linalg.__all__
    assert name in _documented_names()


def test_the_api_reference_does_not_document_names_that_no_longer_exist():
    """A renamed or removed export leaves a directive Sphinx cannot resolve."""
    namespaces = (ms, linalg, ms.random, ms.runtime)
    live = set()
    for namespace in namespaces:
        live.update(dir(namespace))

    ghosts = sorted(name for name in _documented_names() if name not in live)

    assert ghosts == [], f"documented but not importable: {ghosts}"


def test_every_export_is_reachable_from_the_api_reference():
    """The check the hand-written lists above cannot make.

    They can only ask about names someone typed. This asks the package what it
    exports and requires the reference to name each one somewhere, so a new
    export nobody remembered to document is reported rather than overlooked.
    """
    missing = undocumented_exports(
        PUBLIC_NAMESPACES, _api_mentions(), EXEMPT_FROM_THE_API_REFERENCE
    )
    assert missing == [], (
        "these names are exported but the API reference never mentions them, so "
        f"a reader cannot discover they exist: {missing}. Document them, or add "
        "each to EXEMPT_FROM_THE_API_REFERENCE with the reason it is deliberate."
    )


class TestTheDiscoveryRuleItself:
    """Teeth. The cell above passes when the tree is healthy, which is most of
    the time, so on its own it is indistinguishable from a cell that cannot
    fail. These run the rule against namespaces built to break it."""

    class _Namespace:
        def __init__(self, **names):
            self.__all__ = sorted(names)
            self.__dict__.update(names)

    def test_an_undocumented_export_is_reported(self):
        ns = self._Namespace(documented=1, forgotten=2)
        missing = undocumented_exports([("pkg", ns)], {"documented"}, {})
        assert missing == ["pkg.forgotten"]

    def test_an_exempt_export_is_not_reported(self):
        ns = self._Namespace(documented=1, deliberate=2)
        missing = undocumented_exports(
            [("pkg", ns)], {"documented"}, {"deliberate": "a reason"}
        )
        assert missing == []

    def test_names_are_reported_from_every_namespace_not_just_the_first(self):
        """Two namespaces, because one cannot show the loop visits them all."""
        first = self._Namespace(a=1, missing_from_first=2)
        second = self._Namespace(b=1, missing_from_second=2)
        missing = undocumented_exports(
            [("one", first), ("two", second)], {"a", "b"}, {}
        )
        assert missing == ["one.missing_from_first", "two.missing_from_second"]

    def test_a_namespace_without_exports_is_an_error_not_a_pass(self):
        """An empty ``__all__`` would otherwise read as a clean namespace."""
        with pytest.raises(AssertionError, match="exports no __all__"):
            undocumented_exports([("empty", self._Namespace())], set(), {})

    def test_every_exemption_carries_a_reason(self):
        for name, reason in EXEMPT_FROM_THE_API_REFERENCE.items():
            assert isinstance(reason, str) and len(reason) > 20, (
                f"{name} is exempted from the API reference without saying why"
            )
