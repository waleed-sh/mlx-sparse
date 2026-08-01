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

"""The API reference must not omit one member of a documented family.

Scoped deliberately: this does not require every export to have a page, because
whole groups (the format reductions, for instance) are intentionally covered by
prose rather than per-function entries. It requires that where a family IS
documented, no sibling is missing, which is the shape a gap actually takes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import mlx_sparse as ms
from mlx_sparse import linalg

DIRECTIVE = re.compile(
    r"^\.\.\s+auto(?:function|class|data|method|attribute)::\s+(\S+)", re.M
)


def _documented_names():
    api_dir = Path(__file__).resolve().parents[1] / "docs" / "api"
    names = set()
    for rst in api_dir.glob("*.rst"):
        names.update(m.split(".")[-1] for m in DIRECTIVE.findall(rst.read_text()))
    # Without this, a moved docs tree or a changed directive spelling would make
    # every check below pass against an empty set.
    assert len(names) > 50, f"parsed only {len(names)} directives from {api_dir}"
    return names


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
