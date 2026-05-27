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

import pytest

from mlx_sparse._ext_loader import extension

pytestmark = [pytest.mark.native, pytest.mark.accelerate]


def _require_ext():
    ext = extension()
    if ext is None:
        pytest.skip("native extension unavailable")
    return ext


def _require_accelerate_factorization(ext):
    summary = ext._accelerate_factorization_wrapper_summary_for_testing()
    if not summary["accelerate_framework"]:
        pytest.skip("Accelerate framework support is not compiled in")
    return summary


def test_factorization_wrapper_summary_reflects_framework_state():
    ext = _require_ext()

    summary = ext._accelerate_factorization_wrapper_summary_for_testing()

    assert isinstance(summary["accelerate_framework"], bool)
    if not summary["accelerate_framework"]:
        assert set(summary) == {"accelerate_framework"}
        return

    assert summary["symbolic_status"] == 0
    assert summary["retained_symbolic_status"] == 0
    assert summary["numeric_status"] == 0
    assert summary["retained_numeric_status"] == 0
    assert summary["factorization_type"] == "SparseFactorizationCholesky"
    assert summary["row_count"] == 2
    assert summary["column_count"] == 2
    assert summary["rhs_size"] == 2
    assert summary["solution_size"] == 2
    assert summary["moved_from_owns"] is False
    assert summary["moved_to_owns"] is True

    workspace_static = summary["solve_workspace_static"]
    workspace_per_rhs = summary["solve_workspace_per_rhs"]
    assert summary["solve_workspace_one_rhs"] == workspace_static + workspace_per_rhs
    assert workspace_static >= 0
    assert workspace_per_rhs >= 0

    assert list(summary["solution"]) == pytest.approx([1.0 / 11.0, 7.0 / 11.0])
    assert summary["subfactor_status"] == 0
    assert summary["subfactor_contents"] == "SparseSubfactorL"
    assert summary["retained_subfactor_status"] == 0


def test_factorization_failure_maps_to_python_exception():
    ext = _require_ext()
    _require_accelerate_factorization(ext)

    with pytest.raises(
        RuntimeError,
        match="SparseFactorizationFailed|SparseMatrixIsSingular|numerical|singular",
    ):
        ext._accelerate_factorization_failure_for_testing()


def test_factorization_hooks_keep_unavailable_builds_non_trapping():
    ext = _require_ext()
    summary = ext._accelerate_factorization_wrapper_summary_for_testing()

    if summary["accelerate_framework"]:
        return

    with pytest.raises(RuntimeError, match="not available"):
        ext._accelerate_factorization_failure_for_testing()
