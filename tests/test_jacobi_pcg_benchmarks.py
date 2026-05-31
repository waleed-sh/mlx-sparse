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

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _jacobi_benchmark():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "bench_jacobi_pcg_validation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bench_jacobi_pcg_validation_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scaled_diagonal_family_has_requested_condition():
    bench = _jacobi_benchmark()

    matrix = bench.make_scaled_diagonal(8, condition=1.0e6)
    diag = matrix.diagonal()

    assert matrix.shape == (8, 8)
    assert matrix.nnz == 8
    assert np.all(diag > 0.0)
    assert diag[-1] / diag[0] == pytest.approx(1.0e6, rel=1e-6)


def test_scaled_poisson_family_is_symmetric_positive_definite():
    bench = _jacobi_benchmark()

    matrix = bench.make_scaled_poisson_1d(6, scale_span=10.0)
    dense = matrix.toarray()

    np.testing.assert_allclose(dense, dense.T, rtol=1e-7, atol=1e-7)
    assert np.linalg.eigvalsh(dense).min() > 0.0
    assert np.all(matrix.diagonal() > 0.0)


def test_scipy_jacobi_operator_applies_inverse_diagonal():
    bench = _jacobi_benchmark()
    matrix = bench.make_scaled_diagonal(4, condition=16.0)
    rhs = np.arange(1, 5, dtype=np.float32)

    operator = bench.scipy_jacobi_operator(matrix)
    got = operator.matvec(rhs)

    np.testing.assert_allclose(got, rhs / matrix.diagonal(), rtol=1e-7)


def test_summarize_records_reports_convergence_and_residual_bounds():
    bench = _jacobi_benchmark()
    records = [
        {
            "family": "scaled_diagonal",
            "n": 4,
            "info": 0,
            "relative_true_residual": 1.0e-7,
        },
        {
            "family": "scaled_poisson_1d",
            "n": 8,
            "info": 4,
            "relative_true_residual": 1.0e-4,
        },
    ]

    summary = bench.summarize_records(records)

    assert summary == {
        "record_count": 2,
        "converged_count": 1,
        "max_relative_true_residual": 1.0e-4,
        "families": ["scaled_diagonal", "scaled_poisson_1d"],
        "sizes": [4, 8],
    }
