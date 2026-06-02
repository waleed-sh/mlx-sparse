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


def _validation_benchmark():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "bench_preconditioner_validation_matrix.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bench_preconditioner_validation_matrix_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("family", "size", "kind"),
    [
        ("poisson_1d", 6, "spd"),
        ("poisson_2d", 3, "spd"),
        ("poisson_3d", 2, "spd"),
        ("anisotropic_diffusion_2d", 3, "spd"),
        ("badly_scaled_diagonal", 6, "spd"),
        ("block_diagonal_spd", 3, "spd"),
        ("convection_diffusion_1d", 6, "general"),
        ("random_diagonal_dominant", 6, "general"),
        ("hilbert_like", 5, "spd"),
        ("suitesparse_well1033_normal", 4, "spd"),
        ("suitesparse_illc1033_normal", 4, "spd"),
    ],
)
def test_validation_matrix_families_have_expected_structure(family, size, kind):
    bench = _validation_benchmark()

    case = bench.make_matrix_case(family, size)

    assert case.family == family
    assert case.kind == kind
    assert case.matrix.shape[0] == case.matrix.shape[1]
    assert case.matrix.nnz > 0
    assert case.rhs.shape == (case.matrix.shape[0],)
    assert np.isfinite(case.norm_rhs)
    assert case.norm_rhs > 0.0
    if kind == "spd":
        dense = case.matrix.toarray()
        np.testing.assert_allclose(dense, dense.T, rtol=2e-5, atol=2e-5)
        assert np.linalg.eigvalsh(dense.astype(np.float64)).min() > -1e-5


def test_validation_matrix_scipy_jacobi_operator_matches_inverse_diagonal():
    bench = _validation_benchmark()
    matrix = bench.badly_scaled_diagonal(5, condition=25.0)
    rhs = np.arange(1, 6, dtype=np.float32)

    operator = bench.scipy_jacobi_operator(matrix)

    np.testing.assert_allclose(operator @ rhs, rhs / matrix.diagonal(), rtol=1e-7)


def test_validation_matrix_record_schema_for_small_jacobi_case():
    mx = pytest.importorskip("mlx.core")
    ms = pytest.importorskip("mlx_sparse")
    mx.set_default_device(mx.Device(mx.cpu, 0))
    ms.use_cpu(require_available=False)
    bench = _validation_benchmark()
    try:
        bench.to_mlx_csr(bench.poisson_1d(2))
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("local MLX runtime routes CPU array construction through Metal")
        raise

    records = bench.benchmark_case(
        family="poisson_1d",
        size=6,
        warmup=0,
        iters=1,
        rtol=1e-4,
        atol=1e-7,
        restart=8,
        maxiter=64,
        include_scipy=True,
        selected_preconditioners=("none", "jacobi", "scipy_jacobi"),
    )

    assert {record["implementation"] for record in records} == {"mlx_sparse", "scipy"}
    assert {record["preconditioner"] for record in records} == {"none", "jacobi"}
    for record in records:
        assert record["suite"] == "preconditioner_validation_matrix"
        assert record["matrix"]["family"] == "poisson_1d"
        assert record["matrix"]["kind"] == "spd"
        assert record["matrix"]["shape"] == [6, 6]
        assert record["matrix"]["nnz"] > 0
        assert "dtype" in record["matrix"]
        assert "index_dtype" in record["matrix"]
        assert "solver_n_threads" in record["runtime"]
        assert "device" in record["settings"]
        assert record["solve_time_ms"]["median_ms"] >= 0.0
        assert record["iterations"] <= record["thresholds"]["max_iterations"]
        assert record["threshold_status"]["passed"] is True


def test_validation_matrix_summary_reports_threshold_status():
    bench = _validation_benchmark()
    records = [
        {
            "matrix": {"family": "poisson_1d"},
            "implementation": "mlx_sparse",
            "status": 0,
            "relative_true_residual": 1.0e-7,
            "threshold_status": {"passed": True},
        },
        {
            "matrix": {"family": "convection_diffusion_1d"},
            "implementation": "scipy",
            "status": 0,
            "relative_true_residual": 2.0e-7,
            "threshold_status": {"passed": True},
        },
    ]

    summary = bench.summarize_records(records)

    assert summary == {
        "record_count": 2,
        "solved_count": 2,
        "threshold_pass_count": 2,
        "max_relative_true_residual": 2.0e-7,
        "families": ["convection_diffusion_1d", "poisson_1d"],
        "implementations": ["mlx_sparse", "scipy"],
    }
