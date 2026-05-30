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

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _sparse_ops_benchmark():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "bench_native_cpu_sparse_ops.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bench_native_cpu_sparse_ops_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_cases_cover_required_bottleneck_families():
    bench = _sparse_ops_benchmark()
    rng = np.random.default_rng(0)

    cases = bench.make_cases(
        families=list(bench.FAMILIES),
        size=12,
        density=0.05,
        short_row_nnz=3,
        duplicate_factor=3,
        output_densities=[0.02, 0.08],
        rng=rng,
    )

    families = {case.family for case in cases}
    assert {
        "uniform_short_rows",
        "imbalanced_rows",
        "banded",
        "diagonal_dominant",
        "duplicate_heavy",
        "exact_cancellation",
        "output_density_sweep",
    }.issubset(families)
    assert {
        case.target_output_density
        for case in cases
        if case.family == "output_density_sweep"
    } == {0.02, 0.08}

    duplicate_case = next(case for case in cases if case.family == "duplicate_heavy")
    assert _has_duplicate_compressed_entries(duplicate_case.primary)

    exact_case = next(case for case in cases if case.family == "exact_cancellation")
    product = exact_case.primary @ exact_case.spgemm_rhs
    product.eliminate_zeros()
    assert product.nnz == 0


def test_shuffle_preserves_csr_dense_values_but_changes_structure_order():
    bench = _sparse_ops_benchmark()
    rng = np.random.default_rng(1)
    matrix = bench.uniform_short_rows(8, 16, 5, rng)

    shuffled = bench.shuffle_csr_within_rows(matrix, rng)

    np.testing.assert_allclose(shuffled.toarray(), matrix.toarray())
    assert any(
        not np.all(
            shuffled.indices[shuffled.indptr[row] : shuffled.indptr[row + 1]][1:]
            >= shuffled.indices[shuffled.indptr[row] : shuffled.indptr[row + 1]][:-1]
        )
        for row in range(shuffled.shape[0])
        if shuffled.indptr[row + 1] - shuffled.indptr[row] > 1
    )


def test_density_for_size_keeps_target_row_occupancy():
    bench = _sparse_ops_benchmark()

    assert bench.density_for_size(1024, target_nnz_per_row=16) == pytest.approx(
        16 / 1024
    )
    assert bench.density_for_size(16, target_nnz_per_row=16) == pytest.approx(0.25)
    assert bench.density_for_size(
        1024, target_nnz_per_row=64, max_nnz_per_matrix=8192
    ) == pytest.approx(8192 / (1024 * 1024))


def test_sweep_cases_cover_sizes_and_target_nnz_grid():
    bench = _sparse_ops_benchmark()
    rng = np.random.default_rng(2)

    cases = bench.make_sweep_cases(
        families=["uniform_short_rows", "diagonal_dominant"],
        sizes=[8, 16],
        densities=None,
        target_nnzs_per_row=[2, 4],
        max_density=0.25,
        short_row_nnzs=[2, 4],
        duplicate_factor=3,
        output_densities=None,
        output_target_nnzs_per_row=[2],
        max_nnz_per_matrix=0,
        rng=rng,
    )

    assert {case.size for case in cases} == {8, 16}
    assert {
        case.short_row_nnz for case in cases if case.family == "uniform_short_rows"
    } == {2, 4}
    assert {
        case.target_nnz_per_row for case in cases if case.family == "diagonal_dominant"
    } == {2.0, 4.0}
    assert {
        case.effective_density
        for case in cases
        if case.family == "diagonal_dominant" and case.size == 16
    } == {0.125, 0.25}


def test_benchmark_fromdense_record_has_verification_and_timing(mx):
    bench = _sparse_ops_benchmark()
    rng = np.random.default_rng(3)
    matrix = bench.uniform_short_rows(4, 4, 2, rng)
    case = bench.MatrixCase(
        family="uniform_short_rows",
        label="uniform_short_rows_n4_r2",
        primary=matrix,
        spgemm_rhs=matrix,
        size=4,
        short_row_nnz=2,
    )
    records = []

    bench._bench_fromdense(
        records,
        case,
        mx.int32,
        warmup=0,
        iters=1,
        verify=True,
        verify_max_elements=64,
        max_dense_elements=64,
    )

    assert len(records) == 1
    record = records[0]
    assert record["suite"] == "fromdense"
    assert record["operation"] == "fromdense"
    assert record["verification"]["status"] == "checked_sparse_dense"
    assert record["output"]["kind"] == "sparse"
    assert record["scipy"]["status"] == "timed"
    assert record["scipy"]["output"]["format"] == "csr"
    assert record["scipy"]["timing"]["median_ms"] >= 0.0
    assert record["timing"]["median_ms"] >= 0.0
    assert record["key"].startswith("fromdense|fromdense|uniform_short_rows_n4_r2")


def test_fromdense_large_cases_are_reported_as_skipped(mx):
    bench = _sparse_ops_benchmark()
    rng = np.random.default_rng(4)
    matrix = bench.uniform_short_rows(8, 8, 2, rng)
    case = bench.MatrixCase(
        family="uniform_short_rows",
        label="uniform_short_rows_n8_r2",
        primary=matrix,
        spgemm_rhs=matrix,
        size=8,
        short_row_nnz=2,
    )
    records = []

    bench._bench_fromdense(
        records,
        case,
        mx.int32,
        warmup=0,
        iters=1,
        verify=True,
        verify_max_elements=64,
        max_dense_elements=16,
    )

    assert records[0]["verification"]["status"] == "skipped"
    assert records[0]["scipy"]["status"] == "skipped_with_native"
    assert "timing" not in records[0]
    assert "dense materialization" in records[0]["skip_reason"]


def test_coo_csc_dense_product_benchmarks_include_batched_records(mx):
    bench = _sparse_ops_benchmark()
    rng = np.random.default_rng(5)
    matrix = bench.uniform_short_rows(6, 8, 2, rng)
    case = bench.MatrixCase(
        family="uniform_short_rows",
        label="uniform_short_rows_n6_r2",
        primary=matrix,
        spgemm_rhs=matrix,
        size=6,
        short_row_nnz=2,
    )
    records = []

    bench._bench_coo_csc_dense_products(
        records,
        case,
        rhs_cols=2,
        batch_size=3,
        index_dtype=mx.int32,
        warmup=0,
        iters=1,
        verify=True,
        verify_max_elements=256,
        rng=rng,
    )

    assert {
        "coo_matvec",
        "coo_matmul",
        "coo_batched_matvec",
        "coo_batched_matmul",
        "csc_matvec",
        "csc_matmul",
        "csc_batched_matvec",
        "csc_batched_matmul",
    } == {record["operation"] for record in records}
    assert all(
        record["verification"]["status"] == "checked_dense" for record in records
    )


def test_baseline_comparison_uses_loose_thresholds(tmp_path):
    bench = _sparse_ops_benchmark()
    baseline = {
        "records": [
            {
                "key": "suite|op|family|fmt|out|4x4|none",
                "timing": {"median_ms": 10.0},
            }
        ]
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current = [
        {
            "key": "suite|op|family|fmt|out|4x4|none",
            "timing": {"median_ms": 24.0},
        },
        {
            "key": "new|record",
            "timing": {"median_ms": 1.0},
        },
    ]

    comparison = bench.compare_to_baseline(
        current_records=current,
        baseline_path=baseline_path,
        factor=2.5,
        absolute_ms=50.0,
    )

    assert comparison["failures"] == 0
    assert comparison["records"][0]["status"] == "pass"
    assert comparison["records"][1]["status"] == "missing_baseline"


def test_validation_rejects_dimensions_above_32k():
    bench = _sparse_ops_benchmark()
    args = argparse.Namespace(
        size=None,
        sizes=[bench.MAX_BENCHMARK_DIMENSION + 1],
        rhs_cols=1,
        batch_size=4,
        density=None,
        densities=[0.01],
        target_nnz_per_row=None,
        target_nnzs_per_row=None,
        max_density=0.25,
        short_row_nnz=None,
        short_row_nnzs=[2],
        duplicate_factor=2,
        warmup=0,
        iters=1,
        output_densities=[0.01],
        output_target_nnz_per_row=None,
        output_target_nnzs_per_row=None,
        max_dense_elements=1024,
        max_nnz_per_matrix=1024,
        verify_max_elements=1024,
        regression_factor=2.5,
        regression_absolute_ms=50.0,
    )

    with pytest.raises(ValueError, match="32768"):
        bench._validate_args(args)


def _has_duplicate_compressed_entries(matrix) -> bool:
    csr = matrix.tocsr(copy=True)
    for row in range(csr.shape[0]):
        start, end = int(csr.indptr[row]), int(csr.indptr[row + 1])
        row_indices = csr.indices[start:end]
        if np.unique(row_indices).size != row_indices.size:
            return True
    return False
