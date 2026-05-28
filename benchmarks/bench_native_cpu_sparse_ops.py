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

"""CPU-only native sparse-operation benchmark suites for v0.0.4b1.

The suites here cover dynamic-output construction/canonicalization, same-format
SpGEMM, transpose sparse-dense products, and COO/CSC dense products.  Reports
are structured so the same command shape can be used before and after each CPU
optimization:

    python benchmarks/bench_native_cpu_sparse_ops.py \\
      --run-label before --report-dir benchmarks/reports/v0.0.4b1

Then compare after an optimization:

    python benchmarks/bench_native_cpu_sparse_ops.py \\
      --run-label after --baseline-report benchmarks/reports/v0.0.4b1/<before>.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
import scipy.sparse as sp

import mlx_sparse as ms
import mlx_sparse._native as _native
from mlx_sparse._host import to_numpy

try:
    from benchmarks.benchmark_utils import (
        BenchmarkTiming,
        cpu_runtime_metadata,
        force_eval,
        force_scipy_eval,
        scipy_speedup,
        sparse_matrix_metadata,
        time_result,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from benchmarks/
    from benchmark_utils import (  # type: ignore
        BenchmarkTiming,
        cpu_runtime_metadata,
        force_eval,
        force_scipy_eval,
        scipy_speedup,
        sparse_matrix_metadata,
        time_result,
    )


FAMILIES = (
    "uniform_short_rows",
    "imbalanced_rows",
    "banded",
    "diagonal_dominant",
    "duplicate_heavy",
    "exact_cancellation",
    "output_density_sweep",
)

SUITES = (
    "fromdense",
    "compressed",
    "coo_conversion",
    "spgemm",
    "transpose_products",
    "coo_csc_dense_products",
)

MAX_BENCHMARK_DIMENSION = 32_768
DEFAULT_SIZE_GRID = [128, 512, 2_048, 8_192, 32_768]
DEFAULT_TARGET_NNZS_PER_ROW = [2.0, 8.0, 32.0]
DEFAULT_SHORT_ROW_NNZS = [2, 8, 32]
DEFAULT_OUTPUT_TARGET_NNZS_PER_ROW = [2.0, 8.0, 32.0]
DEFAULT_MAX_DENSITY = 0.25
DEFAULT_MAX_DENSE_ELEMENTS = 16_777_216
DEFAULT_MAX_NNZ_PER_MATRIX = 2_000_000

DENSITY_SENSITIVE_FAMILIES = frozenset(
    {
        "imbalanced_rows",
        "banded",
        "diagonal_dominant",
        "duplicate_heavy",
    }
)


@dataclass(frozen=True, slots=True)
class MatrixCase:
    """A benchmark matrix family instance."""

    family: str
    label: str
    primary: sp.csr_matrix
    spgemm_rhs: sp.csr_matrix
    size: int | None = None
    requested_density: float | None = None
    effective_density: float | None = None
    density_mode: str | None = None
    target_nnz_per_row: float | None = None
    max_density: float | None = None
    short_row_nnz: int | None = None
    requested_output_density: float | None = None
    target_output_density: float | None = None
    output_density_mode: str | None = None
    target_output_nnz_per_row: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark native CPU sparse construction and product suites."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Compatibility alias for a single-entry --sizes sweep.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Matrix dimensions to sweep. Defaults to a broad grid up to 32k, "
            "values greater than 32768 are rejected."
        ),
    )
    parser.add_argument("--rhs-cols", type=int, default=8)
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Compatibility alias for a single-entry --densities sweep.",
    )
    parser.add_argument(
        "--densities",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Explicit sparse densities for density-sensitive matrix families. "
            "When omitted, densities are derived from --target-nnzs-per-row."
        ),
    )
    parser.add_argument(
        "--target-nnz-per-row",
        type=float,
        default=None,
        help="Compatibility alias for a single-entry --target-nnzs-per-row sweep.",
    )
    parser.add_argument(
        "--target-nnzs-per-row",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Target nonzeros per row for density-sensitive matrix families. "
            "Each effective density is min(--max-density, target / n)."
        ),
    )
    parser.add_argument(
        "--max-density",
        type=float,
        default=DEFAULT_MAX_DENSITY,
        help="Upper density clamp for target-nnz-per-row sweeps.",
    )
    parser.add_argument(
        "--short-row-nnz",
        type=int,
        default=None,
        help="Compatibility alias for a single-entry --short-row-nnzs sweep.",
    )
    parser.add_argument(
        "--short-row-nnzs",
        nargs="+",
        type=int,
        default=None,
        help="Per-row nnz values for uniformly short-row cases.",
    )
    parser.add_argument("--duplicate-factor", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--index-dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument(
        "--output-densities",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Explicit target output densities for the output_density_sweep "
            "SpGEMM family. When omitted, output densities are derived from "
            "--output-target-nnzs-per-row."
        ),
    )
    parser.add_argument(
        "--output-target-nnz-per-row",
        type=float,
        default=None,
        help=(
            "Compatibility alias for a single-entry "
            "--output-target-nnzs-per-row sweep."
        ),
    )
    parser.add_argument(
        "--output-target-nnzs-per-row",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Target output nonzeros per row for output-density SpGEMM sweeps. "
            "Each target output density is min(--max-density, target / n)."
        ),
    )
    parser.add_argument(
        "--max-dense-elements",
        type=int,
        default=DEFAULT_MAX_DENSE_ELEMENTS,
        help=(
            "Maximum dense elements to materialize for fromdense benchmarks. "
            "Larger cases are reported as skipped instead of allocating dense memory."
        ),
    )
    parser.add_argument(
        "--max-nnz-per-matrix",
        type=int,
        default=DEFAULT_MAX_NNZ_PER_MATRIX,
        help=(
            "Soft cap for generated sparse operands. Requested densities are "
            "clamped for large dimensions so default sweeps stay local-machine safe."
        ),
    )
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--verify-max-elements",
        type=int,
        default=262_144,
        help="Maximum dense elements to compare during one-time verification.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--run-label", default="before")
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--regression-factor", type=float, default=2.5)
    parser.add_argument("--regression-absolute-ms", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    sizes = _resolve_sizes(args)
    explicit_densities = _resolve_densities(args)
    target_nnzs_per_row = _resolve_target_nnzs_per_row(args)
    short_row_nnzs = _resolve_short_row_nnzs(args)
    explicit_output_densities = _resolve_output_densities(args)
    output_target_nnzs_per_row = _resolve_output_target_nnzs_per_row(args)

    runtime = cpu_runtime_metadata(warmup=args.warmup, iters=args.iters)
    if not runtime["native_extension_available"]:
        raise RuntimeError("Native sparse-op benchmarks require mlx_sparse._ext.")

    rng = np.random.default_rng(args.seed)
    index_dtype = mx.int32 if args.index_dtype == "int32" else mx.int64
    records: list[dict[str, Any]] = []

    for case in make_sweep_cases(
        families=args.families,
        sizes=sizes,
        densities=explicit_densities,
        target_nnzs_per_row=target_nnzs_per_row,
        max_density=args.max_density,
        short_row_nnzs=short_row_nnzs,
        duplicate_factor=args.duplicate_factor,
        output_densities=explicit_output_densities,
        output_target_nnzs_per_row=output_target_nnzs_per_row,
        max_nnz_per_matrix=args.max_nnz_per_matrix,
        rng=rng,
    ):
        _run_case_suites(
            records,
            case=case,
            suites=args.suites,
            rhs_cols=args.rhs_cols,
            index_dtype=index_dtype,
            warmup=args.warmup,
            iters=args.iters,
            verify=args.verify,
            verify_max_elements=args.verify_max_elements,
            max_dense_elements=args.max_dense_elements,
            rng=rng,
        )

    report = {
        "benchmark": "native_cpu_sparse_ops",
        "version_target": "0.0.4b1",
        "mode": "cpu_only_native_non_accelerate",
        "run_label": args.run_label,
        "runtime": runtime,
        "args": {
            "sizes": sizes,
            "rhs_cols": args.rhs_cols,
            "density_mode": (
                "explicit_density"
                if explicit_densities is not None
                else "target_nnz_per_row"
            ),
            "densities": explicit_densities,
            "target_nnzs_per_row": target_nnzs_per_row,
            "max_density": args.max_density,
            "short_row_nnzs": short_row_nnzs,
            "duplicate_factor": args.duplicate_factor,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "index_dtype": args.index_dtype,
            "families": args.families,
            "suites": args.suites,
            "output_density_mode": (
                "explicit_density"
                if explicit_output_densities is not None
                else "target_nnz_per_row"
            ),
            "output_densities": explicit_output_densities,
            "output_target_nnzs_per_row": output_target_nnzs_per_row,
            "max_dimension": MAX_BENCHMARK_DIMENSION,
            "max_dense_elements": args.max_dense_elements,
            "max_nnz_per_matrix": args.max_nnz_per_matrix,
            "verify": args.verify,
            "verify_max_elements": args.verify_max_elements,
        },
        "regression_policy": {
            "factor": args.regression_factor,
            "absolute_ms": args.regression_absolute_ms,
        },
        "records": records,
    }
    if args.baseline_report is not None:
        report["comparison"] = compare_to_baseline(
            current_records=records,
            baseline_path=args.baseline_report,
            factor=args.regression_factor,
            absolute_ms=args.regression_absolute_ms,
        )

    report_path = _write_report(report, output=args.output, report_dir=args.report_dir)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.json or report_path is not None:
        print(payload)
    else:
        print(_format_text(report))


def make_sweep_cases(
    *,
    families: list[str],
    sizes: list[int],
    short_row_nnzs: list[int],
    duplicate_factor: int,
    max_nnz_per_matrix: int,
    rng: np.random.Generator,
    densities: list[float] | None = None,
    target_nnzs_per_row: list[float] | None = None,
    max_density: float = DEFAULT_MAX_DENSITY,
    output_densities: list[float] | None = None,
    output_target_nnzs_per_row: list[float] | None = None,
) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for size in sizes:
        cases.extend(
            _make_cases_for_size(
                families=families,
                size=size,
                densities=densities,
                target_nnzs_per_row=target_nnzs_per_row,
                max_density=max_density,
                short_row_nnzs=short_row_nnzs,
                duplicate_factor=duplicate_factor,
                output_densities=output_densities,
                output_target_nnzs_per_row=output_target_nnzs_per_row,
                max_nnz_per_matrix=max_nnz_per_matrix,
                rng=rng,
            )
        )
    return cases


def make_cases(
    *,
    families: list[str],
    size: int,
    density: float,
    short_row_nnz: int,
    duplicate_factor: int,
    output_densities: list[float],
    rng: np.random.Generator,
) -> list[MatrixCase]:
    """Build a single-size compatibility case set for tests and ad-hoc runs."""

    return _make_cases_for_size(
        families=families,
        size=size,
        densities=[density],
        target_nnzs_per_row=None,
        max_density=DEFAULT_MAX_DENSITY,
        short_row_nnzs=[short_row_nnz],
        duplicate_factor=duplicate_factor,
        output_densities=output_densities,
        output_target_nnzs_per_row=None,
        max_nnz_per_matrix=0,
        rng=rng,
    )


def _make_cases_for_size(
    *,
    families: list[str],
    size: int,
    short_row_nnzs: list[int],
    duplicate_factor: int,
    max_nnz_per_matrix: int,
    rng: np.random.Generator,
    densities: list[float] | None,
    target_nnzs_per_row: list[float] | None,
    max_density: float,
    output_densities: list[float] | None,
    output_target_nnzs_per_row: list[float] | None,
) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    density_specs = _density_specs_for_size(
        size=size,
        densities=densities,
        target_nnzs_per_row=target_nnzs_per_row,
        max_density=max_density,
        max_nnz_per_matrix=max_nnz_per_matrix,
    )
    output_density_specs = _output_density_specs_for_size(
        size=size,
        output_densities=output_densities,
        output_target_nnzs_per_row=output_target_nnzs_per_row,
        max_density=max_density,
        max_nnz_per_matrix=max_nnz_per_matrix,
    )
    for family in families:
        if family == "uniform_short_rows":
            for row_nnz in short_row_nnzs:
                primary = uniform_short_rows(size, size, row_nnz, rng)
                rhs = uniform_short_rows(
                    primary.shape[1], primary.shape[0], row_nnz, rng
                )
                cases.append(
                    MatrixCase(
                        family=family,
                        label=_case_label(family, size, short_row_nnz=row_nnz),
                        primary=primary,
                        spgemm_rhs=rhs,
                        size=size,
                        short_row_nnz=int(min(row_nnz, size)),
                    )
                )
            continue

        if family == "exact_cancellation":
            lhs, rhs = exact_cancellation_pair(size)
            cases.append(
                MatrixCase(
                    family=family,
                    label=_case_label(family, size),
                    primary=lhs,
                    spgemm_rhs=rhs,
                    size=size,
                )
            )
            continue

        if family == "output_density_sweep":
            for spec in output_density_specs:
                lhs, rhs = output_density_sweep_pair(
                    size,
                    spec["effective_density"],
                    rng,
                    max_nnz_per_matrix=max_nnz_per_matrix,
                )
                cases.append(
                    MatrixCase(
                        family=family,
                        label=_case_label(
                            family,
                            size,
                            target_output_density=spec["effective_density"],
                            requested_output_density=spec["requested_density"],
                            target_output_nnz_per_row=spec["target_nnz_per_row"],
                        ),
                        primary=lhs,
                        spgemm_rhs=rhs,
                        size=size,
                        requested_output_density=spec["requested_density"],
                        target_output_density=spec["effective_density"],
                        output_density_mode=spec["density_mode"],
                        target_output_nnz_per_row=spec["target_nnz_per_row"],
                    )
                )
            continue

        density_values = (
            density_specs if family in DENSITY_SENSITIVE_FAMILIES else [None]
        )
        for density_spec in density_values:
            effective_density = (
                0.0
                if density_spec is None
                else float(density_spec["effective_density"])
            )
            primary = primary_matrix(
                family,
                size=size,
                density=effective_density,
                short_row_nnz=short_row_nnzs[0],
                duplicate_factor=duplicate_factor,
                rng=rng,
            )
            rhs = spgemm_rhs_matrix(
                family,
                lhs=primary,
                density=effective_density,
                short_row_nnz=short_row_nnzs[0],
                duplicate_factor=duplicate_factor,
                rng=rng,
            )
            cases.append(
                MatrixCase(
                    family=family,
                    label=_case_label(
                        family,
                        size,
                        requested_density=(
                            None
                            if density_spec is None
                            else density_spec["requested_density"]
                        ),
                        effective_density=effective_density,
                        target_nnz_per_row=(
                            None
                            if density_spec is None
                            else density_spec["target_nnz_per_row"]
                        ),
                    ),
                    primary=primary,
                    spgemm_rhs=rhs,
                    size=size,
                    requested_density=(
                        None
                        if density_spec is None
                        else density_spec["requested_density"]
                    ),
                    effective_density=float(effective_density),
                    density_mode=(
                        None if density_spec is None else density_spec["density_mode"]
                    ),
                    target_nnz_per_row=(
                        None
                        if density_spec is None
                        else density_spec["target_nnz_per_row"]
                    ),
                    max_density=(
                        None if density_spec is None else density_spec["max_density"]
                    ),
                )
            )
    return cases


def _case_label(
    family: str,
    size: int,
    *,
    requested_density: float | None = None,
    effective_density: float | None = None,
    target_nnz_per_row: float | None = None,
    short_row_nnz: int | None = None,
    target_output_density: float | None = None,
    requested_output_density: float | None = None,
    target_output_nnz_per_row: float | None = None,
) -> str:
    parts = [family, f"n{size}"]
    if short_row_nnz is not None:
        parts.append(f"r{short_row_nnz}")
    if target_nnz_per_row is not None:
        parts.append(f"rnnz{target_nnz_per_row:g}")
    if requested_density is not None:
        parts.append(f"d{requested_density:g}")
    if (
        requested_density is not None
        and effective_density is not None
        and not math.isclose(requested_density, effective_density)
    ):
        parts.append(f"eff{effective_density:g}")
    if target_output_density is not None:
        if target_output_nnz_per_row is not None:
            density_part = (
                f"ornnz{target_output_nnz_per_row:g}_od{target_output_density:g}"
            )
        else:
            density_part = f"od{target_output_density:g}"
        if requested_output_density is not None and not math.isclose(
            requested_output_density, target_output_density
        ):
            density_part = (
                f"od{requested_output_density:g}_eff{target_output_density:g}"
            )
        parts.append(density_part)
    return "_".join(parts)


def _clamp_density_for_nnz(
    *,
    size: int,
    density: float,
    max_nnz_per_matrix: int,
) -> float:
    if max_nnz_per_matrix <= 0:
        return float(density)
    max_density = max_nnz_per_matrix / float(size * size)
    return float(min(density, max_density))


def density_for_size(
    n: int,
    target_nnz_per_row: float = 16.0,
    max_density: float = DEFAULT_MAX_DENSITY,
    max_nnz_per_matrix: int = 0,
) -> float:
    """Return a square-matrix density that preserves row-local sparse work.

    Fixed density is a poor default for scale sweeps: as ``n`` grows it changes
    each row's expected occupancy and can create huge or unrepresentative
    SpGEMM inputs.  The target-nnz form keeps the average row width stable while
    still clamping small matrices away from dense regimes.
    """

    if n <= 0:
        raise ValueError("n must be positive.")
    density = min(float(max_density), float(target_nnz_per_row) / float(n))
    if max_nnz_per_matrix > 0:
        density = min(density, max_nnz_per_matrix / float(n * n))
    return max(0.0, float(density))


def _density_specs_for_size(
    *,
    size: int,
    densities: list[float] | None,
    target_nnzs_per_row: list[float] | None,
    max_density: float,
    max_nnz_per_matrix: int,
) -> list[dict[str, float | str | None]]:
    if densities is not None:
        return [
            {
                "density_mode": "explicit_density",
                "requested_density": float(density),
                "effective_density": _clamp_density_for_nnz(
                    size=size,
                    density=float(density),
                    max_nnz_per_matrix=max_nnz_per_matrix,
                ),
                "target_nnz_per_row": None,
                "max_density": float(max_density),
            }
            for density in densities
        ]

    targets = (
        list(DEFAULT_TARGET_NNZS_PER_ROW)
        if target_nnzs_per_row is None
        else target_nnzs_per_row
    )
    return [
        {
            "density_mode": "target_nnz_per_row",
            "requested_density": None,
            "effective_density": density_for_size(
                size,
                target_nnz_per_row=target,
                max_density=max_density,
                max_nnz_per_matrix=max_nnz_per_matrix,
            ),
            "target_nnz_per_row": float(target),
            "max_density": float(max_density),
        }
        for target in targets
    ]


def _output_density_specs_for_size(
    *,
    size: int,
    output_densities: list[float] | None,
    output_target_nnzs_per_row: list[float] | None,
    max_density: float,
    max_nnz_per_matrix: int,
) -> list[dict[str, float | str | None]]:
    if output_densities is not None:
        return [
            {
                "density_mode": "explicit_density",
                "requested_density": float(density),
                "effective_density": _clamp_density_for_nnz(
                    size=size,
                    density=float(density),
                    max_nnz_per_matrix=max_nnz_per_matrix,
                ),
                "target_nnz_per_row": None,
            }
            for density in output_densities
        ]

    targets = (
        list(DEFAULT_OUTPUT_TARGET_NNZS_PER_ROW)
        if output_target_nnzs_per_row is None
        else output_target_nnzs_per_row
    )
    return [
        {
            "density_mode": "target_nnz_per_row",
            "requested_density": None,
            "effective_density": density_for_size(
                size,
                target_nnz_per_row=target,
                max_density=max_density,
                max_nnz_per_matrix=max_nnz_per_matrix,
            ),
            "target_nnz_per_row": float(target),
        }
        for target in targets
    ]


def primary_matrix(
    family: str,
    *,
    size: int,
    density: float,
    short_row_nnz: int,
    duplicate_factor: int,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    if family == "uniform_short_rows":
        return uniform_short_rows(size, size, short_row_nnz, rng)
    if family == "imbalanced_rows":
        return imbalanced_rows(size, size, density, rng)
    if family == "banded":
        return banded_matrix(size, density)
    if family == "diagonal_dominant":
        return diagonal_dominant_matrix(size, density, rng)
    if family == "duplicate_heavy":
        return duplicate_heavy_csr(size, size, density, duplicate_factor, rng)
    if family == "exact_cancellation":
        lhs, _ = exact_cancellation_pair(size)
        return lhs
    raise ValueError(f"Unknown matrix family {family!r}.")


def spgemm_rhs_matrix(
    family: str,
    *,
    lhs: sp.csr_matrix,
    density: float,
    short_row_nnz: int,
    duplicate_factor: int,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    _, inner = lhs.shape
    out_cols = lhs.shape[0]
    if family == "exact_cancellation":
        _, rhs = exact_cancellation_pair(out_cols)
        return rhs
    if family == "duplicate_heavy":
        return duplicate_heavy_csr(inner, out_cols, density, duplicate_factor, rng)
    if family == "uniform_short_rows":
        return uniform_short_rows(inner, out_cols, short_row_nnz, rng)
    if family == "imbalanced_rows":
        return imbalanced_rows(inner, out_cols, density, rng)
    if family == "banded" and inner == out_cols:
        return banded_matrix(inner, density)
    return diagonal_dominant_matrix(inner, density, rng)


def uniform_short_rows(
    rows: int,
    cols: int,
    entries_per_row: int,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    width = max(1, min(int(entries_per_row), cols))
    indptr = np.arange(rows + 1, dtype=np.int32) * width
    offsets = np.arange(width, dtype=np.int64)
    indices = np.empty(rows * width, dtype=np.int32)
    for row in range(rows):
        cols_for_row = (row * 131 + offsets * 17) % cols
        cols_for_row.sort()
        indices[row * width : (row + 1) * width] = cols_for_row.astype(np.int32)
    data = rng.normal(size=rows * width).astype(np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(rows, cols))


def imbalanced_rows(
    rows: int,
    cols: int,
    density: float = 0.01,
    rng: np.random.Generator | None = None,
) -> sp.csr_matrix:
    if rng is None:
        rng = np.random.default_rng(0)
    hot_rows = max(1, rows // 32)
    cold_width = 1
    target_nnz = max(rows, int(round(rows * cols * max(density, 0.0))))
    cold_budget = max(rows - hot_rows, 0) * cold_width
    hot_budget = max(hot_rows, target_nnz - cold_budget)
    hot_width = max(1, min(cols, int(math.ceil(hot_budget / hot_rows))))
    indptr = [0]
    indices_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for row in range(rows):
        width = hot_width if row < hot_rows else cold_width
        cols_for_row = rng.choice(cols, size=width, replace=False).astype(np.int32)
        cols_for_row.sort()
        indices_parts.append(cols_for_row)
        data_parts.append(rng.normal(size=width).astype(np.float32))
        indptr.append(indptr[-1] + width)
    return sp.csr_matrix(
        (
            np.concatenate(data_parts),
            np.concatenate(indices_parts),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(rows, cols),
    )


def banded_matrix(size: int, density: float = 0.0) -> sp.csr_matrix:
    half_bandwidth = max(1, int(round(max(density, 0.0) * size / 2.0)))
    half_bandwidth = min(half_bandwidth, max(size - 1, 1))
    diagonals: list[np.ndarray] = [
        (2.0 + half_bandwidth) * np.ones(size, dtype=np.float32)
    ]
    offsets = [0]
    for offset in range(1, half_bandwidth + 1):
        width = max(size - offset, 0)
        if width == 0:
            continue
        scale = -1.0 / offset
        diagonals.extend(
            [
                scale * np.ones(width, dtype=np.float32),
                scale * np.ones(width, dtype=np.float32),
            ]
        )
        offsets.extend([-offset, offset])
    return sp.diags(diagonals, offsets, format="csr").astype(np.float32)


def diagonal_dominant_matrix(
    size: int,
    density: float,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    matrix = sp.random(
        size,
        size,
        density=max(min(density, 1.0), 0.0),
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    row_abs = np.asarray(np.abs(matrix).sum(axis=1)).ravel().astype(np.float32)
    matrix.setdiag(row_abs + 1.0)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix.astype(np.float32)


def duplicate_heavy_csr(
    rows: int,
    cols: int,
    density: float,
    duplicate_factor: int,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    unique_density = max(density, 1.0 / max(cols, 1)) / max(duplicate_factor, 1)
    width = max(1, min(int(round(unique_density * cols)), cols))
    dup = max(1, duplicate_factor)
    indptr = [0]
    indices_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for row in range(rows):
        base_cols = rng.choice(cols, size=width, replace=False).astype(np.int32)
        base_cols.sort()
        row_indices = np.repeat(base_cols, dup)
        row_data = rng.normal(size=width * dup).astype(np.float32)
        indices_parts.append(row_indices)
        data_parts.append(row_data)
        indptr.append(indptr[-1] + row_indices.size)
    return sp.csr_matrix(
        (
            np.concatenate(data_parts),
            np.concatenate(indices_parts),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(rows, cols),
    )


def exact_cancellation_pair(size: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    lhs_rows = np.repeat(np.arange(size, dtype=np.int32), 2)
    lhs_cols = np.tile(np.asarray([0, 1], dtype=np.int32), size)
    lhs_data = np.ones(size * 2, dtype=np.float32)
    lhs = sp.csr_matrix((lhs_data, (lhs_rows, lhs_cols)), shape=(size, 2))

    rhs_rows = np.concatenate(
        [
            np.zeros(size, dtype=np.int32),
            np.ones(size, dtype=np.int32),
        ]
    )
    rhs_cols = np.tile(np.arange(size, dtype=np.int32), 2)
    rhs_data = np.concatenate(
        [
            np.ones(size, dtype=np.float32),
            -np.ones(size, dtype=np.float32),
        ]
    )
    rhs = sp.csr_matrix((rhs_data, (rhs_rows, rhs_cols)), shape=(2, size))
    return lhs, rhs


def output_density_sweep_pair(
    size: int,
    target_density: float,
    rng: np.random.Generator,
    max_nnz_per_matrix: int = 0,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    inner = size
    operand_density = min(
        max(math.sqrt(target_density) / max(size**0.25, 1.0), 1 / size), 0.25
    )
    if max_nnz_per_matrix > 0:
        operand_density = min(
            operand_density,
            max_nnz_per_matrix / float(size * inner),
        )
    lhs = sp.random(
        size,
        inner,
        density=operand_density,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    rhs = sp.random(
        inner,
        size,
        density=operand_density,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    lhs.sum_duplicates()
    lhs.sort_indices()
    rhs.sum_duplicates()
    rhs.sort_indices()
    return lhs.astype(np.float32), rhs.astype(np.float32)


def _run_case_suites(
    records: list[dict[str, Any]],
    *,
    case: MatrixCase,
    suites: list[str],
    rhs_cols: int,
    index_dtype,
    warmup: int,
    iters: int,
    verify: bool,
    verify_max_elements: int,
    max_dense_elements: int,
    rng: np.random.Generator,
) -> None:
    if "fromdense" in suites:
        _bench_fromdense(
            records,
            case,
            index_dtype,
            warmup,
            iters,
            verify,
            verify_max_elements,
            max_dense_elements,
        )
    if "compressed" in suites:
        _bench_compressed(
            records, case, index_dtype, warmup, iters, verify, verify_max_elements, rng
        )
    if "coo_conversion" in suites:
        _bench_coo_conversion(
            records, case, index_dtype, warmup, iters, verify, verify_max_elements
        )
    if "spgemm" in suites:
        _bench_spgemm(
            records, case, index_dtype, warmup, iters, verify, verify_max_elements
        )
    if "transpose_products" in suites:
        _bench_transpose_products(
            records,
            case,
            rhs_cols,
            index_dtype,
            warmup,
            iters,
            verify,
            verify_max_elements,
            rng,
        )
    if "coo_csc_dense_products" in suites:
        _bench_coo_csc_dense_products(
            records,
            case,
            rhs_cols,
            index_dtype,
            warmup,
            iters,
            verify,
            verify_max_elements,
            rng,
        )


def _bench_fromdense(
    records: list[dict[str, Any]],
    case: MatrixCase,
    index_dtype,
    warmup: int,
    iters: int,
    verify: bool,
    verify_max_elements: int,
    max_dense_elements: int,
) -> None:
    n_elements = int(case.primary.shape[0] * case.primary.shape[1])
    if n_elements > max_dense_elements:
        expected = case.primary.copy()
        expected.sum_duplicates()
        expected.eliminate_zeros()
        _append_skipped_record(
            records,
            case=case,
            suite="fromdense",
            operation="fromdense",
            input_format="dense",
            output_format="csr",
            matrix_meta=_matrix_meta_from_scipy(expected, "dense"),
            rhs_meta=None,
            reason=(
                f"dense materialization would require {n_elements} elements, "
                f"above --max-dense-elements={max_dense_elements}"
            ),
        )
        return

    dense_np = case.primary.toarray().astype(np.float32, copy=False)
    dense = mx.array(dense_np, dtype=mx.float32)
    force_eval(dense)
    expected = case.primary.copy()
    expected.sum_duplicates()
    expected.eliminate_zeros()
    _record_operation(
        records,
        case=case,
        suite="fromdense",
        operation="fromdense",
        input_format="dense",
        output_format="csr",
        matrix_meta=_matrix_meta_from_scipy(expected, "dense"),
        fn=lambda: ms.fromdense(dense, index_dtype=index_dtype),
        scipy_fn=lambda: sp.csr_matrix(dense_np),
        expected_sparse=expected,
        warmup=warmup,
        iters=iters,
        verify=verify,
        verify_max_elements=verify_max_elements,
    )


def _bench_compressed(
    records: list[dict[str, Any]],
    case: MatrixCase,
    index_dtype,
    warmup: int,
    iters: int,
    verify: bool,
    verify_max_elements: int,
    rng: np.random.Generator,
) -> None:
    csr_sorted = _to_mx_csr(
        case.primary, index_dtype=index_dtype, sorted_indices=True, canonical=False
    )
    csr_unsorted_sp = shuffle_csr_within_rows(case.primary, rng)
    csr_unsorted = _to_mx_csr(
        csr_unsorted_sp,
        index_dtype=index_dtype,
        sorted_indices=False,
        canonical=False,
    )
    csc_sorted_sp = case.primary.tocsc(copy=True)
    csc_unsorted_sp = shuffle_csc_within_cols(csc_sorted_sp, rng)
    csc_sorted = _to_mx_csc(
        csc_sorted_sp, index_dtype=index_dtype, sorted_indices=True, canonical=False
    )
    csc_unsorted = _to_mx_csc(
        csc_unsorted_sp, index_dtype=index_dtype, sorted_indices=False, canonical=False
    )
    expected = case.primary.copy()
    expected.sum_duplicates()
    expected.sort_indices()

    for fmt, array, expected_sparse, scipy_fn in (
        (
            "csr",
            csr_unsorted,
            expected,
            lambda matrix=csr_unsorted_sp: _scipy_sort_indices(matrix),
        ),
        (
            "csc",
            csc_unsorted,
            expected.tocsc(copy=True),
            lambda matrix=csc_unsorted_sp: _scipy_sort_indices(matrix),
        ),
    ):
        _record_operation(
            records,
            case=case,
            suite="compressed",
            operation=f"{fmt}_sort_indices",
            input_format=fmt,
            output_format=fmt,
            matrix_meta=sparse_matrix_metadata(array),
            fn=lambda array=array: array.sort_indices(),
            scipy_fn=scipy_fn,
            expected_sparse=expected_sparse,
            warmup=warmup,
            iters=iters,
            verify=verify,
            verify_max_elements=verify_max_elements,
        )

    for fmt, array, expected_sparse, scipy_fn in (
        (
            "csr",
            csr_sorted,
            expected,
            lambda matrix=case.primary: _scipy_sum_duplicates(matrix),
        ),
        (
            "csc",
            csc_sorted,
            expected.tocsc(copy=True),
            lambda matrix=csc_sorted_sp: _scipy_sum_duplicates(matrix),
        ),
    ):
        _record_operation(
            records,
            case=case,
            suite="compressed",
            operation=f"{fmt}_sum_duplicates",
            input_format=fmt,
            output_format=fmt,
            matrix_meta=sparse_matrix_metadata(array),
            fn=lambda array=array: array.sum_duplicates(),
            scipy_fn=scipy_fn,
            expected_sparse=expected_sparse,
            warmup=warmup,
            iters=iters,
            verify=verify,
            verify_max_elements=verify_max_elements,
        )


def _bench_coo_conversion(
    records: list[dict[str, Any]],
    case: MatrixCase,
    index_dtype,
    warmup: int,
    iters: int,
    verify: bool,
    verify_max_elements: int,
) -> None:
    coo_sp = case.primary.tocoo(copy=True)
    coo = _to_mx_coo(coo_sp, index_dtype=index_dtype, canonical=False)
    _record_operation(
        records,
        case=case,
        suite="coo_conversion",
        operation="coo_tocsr",
        input_format="coo",
        output_format="csr",
        matrix_meta=sparse_matrix_metadata(coo),
        fn=lambda: coo.tocsr(canonical=False),
        scipy_fn=lambda: coo_sp.tocsr(copy=True),
        expected_sparse=case.primary,
        warmup=warmup,
        iters=iters,
        verify=verify,
        verify_max_elements=verify_max_elements,
    )
    _record_operation(
        records,
        case=case,
        suite="coo_conversion",
        operation="coo_tocsc",
        input_format="coo",
        output_format="csc",
        matrix_meta=sparse_matrix_metadata(coo),
        fn=lambda: coo.tocsc(canonical=False),
        scipy_fn=lambda: coo_sp.tocsc(copy=True),
        expected_sparse=case.primary.tocsc(copy=True),
        warmup=warmup,
        iters=iters,
        verify=verify,
        verify_max_elements=verify_max_elements,
    )


def _bench_spgemm(
    records: list[dict[str, Any]],
    case: MatrixCase,
    index_dtype,
    warmup: int,
    iters: int,
    verify: bool,
    verify_max_elements: int,
) -> None:
    expected = (case.primary @ case.spgemm_rhs).astype(np.float32)
    expected.sum_duplicates()
    expected.eliminate_zeros()
    for fmt in ("csr", "coo", "csc"):
        lhs = _to_format(case.primary, fmt, index_dtype=index_dtype)
        rhs = _to_format(case.spgemm_rhs, fmt, index_dtype=index_dtype)
        lhs_sp = case.primary.asformat(fmt)
        rhs_sp = case.spgemm_rhs.asformat(fmt)
        _record_operation(
            records,
            case=case,
            suite="spgemm",
            operation=f"{fmt}_matmat",
            input_format=fmt,
            output_format=fmt,
            matrix_meta=sparse_matrix_metadata(lhs),
            rhs_meta=_matrix_meta_from_scipy(case.spgemm_rhs, fmt),
            fn=lambda lhs=lhs, rhs=rhs: lhs @ rhs,
            scipy_fn=lambda lhs_sp=lhs_sp, rhs_sp=rhs_sp: lhs_sp @ rhs_sp,
            expected_sparse=expected.asformat(fmt if fmt != "coo" else "coo"),
            warmup=warmup,
            iters=iters,
            verify=verify,
            verify_max_elements=verify_max_elements,
        )


def _bench_transpose_products(
    records: list[dict[str, Any]],
    case: MatrixCase,
    rhs_cols: int,
    index_dtype,
    warmup: int,
    iters: int,
    verify: bool,
    verify_max_elements: int,
    rng: np.random.Generator,
) -> None:
    csr = _to_mx_csr(
        case.primary, index_dtype=index_dtype, sorted_indices=True, canonical=False
    )
    csc = _to_mx_csc(
        case.primary.tocsc(copy=True),
        index_dtype=index_dtype,
        sorted_indices=True,
        canonical=False,
    )
    rhs_vec_np = rng.normal(size=(case.primary.shape[0],)).astype(np.float32)
    rhs_mat_np = rng.normal(size=(case.primary.shape[0], rhs_cols)).astype(np.float32)
    rhs_vec = mx.array(rhs_vec_np, dtype=mx.float32)
    rhs_mat = mx.array(rhs_mat_np, dtype=mx.float32)
    force_eval(rhs_vec)
    force_eval(rhs_mat)

    expected_vec = case.primary.T @ rhs_vec_np
    expected_mat = case.primary.T @ rhs_mat_np
    for fmt, array, matvec_fn, matmul_fn in (
        (
            "csr",
            csr,
            lambda array=csr: _native.csr_matvec_transpose(
                array.data, array.indices, array.indptr, rhs_vec, array.shape
            ),
            lambda array=csr: _native.csr_matmul_transpose(
                array.data, array.indices, array.indptr, rhs_mat, array.shape
            ),
        ),
        (
            "csc",
            csc,
            lambda array=csc: _native.csc_matvec_transpose(
                array.data, array.indices, array.indptr, rhs_vec, array.shape
            ),
            lambda array=csc: _native.csc_matmul_transpose(
                array.data, array.indices, array.indptr, rhs_mat, array.shape
            ),
        ),
    ):
        _record_operation(
            records,
            case=case,
            suite="transpose_products",
            operation=f"{fmt}_transpose_matvec",
            input_format=fmt,
            output_format="dense",
            matrix_meta=sparse_matrix_metadata(array),
            rhs_meta={"shape": [int(rhs_vec.shape[0])], "dtype": "float32"},
            fn=matvec_fn,
            scipy_fn=lambda matrix=case.primary, rhs=rhs_vec_np: matrix.T @ rhs,
            expected_dense=np.asarray(expected_vec, dtype=np.float32),
            warmup=warmup,
            iters=iters,
            verify=verify,
            verify_max_elements=verify_max_elements,
        )
        _record_operation(
            records,
            case=case,
            suite="transpose_products",
            operation=f"{fmt}_transpose_matmul",
            input_format=fmt,
            output_format="dense",
            matrix_meta=sparse_matrix_metadata(array),
            rhs_meta={
                "shape": [int(rhs_mat.shape[0]), int(rhs_mat.shape[1])],
                "dtype": "float32",
            },
            fn=matmul_fn,
            scipy_fn=lambda matrix=case.primary, rhs=rhs_mat_np: matrix.T @ rhs,
            expected_dense=np.asarray(expected_mat, dtype=np.float32),
            warmup=warmup,
            iters=iters,
            verify=verify,
            verify_max_elements=verify_max_elements,
        )


def _bench_coo_csc_dense_products(
    records: list[dict[str, Any]],
    case: MatrixCase,
    rhs_cols: int,
    index_dtype,
    warmup: int,
    iters: int,
    verify: bool,
    verify_max_elements: int,
    rng: np.random.Generator,
) -> None:
    coo = _to_mx_coo(
        case.primary.tocoo(copy=True), index_dtype=index_dtype, canonical=False
    )
    csc = _to_mx_csc(
        case.primary.tocsc(copy=True),
        index_dtype=index_dtype,
        sorted_indices=True,
        canonical=False,
    )
    rhs_vec_np = rng.normal(size=(case.primary.shape[1],)).astype(np.float32)
    rhs_mat_np = rng.normal(size=(case.primary.shape[1], rhs_cols)).astype(np.float32)
    rhs_vec = mx.array(rhs_vec_np, dtype=mx.float32)
    rhs_mat = mx.array(rhs_mat_np, dtype=mx.float32)
    force_eval(rhs_vec)
    force_eval(rhs_mat)
    expected_vec = case.primary @ rhs_vec_np
    expected_mat = case.primary @ rhs_mat_np

    for fmt, array in (("coo", coo), ("csc", csc)):
        scipy_matrix = case.primary.asformat(fmt)
        _record_operation(
            records,
            case=case,
            suite="coo_csc_dense_products",
            operation=f"{fmt}_matvec",
            input_format=fmt,
            output_format="dense",
            matrix_meta=sparse_matrix_metadata(array),
            rhs_meta={"shape": [int(rhs_vec.shape[0])], "dtype": "float32"},
            fn=lambda array=array: array @ rhs_vec,
            scipy_fn=lambda matrix=scipy_matrix, rhs=rhs_vec_np: matrix @ rhs,
            expected_dense=np.asarray(expected_vec, dtype=np.float32),
            warmup=warmup,
            iters=iters,
            verify=verify,
            verify_max_elements=verify_max_elements,
        )
        _record_operation(
            records,
            case=case,
            suite="coo_csc_dense_products",
            operation=f"{fmt}_matmul",
            input_format=fmt,
            output_format="dense",
            matrix_meta=sparse_matrix_metadata(array),
            rhs_meta={
                "shape": [int(rhs_mat.shape[0]), int(rhs_mat.shape[1])],
                "dtype": "float32",
            },
            fn=lambda array=array: array @ rhs_mat,
            scipy_fn=lambda matrix=scipy_matrix, rhs=rhs_mat_np: matrix @ rhs,
            expected_dense=np.asarray(expected_mat, dtype=np.float32),
            warmup=warmup,
            iters=iters,
            verify=verify,
            verify_max_elements=verify_max_elements,
        )


def _record_operation(
    records: list[dict[str, Any]],
    *,
    case: MatrixCase,
    suite: str,
    operation: str,
    input_format: str,
    output_format: str,
    matrix_meta: dict[str, Any],
    fn: Callable[[], Any],
    warmup: int,
    iters: int,
    verify: bool,
    verify_max_elements: int,
    rhs_meta: dict[str, Any] | None = None,
    expected_sparse: sp.spmatrix | None = None,
    expected_dense: np.ndarray | None = None,
    scipy_fn: Callable[[], Any] | None = None,
) -> None:
    result = force_eval(fn())
    output_meta = _output_metadata(result)
    verification = _verify_result(
        result,
        expected_sparse=expected_sparse,
        expected_dense=expected_dense,
        enabled=verify,
        max_elements=verify_max_elements,
    )
    timing = time_result(fn, warmup=warmup, iters=iters)
    scipy_record = _time_scipy_reference(
        scipy_fn=scipy_fn,
        native_timing=timing,
        warmup=warmup,
        iters=iters,
    )
    record = _base_record(
        case=case,
        suite=suite,
        operation=operation,
        input_format=input_format,
        output_format=output_format,
        matrix_meta=matrix_meta,
        rhs_meta=rhs_meta,
        output=output_meta,
        verification=verification,
        extra={"timing": timing.as_dict(), "scipy": scipy_record},
    )
    records.append(record)


def _time_scipy_reference(
    *,
    scipy_fn: Callable[[], Any] | None,
    native_timing: BenchmarkTiming,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if scipy_fn is None:
        return {"status": "not_available"}
    result = force_scipy_eval(scipy_fn())
    timing = time_result(
        scipy_fn,
        warmup=warmup,
        iters=iters,
        evaluator=force_scipy_eval,
    )
    native_ms = native_timing.median_ms
    scipy_ms = timing.median_ms
    return {
        "status": "timed",
        "timing": timing.as_dict(),
        "output": _scipy_output_metadata(result),
        "speedup_vs_scipy": scipy_speedup(
            scipy_ms=scipy_ms,
            native_ms=native_ms,
        ),
    }


def _append_skipped_record(
    records: list[dict[str, Any]],
    *,
    case: MatrixCase,
    suite: str,
    operation: str,
    input_format: str,
    output_format: str,
    matrix_meta: dict[str, Any],
    rhs_meta: dict[str, Any] | None,
    reason: str,
) -> None:
    records.append(
        _base_record(
            case=case,
            suite=suite,
            operation=operation,
            input_format=input_format,
            output_format=output_format,
            matrix_meta=matrix_meta,
            rhs_meta=rhs_meta,
            output={"kind": "skipped"},
            verification={"status": "skipped"},
            extra={
                "skip_reason": reason,
                "scipy": {
                    "status": "skipped_with_native",
                    "reason": reason,
                },
            },
        )
    )


def _base_record(
    *,
    case: MatrixCase,
    suite: str,
    operation: str,
    input_format: str,
    output_format: str,
    matrix_meta: dict[str, Any],
    rhs_meta: dict[str, Any] | None,
    output: dict[str, Any],
    verification: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "key": record_key(
            suite=suite,
            operation=operation,
            family=case.label,
            input_format=input_format,
            output_format=output_format,
            shape=matrix_meta["shape"],
            rhs_shape=None if rhs_meta is None else rhs_meta["shape"],
        ),
        "suite": suite,
        "operation": operation,
        "backend": "native",
        "matrix_family": case.family,
        "matrix_label": case.label,
        "size": case.size,
        "requested_density": case.requested_density,
        "effective_density": case.effective_density,
        "density_mode": case.density_mode,
        "target_nnz_per_row": case.target_nnz_per_row,
        "max_density": case.max_density,
        "short_row_nnz": case.short_row_nnz,
        "requested_output_density": case.requested_output_density,
        "target_output_density": case.target_output_density,
        "output_density_mode": case.output_density_mode,
        "target_output_nnz_per_row": case.target_output_nnz_per_row,
        "input_format": input_format,
        "output_format": output_format,
        "matrix": matrix_meta,
        "rhs": rhs_meta,
        "output": output,
        "verification": verification,
    }
    if extra is not None:
        record.update(extra)
    return record


def _verify_result(
    result: Any,
    *,
    expected_sparse: sp.spmatrix | None,
    expected_dense: np.ndarray | None,
    enabled: bool,
    max_elements: int,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "disabled"}
    if expected_sparse is not None:
        if expected_sparse.shape[0] * expected_sparse.shape[1] > max_elements:
            return {"status": "skipped_large_sparse_dense_compare"}
        expected = expected_sparse.toarray()
        actual = _sparse_to_numpy_dense(result)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        return {"status": "checked_sparse_dense", "elements": int(expected.size)}
    if expected_dense is not None:
        if expected_dense.size > max_elements:
            return {"status": "skipped_large_dense_compare"}
        actual = np.asarray(to_numpy(result))
        np.testing.assert_allclose(actual, expected_dense, rtol=1e-5, atol=1e-5)
        return {"status": "checked_dense", "elements": int(expected_dense.size)}
    return {"status": "no_reference"}


def _sparse_to_numpy_dense(result: Any) -> np.ndarray:
    if not isinstance(result, ms.COOArray | ms.CSRArray | ms.CSCArray):
        raise TypeError(f"Expected sparse result, got {type(result).__name__}.")
    return np.asarray(to_numpy(result.todense()))


def _output_metadata(result: Any) -> dict[str, Any]:
    if isinstance(result, ms.COOArray | ms.CSRArray | ms.CSCArray):
        metadata = sparse_matrix_metadata(result)
        return {
            "kind": "sparse",
            "format": metadata["format"],
            "shape": metadata["shape"],
            "nnz": metadata["nnz"],
            "density": metadata["density"],
            "dtype": metadata["dtype"],
            "index_dtype": metadata["index_dtype"],
        }
    if isinstance(result, mx.array):
        return {
            "kind": "dense",
            "shape": [int(dim) for dim in result.shape],
            "dtype": str(result.dtype).replace("mlx.core.", ""),
        }
    return {"kind": type(result).__name__}


def _scipy_output_metadata(result: Any) -> dict[str, Any]:
    if sp.issparse(result):
        n_rows, n_cols = (int(result.shape[0]), int(result.shape[1]))
        nnz = int(result.nnz)
        return {
            "kind": "sparse",
            "format": result.getformat(),
            "shape": [n_rows, n_cols],
            "nnz": nnz,
            "density": float(nnz / (n_rows * n_cols)) if n_rows and n_cols else 0.0,
            "dtype": str(result.dtype),
            "index_dtype": _scipy_index_dtype(result),
        }
    if isinstance(result, np.ndarray | np.generic):
        array = np.asarray(result)
        return {
            "kind": "dense",
            "shape": [int(dim) for dim in array.shape],
            "dtype": str(array.dtype),
        }
    return {"kind": type(result).__name__}


def _scipy_index_dtype(matrix: sp.spmatrix) -> str | None:
    if hasattr(matrix, "indices"):
        return str(matrix.indices.dtype)
    if hasattr(matrix, "row"):
        return str(matrix.row.dtype)
    return None


def _matrix_meta_from_scipy(matrix: sp.spmatrix, fmt: str) -> dict[str, Any]:
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    row_lengths = np.diff(matrix.indptr.astype(np.int64, copy=False))
    col_lengths = np.bincount(
        matrix.indices.astype(np.int64, copy=False),
        minlength=matrix.shape[1],
    )
    nnz = int(matrix.nnz)
    n_rows, n_cols = (int(matrix.shape[0]), int(matrix.shape[1]))
    return {
        "format": fmt,
        "shape": [n_rows, n_cols],
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": nnz,
        "density": float(nnz / (n_rows * n_cols)) if n_rows and n_cols else 0.0,
        "dtype": "float32",
        "index_dtype": "int32",
        "row_lengths": _length_stats(row_lengths),
        "col_lengths": _length_stats(col_lengths),
    }


def _length_stats(lengths: np.ndarray) -> dict[str, Any]:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "empty": 0,
            "nonempty": 0,
        }
    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "std": float(lengths.std()),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "empty": int(np.count_nonzero(lengths == 0)),
        "nonempty": int(np.count_nonzero(lengths != 0)),
    }


def _to_format(matrix: sp.spmatrix, fmt: str, *, index_dtype):
    if fmt == "csr":
        return _to_mx_csr(
            matrix.tocsr(copy=True),
            index_dtype=index_dtype,
            sorted_indices=True,
            canonical=False,
        )
    if fmt == "csc":
        return _to_mx_csc(
            matrix.tocsc(copy=True),
            index_dtype=index_dtype,
            sorted_indices=True,
            canonical=False,
        )
    if fmt == "coo":
        return _to_mx_coo(
            matrix.tocoo(copy=True), index_dtype=index_dtype, canonical=False
        )
    raise ValueError(f"Unknown sparse format {fmt!r}.")


def _to_mx_csr(
    matrix: sp.spmatrix,
    *,
    index_dtype,
    sorted_indices: bool,
    canonical: bool,
) -> ms.CSRArray:
    csr = matrix.tocsr(copy=True)
    index_np_dtype = _np_index_dtype(index_dtype)
    return ms.csr_array(
        (
            mx.array(np.asarray(csr.data, dtype=np.float32), dtype=mx.float32),
            mx.array(np.asarray(csr.indices, dtype=index_np_dtype), dtype=index_dtype),
            mx.array(np.asarray(csr.indptr, dtype=index_np_dtype), dtype=index_dtype),
        ),
        shape=csr.shape,
        sorted_indices=sorted_indices,
        canonical=canonical,
        validate="metadata",
    )


def _to_mx_csc(
    matrix: sp.spmatrix,
    *,
    index_dtype,
    sorted_indices: bool,
    canonical: bool,
) -> ms.CSCArray:
    csc = matrix.tocsc(copy=True)
    index_np_dtype = _np_index_dtype(index_dtype)
    return ms.csc_array(
        (
            mx.array(np.asarray(csc.data, dtype=np.float32), dtype=mx.float32),
            mx.array(np.asarray(csc.indices, dtype=index_np_dtype), dtype=index_dtype),
            mx.array(np.asarray(csc.indptr, dtype=index_np_dtype), dtype=index_dtype),
        ),
        shape=csc.shape,
        sorted_indices=sorted_indices,
        canonical=canonical,
        validate="metadata",
    )


def _to_mx_coo(
    matrix: sp.spmatrix,
    *,
    index_dtype,
    canonical: bool,
) -> ms.COOArray:
    coo = matrix.tocoo(copy=True)
    index_np_dtype = _np_index_dtype(index_dtype)
    return ms.coo_array(
        (
            mx.array(np.asarray(coo.data, dtype=np.float32), dtype=mx.float32),
            (
                mx.array(np.asarray(coo.row, dtype=index_np_dtype), dtype=index_dtype),
                mx.array(np.asarray(coo.col, dtype=index_np_dtype), dtype=index_dtype),
            ),
        ),
        shape=coo.shape,
        canonical=canonical,
        validate="metadata",
    )


def _np_index_dtype(index_dtype):
    if index_dtype == mx.int32:
        return np.int32
    if index_dtype == mx.int64:
        return np.int64
    raise TypeError(f"Unsupported index dtype {index_dtype!r}.")


def _scipy_sort_indices(matrix: sp.spmatrix) -> sp.spmatrix:
    out = matrix.copy()
    out.sort_indices()
    return out


def _scipy_sum_duplicates(matrix: sp.spmatrix) -> sp.spmatrix:
    out = matrix.copy()
    out.sum_duplicates()
    return out


def shuffle_csr_within_rows(
    matrix: sp.spmatrix,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    csr = matrix.tocsr(copy=True)
    data = csr.data.copy()
    indices = csr.indices.copy()
    for row in range(csr.shape[0]):
        start, end = int(csr.indptr[row]), int(csr.indptr[row + 1])
        if end - start <= 1:
            continue
        order = rng.permutation(end - start)
        data[start:end] = data[start:end][order]
        indices[start:end] = indices[start:end][order]
    return sp.csr_matrix((data, indices, csr.indptr.copy()), shape=csr.shape)


def shuffle_csc_within_cols(
    matrix: sp.spmatrix,
    rng: np.random.Generator,
) -> sp.csc_matrix:
    csc = matrix.tocsc(copy=True)
    data = csc.data.copy()
    indices = csc.indices.copy()
    for col in range(csc.shape[1]):
        start, end = int(csc.indptr[col]), int(csc.indptr[col + 1])
        if end - start <= 1:
            continue
        order = rng.permutation(end - start)
        data[start:end] = data[start:end][order]
        indices[start:end] = indices[start:end][order]
    return sp.csc_matrix((data, indices, csc.indptr.copy()), shape=csc.shape)


def record_key(
    *,
    suite: str,
    operation: str,
    family: str,
    input_format: str,
    output_format: str,
    shape: list[int],
    rhs_shape: list[int] | None,
) -> str:
    rhs = "none" if rhs_shape is None else "x".join(str(dim) for dim in rhs_shape)
    matrix_shape = "x".join(str(dim) for dim in shape)
    return "|".join(
        [suite, operation, family, input_format, output_format, matrix_shape, rhs]
    )


def compare_to_baseline(
    *,
    current_records: list[dict[str, Any]],
    baseline_path: Path,
    factor: float,
    absolute_ms: float,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_by_key = {record["key"]: record for record in baseline.get("records", [])}
    comparisons = []
    failures = 0
    for record in current_records:
        if "timing" not in record:
            comparisons.append(
                {
                    "key": record["key"],
                    "status": "skipped_current",
                    "reason": record.get("skip_reason", "missing timing"),
                }
            )
            continue
        base = baseline_by_key.get(record["key"])
        if base is None:
            comparisons.append({"key": record["key"], "status": "missing_baseline"})
            continue
        if "timing" not in base:
            comparisons.append({"key": record["key"], "status": "skipped_baseline"})
            continue
        current_ms = float(record["timing"]["median_ms"])
        baseline_ms = float(base["timing"]["median_ms"])
        limit = max(baseline_ms * factor, baseline_ms + absolute_ms)
        passed = current_ms <= limit
        failures += 0 if passed else 1
        comparisons.append(
            {
                "key": record["key"],
                "status": "pass" if passed else "regression",
                "current_median_ms": current_ms,
                "baseline_median_ms": baseline_ms,
                "limit_ms": limit,
                "ratio": None if baseline_ms == 0.0 else current_ms / baseline_ms,
            }
        )
    return {
        "baseline_report": str(baseline_path),
        "factor": factor,
        "absolute_ms": absolute_ms,
        "failures": failures,
        "records": comparisons,
    }


def _write_report(
    report: dict[str, Any],
    *,
    output: Path | None,
    report_dir: Path | None,
) -> Path | None:
    if output is None and report_dir is None:
        return None
    if output is None:
        assert report_dir is not None
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output = (
            report_dir / f"native_cpu_sparse_ops_{report['run_label']}_{timestamp}.json"
        )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(output)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _format_text(report: dict[str, Any]) -> str:
    runtime = report["runtime"]
    lines = [
        "native CPU sparse-op benchmark",
        (
            "device={device} native={native} metal={metal} accelerate={accel} "
            "workers={workers}"
        ).format(
            device=runtime["selected_mlx_device"],
            native=runtime["native_extension_available"],
            metal=runtime["metal_available"],
            accel=runtime["accelerate_available"],
            workers=runtime["configured_worker_count"],
        ),
        "",
        "| suite | operation | family | format | shape | median ms | SciPy ms | vs SciPy | out nnz | verification |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for record in report["records"]:
        output = record["output"]
        out_nnz = output.get("nnz", "")
        timing = record.get("timing")
        median = "" if timing is None else f"{timing['median_ms']:.4f}"
        scipy = record.get("scipy", {})
        scipy_timing = scipy.get("timing")
        scipy_median = (
            "" if scipy_timing is None else f"{scipy_timing['median_ms']:.4f}"
        )
        speedup = scipy.get("speedup_vs_scipy")
        speedup_text = "" if speedup is None else f"{speedup:.2f}x"
        verification = record["verification"]["status"]
        if "skip_reason" in record:
            verification = f"skipped: {record['skip_reason']}"
        lines.append(
            "| {suite} | {operation} | {family} | {fmt}->{out_fmt} | {shape} | "
            "{median} | {scipy_median} | {speedup} | {out_nnz} | {verification} |".format(
                suite=record["suite"],
                operation=record["operation"],
                family=record["matrix_label"],
                fmt=record["input_format"],
                out_fmt=record["output_format"],
                shape="x".join(str(dim) for dim in record["matrix"]["shape"]),
                median=median,
                scipy_median=scipy_median,
                speedup=speedup_text,
                out_nnz=out_nnz,
                verification=verification,
            )
        )
    return "\n".join(lines)


def _resolve_sizes(args: argparse.Namespace) -> list[int]:
    if args.sizes is not None:
        return list(args.sizes)
    if args.size is not None:
        return [args.size]
    return list(DEFAULT_SIZE_GRID)


def _resolve_densities(args: argparse.Namespace) -> list[float] | None:
    if args.densities is not None:
        return [float(density) for density in args.densities]
    if args.density is not None:
        return [float(args.density)]
    return None


def _resolve_target_nnzs_per_row(args: argparse.Namespace) -> list[float]:
    if args.target_nnzs_per_row is not None:
        return [float(value) for value in args.target_nnzs_per_row]
    if args.target_nnz_per_row is not None:
        return [float(args.target_nnz_per_row)]
    return list(DEFAULT_TARGET_NNZS_PER_ROW)


def _resolve_short_row_nnzs(args: argparse.Namespace) -> list[int]:
    if args.short_row_nnzs is not None:
        return list(args.short_row_nnzs)
    if args.short_row_nnz is not None:
        return [args.short_row_nnz]
    return list(DEFAULT_SHORT_ROW_NNZS)


def _resolve_output_densities(args: argparse.Namespace) -> list[float] | None:
    if args.output_densities is not None:
        return [float(density) for density in args.output_densities]
    return None


def _resolve_output_target_nnzs_per_row(args: argparse.Namespace) -> list[float]:
    if args.output_target_nnzs_per_row is not None:
        return [float(value) for value in args.output_target_nnzs_per_row]
    if args.output_target_nnz_per_row is not None:
        return [float(args.output_target_nnz_per_row)]
    return list(DEFAULT_OUTPUT_TARGET_NNZS_PER_ROW)


def _validate_args(args: argparse.Namespace) -> None:
    sizes = _resolve_sizes(args)
    densities = _resolve_densities(args)
    target_nnzs_per_row = _resolve_target_nnzs_per_row(args)
    short_row_nnzs = _resolve_short_row_nnzs(args)
    output_densities = _resolve_output_densities(args)
    output_target_nnzs_per_row = _resolve_output_target_nnzs_per_row(args)
    if any(size <= 0 for size in sizes):
        raise ValueError("--sizes must contain positive matrix dimensions.")
    if any(size > MAX_BENCHMARK_DIMENSION for size in sizes):
        raise ValueError(
            f"--sizes must not exceed {MAX_BENCHMARK_DIMENSION} dimensions."
        )
    if args.rhs_cols <= 0:
        raise ValueError("--rhs-cols must be positive.")
    if densities is not None and any(
        density < 0.0 or density > 1.0 for density in densities
    ):
        raise ValueError("--densities must be in [0, 1].")
    if any(value <= 0.0 for value in target_nnzs_per_row):
        raise ValueError("--target-nnzs-per-row must contain positive values.")
    if args.max_density <= 0.0 or args.max_density > 1.0:
        raise ValueError("--max-density must be in (0, 1].")
    if any(row_nnz <= 0 for row_nnz in short_row_nnzs):
        raise ValueError("--short-row-nnzs must contain positive values.")
    if args.duplicate_factor <= 0:
        raise ValueError("--duplicate-factor must be positive.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")
    if output_densities is not None and any(
        d <= 0.0 or d > 1.0 for d in output_densities
    ):
        raise ValueError("--output-densities must be in (0, 1].")
    if any(value <= 0.0 for value in output_target_nnzs_per_row):
        raise ValueError("--output-target-nnzs-per-row must contain positive values.")
    if args.max_dense_elements <= 0:
        raise ValueError("--max-dense-elements must be positive.")
    if args.max_nnz_per_matrix < 0:
        raise ValueError("--max-nnz-per-matrix must be non-negative.")
    if args.verify_max_elements < 0:
        raise ValueError("--verify-max-elements must be non-negative.")
    if args.regression_factor < 1.0:
        raise ValueError("--regression-factor must be at least 1.0.")
    if args.regression_absolute_ms < 0.0:
        raise ValueError("--regression-absolute-ms must be non-negative.")


if __name__ == "__main__":
    main()
