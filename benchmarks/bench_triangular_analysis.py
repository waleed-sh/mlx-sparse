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

"""CPU triangular-solve analysis benchmark.

This benchmark is intentionally separate from the production repeated-solve
benchmark. It measures the structural-analysis variants directly so
diagonal-position caching and level scheduling can be accepted or rejected from
evidence rather than from the surrounding factorized-solve path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

import mlx_sparse as ms
import mlx_sparse._native as native
from mlx_sparse._host import to_numpy

try:
    from benchmarks.benchmark_utils import (
        CPU_THREADS_ENV,
        cpu_runtime_metadata,
        time_result,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from benchmarks/
    from benchmark_utils import (  # type: ignore
        CPU_THREADS_ENV,
        cpu_runtime_metadata,
        time_result,
    )


FAMILIES = ("chain", "two_level_blocks", "diagonal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CSR triangular analysis variants."
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 4096])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--nrhs", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--solver-parallel", action="store_true")
    parser.add_argument("--solver-threads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.cpu_threads is not None:
        os.environ[CPU_THREADS_ENV] = str(args.cpu_threads)
    if args.solver_parallel:
        os.environ["MLX_SPARSE_SOLVER_PARALLEL"] = "1"
    if args.solver_threads is not None:
        os.environ["MLX_SPARSE_SOLVER_THREADS"] = str(args.solver_threads)

    runtime = cpu_runtime_metadata(warmup=args.warmup, iters=args.iters)
    runtime["solver_parallel_env"] = os.environ.get("MLX_SPARSE_SOLVER_PARALLEL")
    runtime["solver_threads_env"] = os.environ.get("MLX_SPARSE_SOLVER_THREADS")

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    for n in args.sizes:
        for family in args.families:
            data, indices, indptr = _make_lower_triangular(family, n)
            shape = (n, n)
            positions = native.csr_triangular_diagonal_positions(indices, indptr, shape)
            schedule = native.csr_triangular_level_schedule(
                indices, indptr, shape, lower=True
            )
            mx.eval(data, indices, indptr, positions, *schedule)
            schedule_offsets = np.asarray(to_numpy(schedule[0]), dtype=np.int32)
            schedule_rows = np.asarray(to_numpy(schedule[1]), dtype=np.int32)

            for nrhs in args.nrhs:
                rhs_np = rng.normal(size=(n, nrhs)).astype(np.float32)
                rhs = mx.array(rhs_np[:, 0] if nrhs == 1 else rhs_np)
                mx.eval(rhs)
                baseline = native.csr_triangular_solve(
                    data,
                    indices,
                    indptr,
                    rhs,
                    shape,
                    lower=True,
                    unit_diagonal=False,
                )
                analyzed = native.csr_triangular_solve(
                    data,
                    indices,
                    indptr,
                    rhs,
                    shape,
                    lower=True,
                    unit_diagonal=False,
                    diagonal_positions=positions,
                    level_schedule=schedule,
                )
                mx.eval(baseline, analyzed)
                np.testing.assert_allclose(
                    to_numpy(analyzed), to_numpy(baseline), rtol=0.0, atol=0.0
                )

                baseline_timing = time_result(
                    lambda: native.csr_triangular_solve(
                        data,
                        indices,
                        indptr,
                        rhs,
                        shape,
                        lower=True,
                        unit_diagonal=False,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                )
                analyzed_timing = time_result(
                    lambda: native.csr_triangular_solve(
                        data,
                        indices,
                        indptr,
                        rhs,
                        shape,
                        lower=True,
                        unit_diagonal=False,
                        diagonal_positions=positions,
                        level_schedule=schedule,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                )
                records.append(
                    {
                        "family": family,
                        "n": n,
                        "nnz": int(data.shape[0]),
                        "nrhs": nrhs,
                        "rhs_rank": int(rhs.ndim),
                        "schedule_levels": max(int(schedule_offsets.size) - 1, 0),
                        "schedule_rows": int(schedule_rows.size),
                        "baseline": baseline_timing.as_dict(),
                        "analyzed": analyzed_timing.as_dict(),
                        "analyzed_over_baseline": (
                            analyzed_timing.median_ms / baseline_timing.median_ms
                            if baseline_timing.median_ms > 0.0
                            else float("inf")
                        ),
                    }
                )

    report = {
        "benchmark": "triangular_analysis",
        "version_target": "0.0.4b1",
        "mode": "cpu_only_native_non_accelerate",
        "runtime": runtime,
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "records": records,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json or args.output is not None:
        print(payload)
    else:
        print(_format_text(report))


def _make_lower_triangular(
    family: str,
    n: int,
) -> tuple[mx.array, mx.array, mx.array]:
    data: list[float] = []
    indices: list[int] = []
    indptr: list[int] = [0]
    for row in range(n):
        if family == "chain" and row > 0:
            data.append(-0.25)
            indices.append(row - 1)
        elif family == "two_level_blocks" and row % 2 == 1:
            data.append(-0.25)
            indices.append(row - 1)
        elif family != "diagonal" and family not in {"chain", "two_level_blocks"}:
            raise ValueError(f"unknown triangular family {family!r}.")
        data.append(1.25)
        indices.append(row)
        indptr.append(len(data))
    return (
        mx.array(np.asarray(data, dtype=np.float32)),
        mx.array(np.asarray(indices, dtype=np.int32)),
        mx.array(np.asarray(indptr, dtype=np.int32)),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if any(size <= 0 for size in args.sizes):
        raise ValueError("--sizes must be positive.")
    if any(size > 32768 for size in args.sizes):
        raise ValueError("--sizes must not exceed 32768.")
    if any(nrhs <= 0 for nrhs in args.nrhs):
        raise ValueError("--nrhs values must be positive.")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive.")
    if args.solver_threads is not None and args.solver_threads <= 0:
        raise ValueError("--solver-threads must be positive.")


def _format_text(report: dict[str, Any]) -> str:
    lines = [
        "triangular analysis benchmark",
        "| family | n | nnz | nrhs | levels | baseline ms | analyzed ms | analyzed/base |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in report["records"]:
        lines.append(
            "| {family} | {n} | {nnz} | {nrhs} | {levels} | {base:.4f} | "
            "{analyzed:.4f} | {ratio:.3f}x |".format(
                family=record["family"],
                n=record["n"],
                nnz=record["nnz"],
                nrhs=record["nrhs"],
                levels=record["schedule_levels"],
                base=record["baseline"]["median_ms"],
                analyzed=record["analyzed"]["median_ms"],
                ratio=record["analyzed_over_baseline"],
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
