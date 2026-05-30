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

"""CPU-only repeated sparse direct-solve benchmark.

This benchmark isolates the solve-only phase after a native factorization has
already been computed.  It is aimed at v0.0.4b1 repeated-solve work, where the
important regression target is avoiding Python column loops for dense matrices
of right-hand sides while keeping rank-1 behavior unchanged.

Example:
    python benchmarks/bench_native_cpu_repeated_solves.py \\
      --sizes 128 512 --nrhs 1 2 4 8 16 32 --cpu-threads 1 --json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse._host import to_numpy

try:
    from benchmarks.bench_native_cpu_direct_solvers import (
        MATRIX_FACTORIES,
        MAX_BENCHMARK_DIMENSION,
        density_for_size,
    )
    from benchmarks.benchmark_utils import (
        CPU_THREADS_ENV,
        BenchmarkTiming,
        cpu_runtime_metadata,
        force_eval,
        force_scipy_eval,
        scipy_speedup,
        sparse_matrix_metadata,
        time_result,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from benchmarks/
    from bench_native_cpu_direct_solvers import (  # type: ignore
        MATRIX_FACTORIES,
        MAX_BENCHMARK_DIMENSION,
        density_for_size,
    )
    from benchmark_utils import (  # type: ignore
        CPU_THREADS_ENV,
        BenchmarkTiming,
        cpu_runtime_metadata,
        force_eval,
        force_scipy_eval,
        scipy_speedup,
        sparse_matrix_metadata,
        time_result,
    )


DEFAULT_NRHS = [1, 2, 4, 8, 16, 32]
DEFAULT_TARGET_NNZS_PER_ROW = [4.0, 16.0]
DEFAULT_MAX_DENSITY = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark native CPU repeated sparse direct solves."
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512])
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(MATRIX_FACTORIES),
        default=["banded_spd", "banded_general"],
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=("cholesky", "lu"),
        default=["cholesky", "lu"],
    )
    parser.add_argument(
        "--target-nnzs-per-row",
        nargs="+",
        type=float,
        default=DEFAULT_TARGET_NNZS_PER_ROW,
    )
    parser.add_argument("--max-density", type=float, default=DEFAULT_MAX_DENSITY)
    parser.add_argument("--nrhs", nargs="+", type=int, default=DEFAULT_NRHS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument(
        "--index-dtype",
        choices=("int32", "int64"),
        default="int32",
    )
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--solver-parallel", action="store_true")
    parser.add_argument("--solver-threads", type=int, default=None)
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
    if not runtime["native_extension_available"]:
        raise RuntimeError("Native repeated-solve benchmarks require mlx_sparse._ext.")

    rng = np.random.default_rng(args.seed)
    index_dtype = mx.int32 if args.index_dtype == "int32" else mx.int64
    records: list[dict[str, Any]] = []

    for n in args.sizes:
        density_specs = [
            {
                "target_nnz_per_row": float(target),
                "effective_density": density_for_size(
                    n,
                    target_nnz_per_row=float(target),
                    max_density=args.max_density,
                ),
            }
            for target in args.target_nnzs_per_row
        ]
        for family in args.families:
            for density_spec in density_specs:
                scipy_matrix = MATRIX_FACTORIES[family](
                    n, float(density_spec["effective_density"]), rng
                )
                matrix = ms.from_scipy(
                    scipy_matrix,
                    format="csr",
                    dtype=mx.float32,
                    index_dtype=index_dtype,
                    canonical=True,
                )
                force_eval(matrix)
                matrix_meta = sparse_matrix_metadata(matrix)
                matrix_meta["target_nnz_per_row"] = density_spec["target_nnz_per_row"]
                matrix_meta["effective_density"] = density_spec["effective_density"]
                matrix_meta["max_density"] = args.max_density

                scipy_lu = spla.splu(scipy_matrix.tocsc(copy=True))
                for solver in args.solvers:
                    if solver == "cholesky" and not family.endswith("_spd"):
                        continue
                    native_solver = linalg.factorized(matrix, method=solver)
                    for nrhs in args.nrhs:
                        rhs_np = rng.normal(size=(n, nrhs)).astype(np.float32)
                        if nrhs == 1:
                            rhs_np_for_solve = rhs_np[:, 0]
                        else:
                            rhs_np_for_solve = rhs_np
                        rhs = mx.array(rhs_np_for_solve, dtype=mx.float32)
                        force_eval(rhs)

                        result = native_solver.solve(rhs)
                        force_eval(result)
                        residual = _relative_residual(
                            scipy_matrix, result, rhs_np_for_solve
                        )

                        timing = time_result(
                            lambda: native_solver.solve(rhs),
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                        scipy_timing = time_result(
                            lambda: scipy_lu.solve(rhs_np_for_solve),
                            warmup=args.warmup,
                            iters=args.iters,
                            evaluator=force_scipy_eval,
                        )
                        _append_record(
                            records,
                            solver=solver,
                            family=family,
                            matrix_meta=matrix_meta,
                            rhs=rhs,
                            nrhs=nrhs,
                            timing=timing,
                            scipy_timing=scipy_timing,
                            residual=residual,
                        )

    report = {
        "benchmark": "native_cpu_repeated_solves",
        "version_target": "0.0.4b1",
        "mode": "cpu_only_native_non_accelerate",
        "runtime": runtime,
        "args": {
            "sizes": args.sizes,
            "families": args.families,
            "solvers": args.solvers,
            "target_nnzs_per_row": args.target_nnzs_per_row,
            "max_density": args.max_density,
            "nrhs": args.nrhs,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "index_dtype": args.index_dtype,
            "cpu_threads": args.cpu_threads,
            "solver_parallel": args.solver_parallel,
            "solver_threads": args.solver_threads,
            "max_dimension": MAX_BENCHMARK_DIMENSION,
        },
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


def _append_record(
    records: list[dict[str, Any]],
    *,
    solver: str,
    family: str,
    matrix_meta: dict[str, Any],
    rhs: mx.array,
    nrhs: int,
    timing: BenchmarkTiming,
    scipy_timing: BenchmarkTiming,
    residual: float,
) -> None:
    records.append(
        {
            "solver": solver,
            "backend": "native",
            "matrix_family": family,
            "phase": "solve_only",
            "matrix": matrix_meta,
            "rhs": {
                "rank": int(rhs.ndim),
                "shape": [int(dim) for dim in rhs.shape],
                "nrhs": int(nrhs),
                "dtype": str(rhs.dtype).replace("mlx.core.", ""),
            },
            "timing": timing.as_dict(),
            "rel_residual": residual,
            "scipy": {
                "status": "timed",
                "solver": "scipy.sparse.linalg.SuperLU.solve",
                "timing": scipy_timing.as_dict(),
                "speedup_vs_scipy": scipy_speedup(
                    scipy_ms=scipy_timing.median_ms,
                    native_ms=timing.median_ms,
                ),
            },
        }
    )


def _relative_residual(
    matrix: sp.csr_matrix,
    solution: mx.array,
    rhs_np: np.ndarray,
) -> float:
    x_np = np.asarray(to_numpy(solution), dtype=np.float64)
    rhs = np.asarray(rhs_np, dtype=np.float64)
    residual = matrix @ x_np - rhs
    return float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1.0))


def _validate_args(args: argparse.Namespace) -> None:
    if any(size <= 0 for size in args.sizes):
        raise ValueError("--sizes must contain positive matrix dimensions.")
    if any(size > MAX_BENCHMARK_DIMENSION for size in args.sizes):
        raise ValueError(
            f"--sizes must not exceed {MAX_BENCHMARK_DIMENSION} dimensions."
        )
    if any(value <= 0.0 for value in args.target_nnzs_per_row):
        raise ValueError("--target-nnzs-per-row must contain positive values.")
    if args.max_density <= 0.0 or args.max_density > 1.0:
        raise ValueError("--max-density must be in (0, 1].")
    if any(nrhs <= 0 for nrhs in args.nrhs):
        raise ValueError("--nrhs values must be positive.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive when provided.")
    if args.solver_threads is not None and args.solver_threads <= 0:
        raise ValueError("--solver-threads must be positive when provided.")


def _format_text(report: dict[str, Any]) -> str:
    runtime = report["runtime"]
    lines = [
        "native CPU repeated sparse direct solve benchmark",
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
        ("solver_parallel={solver_parallel} solver_threads={solver_threads}").format(
            solver_parallel=runtime.get("solver_parallel_env"),
            solver_threads=runtime.get("solver_threads_env"),
        ),
        "",
        "| solver | family | n | nnz | nrhs | rhs rank | median ms | SciPy ms | vs SciPy | residual |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in report["records"]:
        matrix = record["matrix"]
        rhs = record["rhs"]
        timing = record["timing"]
        scipy_timing = record["scipy"]["timing"]
        speedup = record["scipy"]["speedup_vs_scipy"]
        lines.append(
            "| {solver} | {family} | {n} | {nnz} | {nrhs} | {rank} | "
            "{median:.4f} | {scipy:.4f} | {speedup:.2f}x | {residual:.3e} |".format(
                solver=record["solver"],
                family=record["matrix_family"],
                n=matrix["n_rows"],
                nnz=matrix["nnz"],
                nrhs=rhs["nrhs"],
                rank=rhs["rank"],
                median=timing["median_ms"],
                scipy=scipy_timing["median_ms"],
                speedup=speedup,
                residual=record["rel_residual"],
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
