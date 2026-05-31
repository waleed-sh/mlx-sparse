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

"""CPU-only native direct-solver phase benchmark.

This benchmark is the v0.0.4b1 baseline for the native, non-Accelerate sparse
direct solver path.  It pins MLX to CPU, records runtime and matrix metadata,
separates input import/canonicalization from native factorization, and
measures native LU and Cholesky in three solve phases:

* ``import_canonicalize``: import a SciPy CSR matrix as materialized canonical
  mlx-sparse CSR input
* ``factor_only``: create explicit sparse factors and force their buffers
* ``solve_only``: reuse one factorization and force the dense solution
* ``factor_plus_solve``: factor and solve in one measured call

The native factorization phase itself is intentionally reported as one fused
kernel.  The current natural-order Cholesky and LU implementations combine
symbolic structure discovery, numeric row updates, and CSR materialization
inside the host factorization routine.

Example:
    python benchmarks/bench_native_cpu_direct_solvers.py \\
      --sizes 128 512 --target-nnzs-per-row 4 16 --json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse._host import to_numpy

try:
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


MatrixFactory = Callable[[int, float, np.random.Generator], sp.csr_matrix]


def make_banded_spd(n: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    del rng
    half_bandwidth = _half_bandwidth_from_density(n, density)
    weights = [1.0 / offset for offset in range(1, half_bandwidth + 1)]
    diagonals: list[np.ndarray] = [
        np.full(n, 1.0 + 2.0 * sum(weights), dtype=np.float32)
    ]
    offsets = [0]
    for offset, weight in enumerate(weights, start=1):
        values = -weight * np.ones(n - offset, dtype=np.float32)
        diagonals.extend([values, values])
        offsets.extend([-offset, offset])
    return sp.diags(diagonals, offsets, shape=(n, n), format="csr").astype(np.float32)


def make_banded_general(
    n: int, density: float, rng: np.random.Generator
) -> sp.csr_matrix:
    del rng
    half_bandwidth = _half_bandwidth_from_density(n, density)
    off_diag_abs_sum = sum(1.0 / offset for offset in range(1, half_bandwidth + 1))
    diagonals = [np.full(n, 1.0 + 2.0 * off_diag_abs_sum, dtype=np.float32)]
    offsets = [0]
    for offset in range(1, half_bandwidth + 1):
        width = max(n - offset, 0)
        if width == 0:
            continue
        lower = -(1.0 / offset) * np.ones(width, dtype=np.float32)
        upper = (0.75 / offset) * np.ones(width, dtype=np.float32)
        diagonals.extend([lower, upper])
        offsets.extend([-offset, offset])
    return sp.diags(diagonals, offsets, shape=(n, n), format="csr").astype(np.float32)


def make_random_spd(n: int, density: float, rng: np.random.Generator) -> sp.csr_matrix:
    random = sp.random(
        n,
        n,
        density=max(min(density / 2.0, 1.0), 0.0),
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    matrix = (random + random.T).tocsr()
    row_abs = np.asarray(np.abs(matrix).sum(axis=1)).ravel().astype(np.float32)
    matrix.setdiag(row_abs + 1.0)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix.astype(np.float32)


def make_random_general(
    n: int, density: float, rng: np.random.Generator
) -> sp.csr_matrix:
    matrix = sp.random(
        n,
        n,
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


MATRIX_FACTORIES: dict[str, MatrixFactory] = {
    "banded_spd": make_banded_spd,
    "banded_general": make_banded_general,
    "random_spd": make_random_spd,
    "random_general": make_random_general,
}

CHOLESKY_FAMILIES = {"banded_spd", "random_spd"}
MAX_BENCHMARK_DIMENSION = 32_768
DEFAULT_TARGET_NNZS_PER_ROW = [4.0, 16.0]
DEFAULT_MAX_DENSITY = 0.25
DENSITY_SENSITIVE_FAMILIES = set(MATRIX_FACTORIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark native CPU sparse LU and Cholesky phases."
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512])
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(MATRIX_FACTORIES),
        default=["banded_spd", "banded_general"],
    )
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
            "Explicit matrix densities. When omitted, densities are derived "
            "from --target-nnzs-per-row for every matrix family."
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
            "Target nonzeros per row for size sweeps. Each effective density "
            "is min(--max-density, target / n)."
        ),
    )
    parser.add_argument(
        "--max-density",
        type=float,
        default=DEFAULT_MAX_DENSITY,
        help="Upper density clamp for target-nnz-per-row sweeps.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument(
        "--index-dtype",
        choices=("int32", "int64"),
        default="int32",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help=(
            "Set MLX_SPARSE_CPU_THREADS for reportability. v0.0.4b0 native "
            "kernels do not consume it yet."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _configure_threads(args.cpu_threads)
    explicit_densities = _resolve_densities(args)
    target_nnzs_per_row = _resolve_target_nnzs_per_row(args)

    runtime = cpu_runtime_metadata(warmup=args.warmup, iters=args.iters)
    if not runtime["native_extension_available"]:
        raise RuntimeError("Native direct solver benchmarks require mlx_sparse._ext.")

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    index_dtype = mx.int32 if args.index_dtype == "int32" else mx.int64

    for n in args.sizes:
        rhs_np = rng.normal(size=(n,)).astype(np.float32)
        rhs = mx.array(rhs_np, dtype=mx.float32)
        force_eval(rhs)
        density_specs = _density_specs_for_size(
            n=n,
            densities=explicit_densities,
            target_nnzs_per_row=target_nnzs_per_row,
            max_density=args.max_density,
        )
        for family in args.families:
            family_densities = (
                density_specs if family in DENSITY_SENSITIVE_FAMILIES else [None]
            )
            for density_spec in family_densities:
                density = (
                    0.0
                    if density_spec is None
                    else float(density_spec["effective_density"])
                )
                scipy_matrix = MATRIX_FACTORIES[family](n, density, rng)
                import_timing = time_result(
                    lambda: _materialized_mlx_csr(scipy_matrix, index_dtype),
                    warmup=args.warmup,
                    iters=args.iters,
                )
                matrix = _materialized_mlx_csr(scipy_matrix, index_dtype)
                force_eval(matrix)
                matrix_meta = sparse_matrix_metadata(matrix)
                matrix_meta["density_mode"] = (
                    None if density_spec is None else density_spec["density_mode"]
                )
                matrix_meta["requested_density"] = (
                    None if density_spec is None else density_spec["requested_density"]
                )
                matrix_meta["effective_density"] = float(density)
                matrix_meta["target_nnz_per_row"] = (
                    None if density_spec is None else density_spec["target_nnz_per_row"]
                )
                matrix_meta["max_density"] = (
                    None if density_spec is None else density_spec["max_density"]
                )
                _append_input_record(
                    records,
                    family=family,
                    matrix_meta=matrix_meta,
                    timing=import_timing,
                )

                if family in CHOLESKY_FAMILIES:
                    _bench_solver(
                        records,
                        solver="cholesky",
                        family=family,
                        matrix=matrix,
                        scipy_matrix=scipy_matrix,
                        rhs=rhs,
                        rhs_np=rhs_np,
                        matrix_meta=matrix_meta,
                        factor_fn=linalg.sparse_cholesky,
                        solve_fn=lambda factor, b: factor.solve(b),
                        warmup=args.warmup,
                        iters=args.iters,
                    )

                _bench_solver(
                    records,
                    solver="lu",
                    family=family,
                    matrix=matrix,
                    scipy_matrix=scipy_matrix,
                    rhs=rhs,
                    rhs_np=rhs_np,
                    matrix_meta=matrix_meta,
                    factor_fn=linalg.sparse_lu,
                    solve_fn=lambda factor, b: factor.solve(b),
                    warmup=args.warmup,
                    iters=args.iters,
                )

    report = {
        "benchmark": "native_cpu_direct_solvers",
        "version_target": "0.0.4b1",
        "mode": "cpu_only_native_non_accelerate",
        "phase_model": {
            "import_canonicalize": (
                "Timed Python/SciPy-to-mlx-sparse import, canonicalization, "
                "and CSR buffer materialization before native factorization."
            ),
            "factor_only": (
                "Timed native host factorization with factor buffers forced. "
                "The production kernels fuse symbolic structure construction, "
                "numeric factorization, and CSR materialization rather than "
                "exposing benchmark-only internal timers."
            ),
            "solve_only": "Timed reuse of one already materialized factorization.",
            "factor_plus_solve": "Timed native factorization followed by one solve.",
        },
        "runtime": runtime,
        "args": {
            "sizes": args.sizes,
            "families": args.families,
            "density_mode": (
                "explicit_density"
                if explicit_densities is not None
                else "target_nnz_per_row"
            ),
            "densities": explicit_densities,
            "target_nnzs_per_row": target_nnzs_per_row,
            "max_density": args.max_density,
            "max_dimension": MAX_BENCHMARK_DIMENSION,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "index_dtype": args.index_dtype,
            "cpu_threads": args.cpu_threads,
        },
        "records": records,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json or args.output is not None:
        print(payload)
    else:
        print(_format_text(report))


def _bench_solver(
    records: list[dict[str, Any]],
    *,
    solver: str,
    family: str,
    matrix: ms.CSRArray,
    scipy_matrix: sp.csr_matrix,
    rhs: mx.array,
    rhs_np: np.ndarray,
    matrix_meta: dict[str, Any],
    factor_fn: Callable[[ms.CSRArray], Any],
    solve_fn: Callable[[Any, mx.array], mx.array],
    warmup: int,
    iters: int,
) -> None:
    factor = factor_fn(matrix)
    force_eval(factor)
    solution = solve_fn(factor, rhs)
    force_eval(solution)
    residual = _relative_residual(scipy_matrix, solution, rhs_np)
    scipy_records = _scipy_phase_timings(
        solver=solver,
        scipy_matrix=scipy_matrix,
        rhs_np=rhs_np,
        warmup=warmup,
        iters=iters,
    )

    factor_timing = time_result(
        lambda: factor_fn(matrix),
        warmup=warmup,
        iters=iters,
    )
    _append_record(
        records,
        solver=solver,
        family=family,
        phase="factor_only",
        matrix_meta=matrix_meta,
        timing=factor_timing,
        factor_nnz=_factor_nnz(factor),
        scipy=scipy_records["factor_only"],
    )

    solve_timing = time_result(
        lambda: solve_fn(factor, rhs),
        warmup=warmup,
        iters=iters,
    )
    _append_record(
        records,
        solver=solver,
        family=family,
        phase="solve_only",
        matrix_meta=matrix_meta,
        timing=solve_timing,
        rel_residual=residual,
        factor_nnz=_factor_nnz(factor),
        scipy=scipy_records["solve_only"],
    )

    combined_timing = time_result(
        lambda: solve_fn(factor_fn(matrix), rhs),
        warmup=warmup,
        iters=iters,
    )
    _append_record(
        records,
        solver=solver,
        family=family,
        phase="factor_plus_solve",
        matrix_meta=matrix_meta,
        timing=combined_timing,
        rel_residual=residual,
        scipy=scipy_records["factor_plus_solve"],
    )


def _append_record(
    records: list[dict[str, Any]],
    *,
    solver: str,
    family: str,
    phase: str,
    matrix_meta: dict[str, Any],
    timing: BenchmarkTiming,
    rel_residual: float | None = None,
    factor_nnz: dict[str, int] | None = None,
    scipy: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {
        "solver": solver,
        "backend": "native",
        "matrix_family": family,
        "phase": phase,
        "matrix": matrix_meta,
        "timing": timing.as_dict(),
    }
    if rel_residual is not None:
        record["rel_residual"] = rel_residual
    if factor_nnz is not None:
        record["factor_nnz"] = factor_nnz
    if scipy is not None:
        native_ms = timing.median_ms
        scipy_timing = scipy.get("timing")
        if scipy_timing is not None and "median_ms" in scipy_timing:
            scipy = {
                **scipy,
                "speedup_vs_scipy": scipy_speedup(
                    scipy_ms=float(scipy_timing["median_ms"]),
                    native_ms=native_ms,
                ),
            }
        record["scipy"] = scipy
    records.append(record)


def _append_input_record(
    records: list[dict[str, Any]],
    *,
    family: str,
    matrix_meta: dict[str, Any],
    timing: BenchmarkTiming,
) -> None:
    records.append(
        {
            "solver": "csr_input",
            "backend": "native",
            "matrix_family": family,
            "phase": "import_canonicalize",
            "phase_detail": (
                "SciPy CSR to canonical materialized mlx-sparse CSR input."
            ),
            "matrix": matrix_meta,
            "timing": timing.as_dict(),
        }
    )


def _scipy_phase_timings(
    *,
    solver: str,
    scipy_matrix: sp.csr_matrix,
    rhs_np: np.ndarray,
    warmup: int,
    iters: int,
) -> dict[str, dict[str, Any]]:
    if solver != "lu":
        reason = (
            "scipy.sparse.linalg does not provide a built-in sparse Cholesky "
            "factorization; SuperLU timings are recorded on LU records for the "
            "same matrices."
        )
        return {
            phase: {"status": "no_equivalent", "reason": reason}
            for phase in ("factor_only", "solve_only", "factor_plus_solve")
        }

    scipy_csc = scipy_matrix.tocsc(copy=True)
    scipy_factor = spla.splu(scipy_csc)
    scipy_solution = scipy_factor.solve(rhs_np)
    residual = _relative_residual_numpy(scipy_matrix, scipy_solution, rhs_np)

    factor_timing = time_result(
        lambda: spla.splu(scipy_csc),
        warmup=warmup,
        iters=iters,
        evaluator=force_scipy_eval,
    )
    solve_timing = time_result(
        lambda: scipy_factor.solve(rhs_np),
        warmup=warmup,
        iters=iters,
        evaluator=force_scipy_eval,
    )
    combined_timing = time_result(
        lambda: spla.splu(scipy_csc).solve(rhs_np),
        warmup=warmup,
        iters=iters,
        evaluator=force_scipy_eval,
    )
    return {
        "factor_only": {
            "status": "timed",
            "solver": "scipy.sparse.linalg.splu",
            "timing": factor_timing.as_dict(),
        },
        "solve_only": {
            "status": "timed",
            "solver": "scipy.sparse.linalg.SuperLU.solve",
            "timing": solve_timing.as_dict(),
            "rel_residual": residual,
        },
        "factor_plus_solve": {
            "status": "timed",
            "solver": "scipy.sparse.linalg.splu.solve",
            "timing": combined_timing.as_dict(),
            "rel_residual": residual,
        },
    }


def _factor_nnz(factor: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    if hasattr(factor, "L"):
        out["L"] = int(factor.L.nnz)
    if hasattr(factor, "U"):
        out["U"] = int(factor.U.nnz)
    if out:
        out["total"] = sum(out.values())
    return out


def _relative_residual(
    matrix: sp.csr_matrix,
    solution: mx.array,
    rhs_np: np.ndarray,
) -> float:
    x_np = np.asarray(to_numpy(solution), dtype=np.float64)
    residual = matrix @ x_np - rhs_np
    return float(np.linalg.norm(residual) / max(np.linalg.norm(rhs_np), 1.0))


def _relative_residual_numpy(
    matrix: sp.csr_matrix,
    solution: np.ndarray,
    rhs_np: np.ndarray,
) -> float:
    x_np = np.asarray(solution, dtype=np.float64)
    residual = matrix @ x_np - rhs_np
    return float(np.linalg.norm(residual) / max(np.linalg.norm(rhs_np), 1.0))


def _materialized_mlx_csr(
    scipy_matrix: sp.csr_matrix,
    index_dtype: mx.Dtype,
) -> ms.CSRArray:
    matrix = ms.from_scipy(
        scipy_matrix,
        format="csr",
        dtype=mx.float32,
        index_dtype=index_dtype,
        canonical=True,
    )
    force_eval(matrix)
    return matrix


def density_for_size(
    n: int,
    target_nnz_per_row: float = 16.0,
    max_density: float = DEFAULT_MAX_DENSITY,
) -> float:
    """Return a density that keeps average row occupancy stable across sizes."""

    if n <= 0:
        raise ValueError("n must be positive.")
    return max(
        0.0,
        float(min(float(max_density), float(target_nnz_per_row) / float(n))),
    )


def _half_bandwidth_from_density(n: int, density: float) -> int:
    target_row_nnz = max(1.0, float(density) * float(n))
    # A symmetric band with half-width k has about 2k + 1 entries per interior row.
    half_bandwidth = max(1, int(round((target_row_nnz - 1.0) / 2.0)))
    return min(half_bandwidth, max(n - 1, 1))


def _density_specs_for_size(
    *,
    n: int,
    densities: list[float] | None,
    target_nnzs_per_row: list[float],
    max_density: float,
) -> list[dict[str, float | str | None]]:
    if densities is not None:
        return [
            {
                "density_mode": "explicit_density",
                "requested_density": float(density),
                "effective_density": float(density),
                "target_nnz_per_row": None,
                "max_density": float(max_density),
            }
            for density in densities
        ]
    return [
        {
            "density_mode": "target_nnz_per_row",
            "requested_density": None,
            "effective_density": density_for_size(
                n,
                target_nnz_per_row=target,
                max_density=max_density,
            ),
            "target_nnz_per_row": float(target),
            "max_density": float(max_density),
        }
        for target in target_nnzs_per_row
    ]


def _validate_args(args: argparse.Namespace) -> None:
    if any(size <= 0 for size in args.sizes):
        raise ValueError("--sizes must contain positive matrix dimensions.")
    if any(size > MAX_BENCHMARK_DIMENSION for size in args.sizes):
        raise ValueError(
            f"--sizes must not exceed {MAX_BENCHMARK_DIMENSION} dimensions."
        )
    densities = _resolve_densities(args)
    if densities is not None and any(
        density < 0.0 or density > 1.0 for density in densities
    ):
        raise ValueError("--densities must be in [0, 1].")
    if any(value <= 0.0 for value in _resolve_target_nnzs_per_row(args)):
        raise ValueError("--target-nnzs-per-row must contain positive values.")
    if args.max_density <= 0.0 or args.max_density > 1.0:
        raise ValueError("--max-density must be in (0, 1].")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters positive.")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive when provided.")


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


def _configure_threads(cpu_threads: int | None) -> None:
    if cpu_threads is not None:
        os.environ[CPU_THREADS_ENV] = str(cpu_threads)


def _format_text(report: dict[str, Any]) -> str:
    runtime = report["runtime"]
    lines = [
        "native CPU direct-solver benchmark",
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
        "| solver | family | n | nnz | phase | median ms | SciPy ms | vs SciPy | residual |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for record in report["records"]:
        matrix = record["matrix"]
        timing = record["timing"]
        residual = record.get("rel_residual")
        residual_text = "" if residual is None else f"{residual:.3e}"
        scipy = record.get("scipy", {})
        scipy_timing = scipy.get("timing")
        scipy_text = "" if scipy_timing is None else f"{scipy_timing['median_ms']:.4f}"
        speedup = scipy.get("speedup_vs_scipy")
        speedup_text = "" if speedup is None else f"{speedup:.2f}x"
        lines.append(
            "| {solver} | {family} | {n} | {nnz} | {phase} | "
            "{median:.4f} | {scipy} | {speedup} | {residual} |".format(
                solver=record["solver"],
                family=record["matrix_family"],
                n=matrix["n_rows"],
                nnz=matrix["nnz"],
                phase=record["phase"],
                median=timing["median_ms"],
                scipy=scipy_text,
                speedup=speedup_text,
                residual=residual_text,
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
