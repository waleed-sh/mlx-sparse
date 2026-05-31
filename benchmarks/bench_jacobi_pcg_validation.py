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

"""Validation benchmark for native Jacobi-preconditioned CG.

This benchmark is intentionally narrow. It measures the first v0.0.5b0
preconditioner path against SciPy CG on matrix families where Jacobi is expected
to be either exact (scaled diagonal) or a conservative baseline (scaled
Poisson). It records true residuals and setup/solve timing separately so release
notes can avoid iteration-only claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse._host import to_numpy
from mlx_sparse.linalg import preconditioners

try:
    from benchmarks.benchmark_utils import (
        force_eval,
        force_scipy_eval,
        scipy_speedup,
        sparse_matrix_metadata,
        time_result,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from benchmarks/
    from benchmark_utils import (  # type: ignore
        force_eval,
        force_scipy_eval,
        scipy_speedup,
        sparse_matrix_metadata,
        time_result,
    )

DEFAULT_SIZES = [16, 64, 256]
DEFAULT_FAMILIES = ["scaled_diagonal", "scaled_poisson_1d"]
DEFAULT_REPORT_DIR = (
    Path(__file__).resolve().parent / "reports" / "v0.0.5b0" / "jacobi_pcg"
)


def make_scaled_diagonal(
    n: int, *, condition: float = 1.0e8, dtype=np.float32
) -> sp.csr_matrix:
    """Return an SPD diagonal matrix with a controlled diagonal range."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if condition < 1.0:
        raise ValueError("condition must be at least 1.")
    diag = np.geomspace(1.0, condition, n).astype(dtype)
    return sp.diags(diag, offsets=0, shape=(n, n), format="csr", dtype=dtype)


def make_scaled_poisson_1d(
    n: int, *, scale_span: float = 1.0e4, dtype=np.float32
) -> sp.csr_matrix:
    """Return ``D @ T @ D`` for a 1-D Poisson tridiagonal ``T``."""

    if n <= 1:
        raise ValueError("n must be greater than 1.")
    if scale_span < 1.0:
        raise ValueError("scale_span must be at least 1.")
    base = sp.diags(
        [-np.ones(n - 1), 2.5 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
        dtype=dtype,
    )
    scaling = np.geomspace(1.0 / scale_span, scale_span, n).astype(dtype)
    D = sp.diags(scaling, offsets=0, format="csr", dtype=dtype)
    return (D @ base @ D).astype(dtype).tocsr()


def make_family(name: str, n: int) -> sp.csr_matrix:
    if name == "scaled_diagonal":
        return make_scaled_diagonal(n)
    if name == "scaled_poisson_1d":
        return make_scaled_poisson_1d(n)
    raise ValueError(f"unknown family {name!r}.")


def to_mlx_csr(matrix: sp.csr_matrix) -> ms.CSRArray:
    matrix = matrix.astype(np.float32).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return ms.csr_array(
        (
            mx.array(matrix.data, dtype=mx.float32),
            mx.array(matrix.indices.astype(np.int32, copy=False), dtype=mx.int32),
            mx.array(matrix.indptr.astype(np.int32, copy=False), dtype=mx.int32),
        ),
        shape=matrix.shape,
        canonical=True,
    )


def scipy_jacobi_operator(matrix: sp.csr_matrix) -> spla.LinearOperator:
    diag = matrix.diagonal().astype(np.float32, copy=False)
    if np.any(~np.isfinite(diag)) or np.any(diag == 0.0):
        raise ValueError("Jacobi diagonal must be finite and nonzero.")
    inv_diag = 1.0 / diag
    return spla.LinearOperator(
        matrix.shape,
        matvec=lambda x: inv_diag * x,
        dtype=np.float32,
    )


def scipy_cg(
    matrix: sp.csr_matrix,
    b: np.ndarray,
    *,
    M=None,
    rtol: float,
    atol: float,
    maxiter: int,
) -> tuple[np.ndarray, int, int]:
    iterations = 0

    def count_iteration(_xk):
        nonlocal iterations
        iterations += 1

    try:
        x, info = spla.cg(
            matrix,
            b,
            M=M,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            callback=count_iteration,
        )
    except TypeError:
        x, info = spla.cg(
            matrix,
            b,
            M=M,
            tol=rtol,
            maxiter=maxiter,
            callback=count_iteration,
        )
    return np.asarray(x, dtype=np.float32), int(info), int(iterations)


def true_residual(matrix: sp.csr_matrix, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(matrix @ x - b))


def _timed_mlx_solve(
    fn,
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, Any], tuple[Any, int]]:
    timing = time_result(fn, warmup=warmup, iters=iters, evaluator=force_eval)
    result = force_eval(fn())
    return timing.as_dict(), result


def _timed_scipy_solve(
    fn,
    *,
    warmup: int,
    iters: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, int, int]]:
    timing = time_result(fn, warmup=warmup, iters=iters, evaluator=force_scipy_eval)
    result = force_scipy_eval(fn())
    return timing.as_dict(), result


def benchmark_case(
    *,
    family: str,
    n: int,
    warmup: int,
    iters: int,
    rtol: float,
    atol: float,
    maxiter: int,
) -> list[dict[str, Any]]:
    scipy_A = make_family(family, n)
    mlx_A = to_mlx_csr(scipy_A)
    mlx_matrix_metadata = sparse_matrix_metadata(mlx_A)
    x_true = np.sin(np.linspace(0.0, np.pi, n, dtype=np.float32))
    b_np = np.asarray(scipy_A @ x_true, dtype=np.float32)
    b_mx = mx.array(b_np, dtype=mx.float32)
    setup_timing = time_result(
        lambda: preconditioners.jacobi(mlx_A, check=True),
        warmup=warmup,
        iters=iters,
        evaluator=force_eval,
    )
    jacobi = force_eval(preconditioners.jacobi(mlx_A, check=True))

    records: list[dict[str, Any]] = []

    def append_mlx(label: str, M) -> dict[str, Any]:
        timing, (x, info) = _timed_mlx_solve(
            lambda: linalg.cg(
                mlx_A,
                b_mx,
                M=M,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
            ),
            warmup=warmup,
            iters=iters,
        )
        x_np = np.asarray(to_numpy(x), dtype=np.float32)
        residual = true_residual(scipy_A, x_np, b_np)
        record = {
            "family": family,
            "n": int(n),
            "matrix": mlx_matrix_metadata,
            "solver": "mlx_sparse.cg",
            "preconditioner": label,
            "info": int(info),
            "true_residual": residual,
            "relative_true_residual": residual / max(float(np.linalg.norm(b_np)), 1.0),
            "setup_timing_ms": setup_timing.as_dict() if label == "jacobi" else None,
            "solve_timing_ms": timing,
            "preconditioner_nnz": jacobi.nnz if label == "jacobi" else None,
            "preconditioner_setup_info": (
                dict(jacobi.setup_info) if label == "jacobi" else None
            ),
        }
        records.append(record)
        return record

    append_mlx("none", None)
    append_mlx("identity", preconditioners.identity(mlx_A))
    jacobi_record = append_mlx("jacobi", jacobi)

    for label, scipy_M in (
        ("none", None),
        ("jacobi", scipy_jacobi_operator(scipy_A)),
    ):
        timing, (x, info, iterations) = _timed_scipy_solve(
            lambda M=scipy_M: scipy_cg(
                scipy_A,
                b_np,
                M=M,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
            ),
            warmup=warmup,
            iters=iters,
        )
        residual = true_residual(scipy_A, x, b_np)
        records.append(
            {
                "family": family,
                "n": int(n),
                "matrix": {
                    "format": "csr",
                    "shape": [int(scipy_A.shape[0]), int(scipy_A.shape[1])],
                    "nnz": int(scipy_A.nnz),
                    "dtype": str(scipy_A.dtype),
                    "index_dtype": str(scipy_A.indices.dtype),
                },
                "solver": "scipy.cg",
                "preconditioner": label,
                "info": int(info),
                "iterations": int(iterations),
                "true_residual": residual,
                "relative_true_residual": residual
                / max(float(np.linalg.norm(b_np)), 1.0),
                "setup_timing_ms": None,
                "solve_timing_ms": timing,
                "preconditioner_nnz": (
                    int(scipy_A.shape[0]) if label == "jacobi" else None
                ),
                "native_vs_scipy_solve_speedup": (
                    scipy_speedup(
                        scipy_ms=timing["median_ms"],
                        native_ms=jacobi_record["solve_timing_ms"]["median_ms"],
                    )
                    if label == "jacobi"
                    else None
                ),
            }
        )

    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    converged = [record for record in records if record["info"] == 0]
    return {
        "record_count": len(records),
        "converged_count": len(converged),
        "max_relative_true_residual": max(
            record["relative_true_residual"] for record in records
        ),
        "families": sorted({record["family"] for record in records}),
        "sizes": sorted({record["n"] for record in records}),
    }


def runtime_metadata(*, device: str, warmup: int, iters: int) -> dict[str, Any]:
    return {
        "native_extension_available": bool(ms.is_available()),
        "selected_mlx_device": str(mx.default_device()),
        "requested_device": device,
        "metal_available": bool(ms.capabilities.METAL),
        "accelerate_available": bool(ms.capabilities.ACCELERATE),
        "warmup_count": int(warmup),
        "iteration_count": int(iters),
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "jacobi_pcg_validation.json"
    summary_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    summary_path.write_text(
        "\n".join(
            [
                "# Jacobi PCG Validation",
                "",
                f"- Records: {summary['record_count']}",
                f"- Converged: {summary['converged_count']}",
                (
                    "- Max relative true residual: "
                    f"{summary['max_relative_true_residual']:.6e}"
                ),
                f"- Families: {', '.join(summary['families'])}",
                f"- Sizes: {', '.join(str(size) for size in summary['sizes'])}",
                "",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--maxiter", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ms.use_device(args.device)
    records: list[dict[str, Any]] = []
    for family in args.families:
        for n in args.sizes:
            records.extend(
                benchmark_case(
                    family=family,
                    n=n,
                    warmup=args.warmup,
                    iters=args.iters,
                    rtol=args.rtol,
                    atol=args.atol,
                    maxiter=args.maxiter,
                )
            )
    report = {
        "metadata": runtime_metadata(
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
        ),
        "parameters": {
            "sizes": [int(size) for size in args.sizes],
            "families": list(args.families),
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "maxiter": int(args.maxiter),
        },
        "records": records,
        "summary": summarize_records(records),
    }
    write_report(report, args.output_dir)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
