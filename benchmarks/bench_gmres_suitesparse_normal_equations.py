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

"""Benchmark GMRES on SuiteSparse least-squares normal-equation fixtures.

This benchmark is a non-regression probe for the GMRES projected solve.  The
``well1033`` and ``illc1033`` fixtures are rectangular Harwell-Boeing systems;
the benchmark forms ``A.T @ A`` and ``A.T @ b`` once, then times restarted
GMRES on the square normal-equation system.  It reports true residuals and
iteration counts instead of relying on timing alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

mx.set_default_device(mx.Device(mx.cpu, 0))

import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg

import mlx_sparse as ms
from benchmarks.benchmark_utils import force_eval, force_scipy_eval, time_result
from mlx_sparse import linalg
from mlx_sparse._host import to_numpy

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "suitesparse" / "HB"
DEFAULT_MATRICES = ("well1033", "illc1033")


def read_design_matrix(name: str) -> scipy.sparse.csr_array:
    """Read a SuiteSparse fixture as canonical float32 CSR."""

    matrix = scipy.io.mmread(FIXTURE_DIR / f"{name}.mtx").astype(np.float32).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return scipy.sparse.csr_array(matrix)


def read_rhs(name: str) -> np.ndarray:
    """Read the matching SuiteSparse right-hand side as a float32 vector."""

    rhs = scipy.io.mmread(FIXTURE_DIR / f"{name}_b.mtx")
    return np.asarray(rhs, dtype=np.float32).reshape(-1)


def normal_equation_case(
    name: str,
) -> tuple[scipy.sparse.csr_array, np.ndarray, dict[str, Any]]:
    """Build ``A.T @ A`` and ``A.T @ b`` for one fixture."""

    design = read_design_matrix(name)
    rhs = read_rhs(name)
    if rhs.shape[0] != design.shape[0]:
        raise ValueError(
            f"{name} RHS length {rhs.shape[0]} does not match rows {design.shape[0]}."
        )
    normal = (design.T @ design).astype(np.float32).tocsr()
    normal.sum_duplicates()
    normal.sort_indices()
    normal_rhs = np.asarray(design.T @ rhs, dtype=np.float32).reshape(-1)
    metadata = {
        "source_matrix": name,
        "source_shape": [int(design.shape[0]), int(design.shape[1])],
        "source_nnz": int(design.nnz),
        "normal_shape": [int(normal.shape[0]), int(normal.shape[1])],
        "normal_nnz": int(normal.nnz),
        "normal_density": float(normal.nnz / (normal.shape[0] * normal.shape[1])),
        "rhs_norm": float(np.linalg.norm(normal_rhs)),
    }
    return scipy.sparse.csr_array(normal), normal_rhs, metadata


def to_mlx_csr(matrix: scipy.sparse.csr_array, *, device: str = "cpu") -> ms.CSRArray:
    """Convert SciPy CSR to canonical mlx-sparse CSR."""

    if device == "cpu":
        mx.set_default_device(mx.Device(mx.cpu, 0))
        ms.use_cpu(require_available=False)
    elif device == "gpu":
        ms.use_device("gpu")
    else:
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}.")
    data = mx.asarray(np.asarray(matrix.data, dtype=np.float32), dtype=mx.float32)
    indices = mx.asarray(
        np.asarray(matrix.indices, dtype=np.int32, order="C"), dtype=mx.int32
    )
    indptr = mx.asarray(
        np.asarray(matrix.indptr, dtype=np.int32, order="C"), dtype=mx.int32
    )
    mx.eval(data, indices, indptr)
    return ms.csr_array(
        (
            data,
            indices,
            indptr,
        ),
        shape=matrix.shape,
        sorted_indices=True,
        canonical=True,
    )


def relative_residual(
    matrix: scipy.sparse.csr_array, x: np.ndarray, rhs: np.ndarray
) -> float:
    """Return the relative true residual of ``matrix @ x = rhs``."""

    return float(np.linalg.norm(matrix @ x - rhs) / max(np.linalg.norm(rhs), 1.0))


def solve_mlx_gmres(
    matrix: ms.CSRArray,
    rhs: mx.array,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
) -> tuple[mx.array, Any]:
    """Run mlx-sparse GMRES with structured diagnostics enabled."""

    return linalg.gmres(
        matrix,
        rhs,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        return_info=True,
    )


def solve_scipy_gmres(
    matrix: scipy.sparse.csr_array,
    rhs: np.ndarray,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
) -> tuple[np.ndarray, int]:
    """Run SciPy GMRES with the same tolerance and restart settings."""

    return scipy.sparse.linalg.gmres(
        matrix,
        rhs,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
    )


def benchmark_case(
    name: str,
    *,
    device: str,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    warmup: int,
    iters: int,
    include_scipy: bool,
) -> dict[str, Any]:
    """Benchmark one normal-equation fixture and return a JSON-safe record."""

    scipy_normal, normal_rhs, metadata = normal_equation_case(name)
    mlx_normal = to_mlx_csr(scipy_normal, device=device)
    rhs_mx = mx.array(normal_rhs, dtype=mx.float32)

    setup = {
        "solver": "gmres",
        "preconditioner": "none",
        "rtol": float(rtol),
        "atol": float(atol),
        "restart": int(restart),
        "maxiter": int(maxiter),
        "warmup": int(warmup),
        "iters": int(iters),
        "device": str(mx.default_device()),
    }

    start_ns = time.perf_counter_ns()
    x_mlx, info_mlx = solve_mlx_gmres(
        mlx_normal,
        rhs_mx,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
    )
    mx.eval(x_mlx)
    solve_ns = time.perf_counter_ns() - start_ns
    x_np = np.asarray(to_numpy(x_mlx), dtype=np.float64)
    mlx_result = {
        "status": int(info_mlx.status),
        "converged": bool(info_mlx.converged),
        "iterations": int(info_mlx.iterations),
        "reported_residual_norm": float(info_mlx.residual_norm),
        "true_relative_residual": relative_residual(scipy_normal, x_np, normal_rhs),
        "single_solve_ms": round(solve_ns / 1_000_000.0, 6),
        "timing": time_result(
            lambda: linalg.gmres(
                mlx_normal,
                rhs_mx,
                rtol=rtol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
            ),
            warmup=warmup,
            iters=iters,
            evaluator=force_eval,
        ).as_dict(),
    }

    scipy_result: dict[str, Any]
    if include_scipy:
        scipy_x, scipy_info = solve_scipy_gmres(
            scipy_normal,
            normal_rhs,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
        )
        scipy_result = {
            "status": int(scipy_info),
            "converged": bool(scipy_info == 0),
            "true_relative_residual": relative_residual(
                scipy_normal, scipy_x, normal_rhs
            ),
            "timing": time_result(
                lambda: solve_scipy_gmres(
                    scipy_normal,
                    normal_rhs,
                    rtol=rtol,
                    atol=atol,
                    restart=restart,
                    maxiter=maxiter,
                ),
                warmup=warmup,
                iters=iters,
                evaluator=force_scipy_eval,
            ).as_dict(),
        }
    else:
        scipy_result = {"status": "skipped"}

    return {
        "suite": "gmres_suitesparse_normal_equations",
        "matrix": metadata,
        "settings": setup,
        "mlx_sparse": mlx_result,
        "scipy": scipy_result,
    }


def main() -> None:
    """Run the benchmark and print a JSON report."""

    parser = argparse.ArgumentParser(
        description="Benchmark GMRES on SuiteSparse normal-equation fixtures."
    )
    parser.add_argument("--matrices", nargs="+", default=list(DEFAULT_MATRICES))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--restart", type=int, default=64)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--skip-scipy",
        action="store_true",
        help="Skip SciPy GMRES timing and report only mlx-sparse results.",
    )
    args = parser.parse_args()

    mx.default_device()
    if args.device == "cpu":
        mx.set_default_device(mx.Device(mx.cpu, 0))
        ms.use_cpu(require_available=False)
    else:
        ms.use_device(args.device)
    records = [
        benchmark_case(
            name,
            device=args.device,
            rtol=args.rtol,
            atol=args.atol,
            restart=args.restart,
            maxiter=args.maxiter,
            warmup=args.warmup,
            iters=args.iters,
            include_scipy=not args.skip_scipy,
        )
        for name in args.matrices
    ]
    print(json.dumps({"records": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
