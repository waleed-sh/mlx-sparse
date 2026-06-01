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

"""Focused Chebyshev setup/apply and PCG validation benchmark."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np
import scipy.sparse as sp

import mlx_sparse as ms
import mlx_sparse._native as _native
from mlx_sparse._host import to_numpy
from mlx_sparse.linalg import preconditioners

try:
    from benchmarks.benchmark_utils import (
        force_eval,
        sparse_matrix_metadata,
        time_result,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from benchmarks/
    from benchmark_utils import (  # type: ignore
        force_eval,
        sparse_matrix_metadata,
        time_result,
    )


def poisson_2d(grid: int) -> sp.csr_array:
    """Return a five-point 2-D Poisson operator on a square grid."""

    main = 4.0 * np.ones(grid, dtype=np.float32)
    off = -1.0 * np.ones(grid - 1, dtype=np.float32)
    T = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    I = sp.eye(grid, format="csr", dtype=np.float32)
    Y = sp.diags([off, off], [-1, 1], shape=(grid, grid), format="csr")
    return (sp.kron(I, T, format="csr") + sp.kron(Y, I, format="csr")).astype(
        np.float32
    )


def anisotropic_diffusion_2d(grid: int, *, ax: float = 0.1, ay: float = 1.25):
    """Return an SPD anisotropic diffusion stencil with a mass shift."""

    diag = (2.0 * ax + 2.0 * ay + 0.35) * np.ones(grid, dtype=np.float32)
    off_x = -ax * np.ones(grid - 1, dtype=np.float32)
    off_y = -ay * np.ones(grid - 1, dtype=np.float32)
    T = sp.diags([off_x, diag, off_x], [-1, 0, 1], format="csr")
    I = sp.eye(grid, format="csr", dtype=np.float32)
    Y = sp.diags([off_y, off_y], [-1, 1], shape=(grid, grid), format="csr")
    return (sp.kron(I, T, format="csr") + sp.kron(Y, I, format="csr")).astype(
        np.float32
    )


def mlx_csr_from_scipy(matrix: sp.csr_array) -> ms.CSRArray:
    """Convert a SciPy CSR matrix to canonical mlx-sparse CSR."""

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


def rel_residual(matrix: sp.csr_array, x: np.ndarray, b: np.ndarray) -> float:
    """Compute the true relative residual on the host."""

    return float(np.linalg.norm(matrix @ x - b) / max(np.linalg.norm(b), 1.0))


def solve_record(
    *,
    label: str,
    fn,
    matrix: sp.csr_array,
    b_np: np.ndarray,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    """Time a native PCG call and record convergence diagnostics."""

    timing = time_result(fn, warmup=warmup, iters=iters, evaluator=force_eval)
    start = time.perf_counter_ns()
    x, info, residual, iterations = fn()
    mx.eval(x, info, residual, iterations)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return {
        "preconditioner": label,
        "info": int(np.asarray(to_numpy(info)).item()),
        "iterations": int(np.asarray(to_numpy(iterations)).item()),
        "residual_norm": float(np.asarray(to_numpy(residual)).item()),
        "true_relative_residual": rel_residual(matrix, to_numpy(x), b_np),
        "solve_timing_ms": timing.as_dict(),
        "single_solve_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=("poisson_2d", "anisotropic_2d"), default="poisson_2d"
    )
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--rhs-cols", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--maxiter", type=int, default=512)
    args = parser.parse_args()

    ms.use_device(args.device)
    scipy_A = (
        poisson_2d(args.grid)
        if args.family == "poisson_2d"
        else anisotropic_diffusion_2d(args.grid)
    )
    A = mlx_csr_from_scipy(scipy_A)
    n = A.shape[0]
    b_np = np.ones(n, dtype=np.float32)
    b = mx.array(b_np, dtype=mx.float32)
    x0 = mx.zeros((n,), dtype=mx.float32)
    rhs_matrix = mx.array(
        np.vstack([np.roll(b_np, col) for col in range(args.rhs_cols)]).T,
        dtype=mx.float32,
    )

    jacobi_setup = time_result(
        lambda: preconditioners.jacobi(A, check=True),
        warmup=args.warmup,
        iters=args.iters,
        evaluator=force_eval,
    )
    chebyshev_setup = time_result(
        lambda: preconditioners.chebyshev(A, degree=args.degree),
        warmup=args.warmup,
        iters=args.iters,
        evaluator=force_eval,
    )
    jacobi = preconditioners.jacobi(A, check=True)
    chebyshev = preconditioners.chebyshev(A, degree=args.degree)

    result = {
        "matrix": sparse_matrix_metadata(A),
        "family": args.family,
        "grid": args.grid,
        "degree": args.degree,
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "rtol": args.rtol,
        "atol": args.atol,
        "maxiter": args.maxiter,
        "preconditioners": {
            "jacobi": {
                "setup_ms": jacobi_setup.as_dict(),
                "apply_vector_ms": time_result(
                    lambda: jacobi(b),
                    warmup=args.warmup,
                    iters=args.iters,
                    evaluator=force_eval,
                ).as_dict(),
                "nnz": jacobi.nnz,
                "setup_info": dict(jacobi.setup_info),
            },
            "chebyshev": {
                "setup_ms": chebyshev_setup.as_dict(),
                "apply_vector_ms": time_result(
                    lambda: chebyshev(b),
                    warmup=args.warmup,
                    iters=args.iters,
                    evaluator=force_eval,
                ).as_dict(),
                "apply_matrix_ms": time_result(
                    lambda: chebyshev(rhs_matrix),
                    warmup=args.warmup,
                    iters=args.iters,
                    evaluator=force_eval,
                ).as_dict(),
                "nnz": chebyshev.nnz,
                "setup_info": dict(chebyshev.setup_info),
            },
        },
        "pcg": [
            solve_record(
                label="none",
                fn=lambda: _native.csr_cg(
                    A.data,
                    A.indices,
                    A.indptr,
                    b,
                    x0,
                    A.shape,
                    rtol=args.rtol,
                    atol=args.atol,
                    maxiter=args.maxiter,
                ),
                matrix=scipy_A,
                b_np=b_np,
                warmup=args.warmup,
                iters=args.iters,
            ),
            solve_record(
                label="jacobi",
                fn=lambda: _native.csr_pcg_jacobi(
                    A.data,
                    A.indices,
                    A.indptr,
                    b,
                    x0,
                    jacobi.inverse_diagonal,
                    A.shape,
                    rtol=args.rtol,
                    atol=args.atol,
                    maxiter=args.maxiter,
                ),
                matrix=scipy_A,
                b_np=b_np,
                warmup=args.warmup,
                iters=args.iters,
            ),
            solve_record(
                label="chebyshev",
                fn=lambda: _native.csr_pcg_chebyshev(
                    A.data,
                    A.indices,
                    A.indptr,
                    b,
                    x0,
                    chebyshev.A.data,
                    chebyshev.A.indices,
                    chebyshev.A.indptr,
                    A.shape,
                    degree=chebyshev.degree,
                    lambda_min=chebyshev.lambda_min,
                    lambda_max=chebyshev.lambda_max,
                    rtol=args.rtol,
                    atol=args.atol,
                    maxiter=args.maxiter,
                ),
                matrix=scipy_A,
                b_np=b_np,
                warmup=args.warmup,
                iters=args.iters,
            ),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
