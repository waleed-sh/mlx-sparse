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

"""README-friendly CPU benchmark: mlx-sparse CG vs SciPy CG.

This benchmark intentionally compares the same iterative solver family on both
sides: conjugate gradients on a sparse symmetric positive-definite system.  It
does not compare against dense MLX, because dense storage changes both the
memory model and the per-iteration cost enough to make the README result less
useful as a solver-to-solver comparison.

The system is a 2-D screened Poisson operator with a deterministic manufactured
solution.  Matrix construction is not timed, the benchmark reports solve time
only, plus accuracy against the known solution.

Example:
    python benchmarks/bench_readme_cg_cpu.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import mlx_sparse as ms
from mlx_sparse import linalg


def screened_poisson_2d(grid: int, shift: float) -> sp.csr_matrix:
    """Return a CSR matrix for ``(-Laplacian + shift * I)`` on a square grid."""

    diag = np.full(grid, 2.0 + 0.5 * shift, dtype=np.float32)
    off_diag = -np.ones(grid - 1, dtype=np.float32)
    one_dim = sp.diags(
        [off_diag, diag, off_diag],
        offsets=[-1, 0, 1],
        shape=(grid, grid),
        format="csr",
    )
    identity = sp.eye(grid, dtype=np.float32, format="csr")
    return (
        sp.kron(identity, one_dim, format="csr")
        + sp.kron(one_dim, identity, format="csr")
    ).astype(np.float32)


def to_mlx_csr(matrix: sp.csr_matrix) -> ms.CSRArray:
    """Create a canonical mlx-sparse CSRArray without timing the conversion."""

    return ms.csr_array(
        (
            mx.array(matrix.data),
            mx.array(matrix.indices.astype(np.int32, copy=False)),
            mx.array(matrix.indptr.astype(np.int32, copy=False)),
        ),
        shape=matrix.shape,
        sorted_indices=True,
        canonical=True,
    )


def to_numpy(array: mx.array) -> np.ndarray:
    mx.eval(array)
    return np.asarray(array)


def relative_norm(x: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float(np.linalg.norm(x - reference))
    return float(np.linalg.norm(x - reference) / denominator)


def median_ms(fn, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(1000.0 * (time.perf_counter() - start))
    return float(statistics.median(samples))


def format_ms(milliseconds: float) -> str:
    if milliseconds < 10.0:
        return f"{milliseconds:.3f} ms"
    return f"{milliseconds:.2f} ms"


def format_comparison(ms_time: float, scipy_time: float) -> str:
    ratio = scipy_time / ms_time
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster"
    return f"{1.0 / ratio:.2f}x slower"


def solve_mlx_sparse(
    matrix: ms.CSRArray,
    rhs: mx.array,
    *,
    rtol: float,
    maxiter: int,
) -> mx.array:
    solution, info = linalg.cg(matrix, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"mlx-sparse CG did not converge: info={info}")
    mx.eval(solution)
    return solution


def solve_scipy(
    matrix: sp.csr_matrix,
    rhs: np.ndarray,
    *,
    rtol: float,
    maxiter: int,
) -> np.ndarray:
    solution, info = spla.cg(matrix, rhs, rtol=rtol, atol=0.0, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"SciPy CG did not converge: info={info}")
    return solution.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only conjugate-gradient benchmark for the README."
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=64,
        help="Grid side length, the system has grid**2 unknowns (default: 64).",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.01,
        help="Positive diagonal shift for the screened Poisson operator.",
    )
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.grid < 2:
        raise ValueError("--grid must be at least 2.")
    if args.shift <= 0.0:
        raise ValueError("--shift must be positive.")
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be positive.")
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("--warmup must be non-negative and --repeat positive.")

    ms.use_cpu(require_available=False)
    if not ms.capabilities.CPU:
        raise RuntimeError("mlx-sparse native CPU kernels are not available.")

    scipy_matrix = screened_poisson_2d(args.grid, args.shift)
    rng = np.random.default_rng(args.seed)
    expected = rng.standard_normal(scipy_matrix.shape[0]).astype(np.float32)
    rhs_np = (scipy_matrix @ expected).astype(np.float32)

    mlx_matrix = to_mlx_csr(scipy_matrix)
    rhs_mx = mx.array(rhs_np)

    mlx_solution = solve_mlx_sparse(
        mlx_matrix,
        rhs_mx,
        rtol=args.rtol,
        maxiter=args.maxiter,
    )
    scipy_solution = solve_scipy(
        scipy_matrix,
        rhs_np,
        rtol=args.rtol,
        maxiter=args.maxiter,
    )

    mlx_ms = median_ms(
        lambda: solve_mlx_sparse(
            mlx_matrix,
            rhs_mx,
            rtol=args.rtol,
            maxiter=args.maxiter,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    scipy_ms = median_ms(
        lambda: solve_scipy(
            scipy_matrix,
            rhs_np,
            rtol=args.rtol,
            maxiter=args.maxiter,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    solution_np = to_numpy(mlx_solution).astype(np.float32, copy=False)
    rel_error = relative_norm(solution_np, expected)
    scipy_delta = relative_norm(solution_np, scipy_solution)
    residual = relative_norm(scipy_matrix @ solution_np, rhs_np)

    print(
        "CPU CG 2-D Poisson "
        f"(n={scipy_matrix.shape[0]:,}, nnz={scipy_matrix.nnz:,}): "
        f"mlx-sparse {format_ms(mlx_ms)} | SciPy {format_ms(scipy_ms)} | "
        f"{format_comparison(mlx_ms, scipy_ms)}"
    )
    print(
        f"Accuracy: rel_error={rel_error:.2e} vs exact | "
        f"delta={scipy_delta:.2e} vs SciPy | residual={residual:.2e}"
    )


if __name__ == "__main__":
    main()
