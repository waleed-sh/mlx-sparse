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

"""Benchmark sparse iterative solvers (CG, GMRES, MINRES) vs dense MLX solve.

Usage examples
--------------
python bench_linalg_solvers.py
python bench_linalg_solvers.py --size 2048 --density 0.005 --device cpu
python bench_linalg_solvers.py --size 512  --density 0.05  --iters 10
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse._host import to_numpy


def bench(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    end = time.perf_counter()
    return 1000.0 * (end - start) / iters


def bench_scipy(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        np.asarray(fn())
    start = time.perf_counter()
    for _ in range(iters):
        np.asarray(fn())
    end = time.perf_counter()
    return 1000.0 * (end - start) / iters


def make_spd_csr(n: int, density: float, rng) -> scipy.sparse.csr_matrix:
    """Random sparse SPD matrix: symmetrize a random matrix + strong diagonal."""
    A = scipy.sparse.random(
        n, n, density=density / 2, format="csr", dtype=np.float32, random_state=rng
    )
    A = (A + A.T).tocsr()
    # Diagonal shift ensures positive definiteness
    shift = float(n) * 0.5 + 1.0
    A += scipy.sparse.eye(n, dtype=np.float32) * shift
    return A.astype(np.float32)


def to_ms_csr(scipy_csr: scipy.sparse.csr_matrix) -> ms.CSRArray:
    return ms.csr_array(
        (
            mx.array(scipy_csr.data),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
        canonical=True,
    )


def rel_error(x_mlx: mx.array, x_ref: np.ndarray) -> float:
    x_np = np.asarray(to_numpy(x_mlx))
    norm = float(np.linalg.norm(x_ref))
    if norm == 0.0:
        return float(np.linalg.norm(x_np - x_ref))
    return float(np.linalg.norm(x_np - x_ref) / norm)


def scipy_cg_solve(matrix, rhs, *, rtol: float, maxiter: int | None) -> np.ndarray:
    x, info = scipy.sparse.linalg.cg(
        matrix,
        rhs,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )
    if info != 0:
        raise RuntimeError(f"SciPy CG did not converge: info={info}")
    return x


def scipy_gmres_solve(matrix, rhs, *, rtol: float, maxiter: int | None) -> np.ndarray:
    x, info = scipy.sparse.linalg.gmres(
        matrix,
        rhs,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )
    if info != 0:
        raise RuntimeError(f"SciPy GMRES did not converge: info={info}")
    return x


def scipy_minres_solve(matrix, rhs, *, rtol: float, maxiter: int | None) -> np.ndarray:
    x, info = scipy.sparse.linalg.minres(
        matrix,
        rhs,
        rtol=rtol,
        maxiter=maxiter,
    )
    if info != 0:
        raise RuntimeError(f"SciPy MINRES did not converge: info={info}")
    return x


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse iterative solvers vs dense MLX linalg.solve"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Matrix dimension n for the n×n system (default: 1024)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.001,
        help="Fraction of non-zeros in the sparse matrix (default: 0.001)",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=None)
    args = parser.parse_args()

    ms.use_device(args.device)
    rng = np.random.default_rng(42)
    n = args.size

    scipy_csr = make_spd_csr(n, args.density, rng)
    b_np = rng.normal(size=(n,)).astype(np.float32)

    A_ms = to_ms_csr(scipy_csr)
    b = mx.array(b_np)
    A_dense = mx.array(scipy_csr.toarray())

    x_ref = scipy.sparse.linalg.spsolve(scipy_csr, b_np)

    x_cg, info_cg = linalg.cg(A_ms, b, rtol=args.rtol, maxiter=args.maxiter)
    mx.eval(x_cg)
    err_cg = rel_error(x_cg, x_ref)

    x_gmres, info_gmres = linalg.gmres(A_ms, b, rtol=args.rtol, maxiter=args.maxiter)
    mx.eval(x_gmres)
    err_gmres = rel_error(x_gmres, x_ref)

    x_minres, info_minres = linalg.minres(A_ms, b, rtol=args.rtol, maxiter=args.maxiter)
    mx.eval(x_minres)
    err_minres = rel_error(x_minres, x_ref)

    cg_ms = bench(
        lambda: linalg.cg(A_ms, b, rtol=args.rtol, maxiter=args.maxiter)[0],
        warmup=args.warmup,
        iters=args.iters,
    )
    scipy_cg_ms = bench_scipy(
        lambda: scipy_cg_solve(
            scipy_csr,
            b_np,
            rtol=args.rtol,
            maxiter=args.maxiter,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    gmres_ms = bench(
        lambda: linalg.gmres(A_ms, b, rtol=args.rtol, maxiter=args.maxiter)[0],
        warmup=args.warmup,
        iters=args.iters,
    )
    scipy_gmres_ms = bench_scipy(
        lambda: scipy_gmres_solve(
            scipy_csr,
            b_np,
            rtol=args.rtol,
            maxiter=args.maxiter,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    minres_ms = bench(
        lambda: linalg.minres(A_ms, b, rtol=args.rtol, maxiter=args.maxiter)[0],
        warmup=args.warmup,
        iters=args.iters,
    )
    scipy_minres_ms = bench_scipy(
        lambda: scipy_minres_solve(
            scipy_csr,
            b_np,
            rtol=args.rtol,
            maxiter=args.maxiter,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )
    dense_ms = bench(
        lambda: mx.linalg.solve(A_dense, b),
        warmup=args.warmup,
        iters=args.iters,
    )

    print(
        {
            "backend": args.device,
            "n": n,
            "nnz": int(scipy_csr.nnz),
            "density": args.density,
            # CG
            "cg_converged": info_cg == 0,
            "cg_rel_error": round(err_cg, 8),
            "cg_ms": round(cg_ms, 4),
            "scipy_cg_ms": round(scipy_cg_ms, 4),
            "cg_speedup_vs_scipy": round(scipy_cg_ms / cg_ms, 2),
            "cg_speedup_vs_dense": round(dense_ms / cg_ms, 2),
            # GMRES
            "gmres_converged": info_gmres == 0,
            "gmres_rel_error": round(err_gmres, 8),
            "gmres_ms": round(gmres_ms, 4),
            "scipy_gmres_ms": round(scipy_gmres_ms, 4),
            "gmres_speedup_vs_scipy": round(scipy_gmres_ms / gmres_ms, 2),
            "gmres_speedup_vs_dense": round(dense_ms / gmres_ms, 2),
            # MINRES
            "minres_converged": info_minres == 0,
            "minres_rel_error": round(err_minres, 8),
            "minres_ms": round(minres_ms, 4),
            "scipy_minres_ms": round(scipy_minres_ms, 4),
            "minres_speedup_vs_scipy": round(scipy_minres_ms / minres_ms, 2),
            "minres_speedup_vs_dense": round(dense_ms / minres_ms, 2),
            # Dense baseline
            "dense_solve_ms": round(dense_ms, 4),
        }
    )


if __name__ == "__main__":
    main()
