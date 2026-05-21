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

"""Benchmark sparse direct factorizations (Cholesky, LU, spsolve) vs dense MLX solve.

The sparse routines factorize once then solve, the dense baseline is
``mx.linalg.solve`` which does an implicit dense LU each call.  The
benchmark times the full factorize+solve path for both.

Usage examples
--------------
python bench_linalg_factorizations.py
python bench_linalg_factorizations.py --size 512 --density 0.05 --device cpu
python bench_linalg_factorizations.py --size 256 --density 0.1  --iters 10
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


def make_spd_csr(n: int, density: float, rng) -> scipy.sparse.csr_matrix:
    """Random sparse SPD matrix for Cholesky and LU."""
    A = scipy.sparse.random(
        n, n, density=density / 2, format="csr", dtype=np.float32, random_state=rng
    )
    A = (A + A.T).tocsr()
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse direct factorizations vs dense MLX linalg.solve"
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
        help="Fraction of non-zeros (default: 0.001)",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    ms.use_device(args.device)
    rng = np.random.default_rng(7)
    n = args.size

    scipy_csr = make_spd_csr(n, args.density, rng)
    b_np = rng.normal(size=(n,)).astype(np.float32)

    A_ms = to_ms_csr(scipy_csr)
    b = mx.array(b_np)
    A_dense = mx.array(scipy_csr.toarray())

    x_ref = scipy.sparse.linalg.spsolve(scipy_csr, b_np)

    x_chol = linalg.sparse_cholesky(A_ms).solve(b)
    mx.eval(x_chol)
    err_chol = rel_error(x_chol, x_ref)

    x_lu = linalg.sparse_lu(A_ms).solve(b)
    mx.eval(x_lu)
    err_lu = rel_error(x_lu, x_ref)

    x_spsolve = linalg.spsolve(A_ms, b)
    mx.eval(x_spsolve)
    err_spsolve = rel_error(x_spsolve, x_ref)

    chol_ms = bench(
        lambda: linalg.sparse_cholesky(A_ms).solve(b),
        warmup=args.warmup,
        iters=args.iters,
    )
    lu_ms = bench(
        lambda: linalg.sparse_lu(A_ms).solve(b),
        warmup=args.warmup,
        iters=args.iters,
    )
    spsolve_ms = bench(
        lambda: linalg.spsolve(A_ms, b),
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
            # Sparse Cholesky
            "cholesky_rel_error": round(err_chol, 8),
            "cholesky_ms": round(chol_ms, 4),
            "cholesky_speedup_vs_dense": round(dense_ms / chol_ms, 2),
            # Sparse LU
            "lu_rel_error": round(err_lu, 8),
            "lu_ms": round(lu_ms, 4),
            "lu_speedup_vs_dense": round(dense_ms / lu_ms, 2),
            # spsolve (LU shorthand)
            "spsolve_rel_error": round(err_spsolve, 8),
            "spsolve_ms": round(spsolve_ms, 4),
            "spsolve_speedup_vs_dense": round(dense_ms / spsolve_ms, 2),
            # Dense baseline
            "dense_solve_ms": round(dense_ms, 4),
        }
    )


if __name__ == "__main__":
    main()
