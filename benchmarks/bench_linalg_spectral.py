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

"""Benchmark sparse spectral routines (eigsh, eigs, svds) vs dense MLX decompositions.

Sparse routines compute k extreme eigenpairs/singular triplets, the dense
baselines compute all n values.  The speedup ratio is meaningful when k << n.

Usage examples
--------------
python bench_linalg_spectral.py
python bench_linalg_spectral.py --size 1024 --k 10 --density 0.01 --device cpu
python bench_linalg_spectral.py --size 512  --k 6  --density 0.02 --iters 10
python bench_linalg_spectral.py --rows 1024 --cols 512 --k 8   # rectangular for svds
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
        _force_numpy(fn())
    start = time.perf_counter()
    for _ in range(iters):
        _force_numpy(fn())
    end = time.perf_counter()
    return 1000.0 * (end - start) / iters


def _force_numpy(result):
    if isinstance(result, tuple | list):
        for value in result:
            _force_numpy(value)
    else:
        np.asarray(result)
    return result


def make_sym_csr(n: int, density: float, rng) -> scipy.sparse.csr_matrix:
    """Random sparse symmetric positive semi-definite matrix."""
    A = scipy.sparse.random(
        n, n, density=density / 2, format="csr", dtype=np.float32, random_state=rng
    )
    A = (A + A.T).tocsr()
    # Mild diagonal shift — keeps eigenvalues positive but spread
    A += scipy.sparse.eye(n, dtype=np.float32) * 2.0
    return A.astype(np.float32)


def make_rect_csr(rows: int, cols: int, density: float, rng) -> scipy.sparse.csr_matrix:
    """Random rectangular sparse matrix for svds."""
    return scipy.sparse.random(
        rows, cols, density=density, format="csr", dtype=np.float32, random_state=rng
    ).astype(np.float32)


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


def rel_error_top_k(
    sparse_vals: mx.array, dense_all: np.ndarray, k: int, largest: bool = True
) -> float:
    """Compare top-k sparse eigenvalues against sorted dense reference."""
    sp = np.sort(np.abs(np.asarray(to_numpy(sparse_vals))))[::-1]
    dn = np.sort(np.abs(dense_all))[::-1]
    ref = dn[:k]
    norm = float(np.linalg.norm(ref))
    if norm == 0.0:
        return 0.0
    return float(np.linalg.norm(sp - ref) / norm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse spectral routines vs dense MLX decompositions"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Square matrix dimension (sets --rows and --cols)",
    )
    parser.add_argument(
        "--rows", type=int, default=512, help="Matrix rows (default: 512)"
    )
    parser.add_argument(
        "--cols", type=int, default=512, help="Matrix cols (default: 512)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of eigenpairs/singular triplets to compute (default: 6)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.001,
        help="Fraction of non-zeros (default: 0.02)",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if args.size is not None:
        args.rows = args.size
        args.cols = args.size

    ms.use_device(args.device)
    rng = np.random.default_rng(13)
    rows, cols, k = args.rows, args.cols, args.k

    n = rows
    sym_csr = make_sym_csr(n, args.density, rng)
    A_sym_ms = to_ms_csr(sym_csr)
    A_sym_dense = mx.array(sym_csr.toarray())

    rect_csr = make_rect_csr(rows, cols, args.density, rng)
    A_rect_ms = to_ms_csr(rect_csr)
    A_rect_dense = mx.array(rect_csr.toarray())

    eigh_vals_ref = np.sort(np.linalg.eigvalsh(sym_csr.toarray()))[::-1]
    svd_vals_ref = np.linalg.svd(rect_csr.toarray(), compute_uv=False)

    vals_eigsh, _ = linalg.eigsh(A_sym_ms, k=k, which="LM")
    mx.eval(vals_eigsh)
    err_eigsh = rel_error_top_k(vals_eigsh, eigh_vals_ref, k)

    vals_eigs, _ = linalg.eigs(A_sym_ms, k=k, which="LM")
    mx.eval(vals_eigs)
    err_eigs = rel_error_top_k(vals_eigs, eigh_vals_ref, k)

    _, vals_svds, _ = linalg.svds(A_rect_ms, k=k, which="LM")
    mx.eval(vals_svds)
    err_svds = rel_error_top_k(vals_svds, svd_vals_ref, k)

    eigsh_ms = bench(
        lambda: linalg.eigsh(A_sym_ms, k=k, which="LM")[0],
        warmup=args.warmup,
        iters=args.iters,
    )
    eigs_ms = bench(
        lambda: linalg.eigs(A_sym_ms, k=k, which="LM")[0],
        warmup=args.warmup,
        iters=args.iters,
    )
    svds_ms = bench(
        lambda: linalg.svds(A_rect_ms, k=k, which="LM")[1],
        warmup=args.warmup,
        iters=args.iters,
    )
    scipy_eigsh_ms = bench_scipy(
        lambda: scipy.sparse.linalg.eigsh(sym_csr, k=k, which="LM")[0],
        warmup=args.warmup,
        iters=args.iters,
    )
    scipy_eigs_ms = bench_scipy(
        lambda: scipy.sparse.linalg.eigs(sym_csr, k=k, which="LM")[0],
        warmup=args.warmup,
        iters=args.iters,
    )
    scipy_svds_ms = bench_scipy(
        lambda: scipy.sparse.linalg.svds(rect_csr, k=k, which="LM")[1],
        warmup=args.warmup,
        iters=args.iters,
    )
    dense_eigh_ms = bench(
        lambda: mx.linalg.eigh(A_sym_dense)[0],
        warmup=args.warmup,
        iters=args.iters,
    )
    dense_svd_ms = bench(
        lambda: mx.linalg.svd(A_rect_dense, compute_uv=False),
        warmup=args.warmup,
        iters=args.iters,
    )

    print(
        {
            "backend": args.device,
            "sym_shape": (n, n),
            "rect_shape": (rows, cols),
            "k": k,
            "sym_nnz": int(sym_csr.nnz),
            "rect_nnz": int(rect_csr.nnz),
            "density": args.density,
            # eigsh
            "eigsh_rel_error": round(err_eigsh, 8),
            "eigsh_ms": round(eigsh_ms, 4),
            "scipy_eigsh_ms": round(scipy_eigsh_ms, 4),
            "eigsh_speedup_vs_scipy": round(scipy_eigsh_ms / eigsh_ms, 2),
            "eigsh_speedup_vs_dense_eigh": round(dense_eigh_ms / eigsh_ms, 2),
            # eigs
            "eigs_rel_error": round(err_eigs, 8),
            "eigs_ms": round(eigs_ms, 4),
            "scipy_eigs_ms": round(scipy_eigs_ms, 4),
            "eigs_speedup_vs_scipy": round(scipy_eigs_ms / eigs_ms, 2),
            "eigs_speedup_vs_dense_eigh": round(dense_eigh_ms / eigs_ms, 2),
            # svds
            "svds_rel_error": round(err_svds, 8),
            "svds_ms": round(svds_ms, 4),
            "scipy_svds_ms": round(scipy_svds_ms, 4),
            "svds_speedup_vs_scipy": round(scipy_svds_ms / svds_ms, 2),
            "svds_speedup_vs_dense_svd": round(dense_svd_ms / svds_ms, 2),
            # Dense baselines (compute all values)
            "dense_eigh_ms": round(dense_eigh_ms, 4),
            "dense_svd_ms": round(dense_svd_ms, 4),
        }
    )


if __name__ == "__main__":
    main()
