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

"""Focused ILU(0) setup/apply and GMRES validation benchmark."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import mlx_sparse as ms
import mlx_sparse._native as _native
from mlx_sparse._host import to_numpy
from mlx_sparse.linalg import preconditioners

try:
    from benchmarks.benchmark_utils import (
        force_eval,
        force_scipy_eval,
        sparse_matrix_metadata,
        time_result,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from benchmarks/
    from benchmark_utils import (  # type: ignore
        force_eval,
        force_scipy_eval,
        sparse_matrix_metadata,
        time_result,
    )


def convection_diffusion_1d(n: int) -> sp.csr_array:
    """Return a nonsymmetric tridiagonal convection-diffusion operator."""

    diffusion = 0.15
    convection = 0.6
    h = 1.0 / (n + 1)
    lower = (-diffusion / h**2 - convection / h) * np.ones(n - 1, dtype=np.float32)
    diag = (2.0 * diffusion / h**2 + convection / h + 1.0) * np.ones(
        n, dtype=np.float32
    )
    upper = (-diffusion / h**2) * np.ones(n - 1, dtype=np.float32)
    return sp.diags(
        [lower, diag, upper], offsets=[-1, 0, 1], format="csr", dtype=np.float32
    )


def mlx_csr_from_scipy(matrix: sp.csr_array) -> ms.CSRArray:
    """Convert a SciPy CSR array to canonical mlx-sparse CSR."""

    matrix = matrix.astype(np.float32).tocsr()
    return ms.csr_array(
        (
            mx.array(matrix.data, dtype=mx.float32),
            mx.array(matrix.indices, dtype=mx.int32),
            mx.array(matrix.indptr, dtype=mx.int32),
        ),
        shape=matrix.shape,
        canonical=True,
    )


def rel_residual(matrix: sp.csr_array, x: np.ndarray, b: np.ndarray) -> float:
    """Compute a true relative residual on the host."""

    residual = matrix @ x - b
    return float(np.linalg.norm(residual) / max(np.linalg.norm(b), 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--rhs-cols", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=2e-6)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--restart", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=128)
    args = parser.parse_args()

    ms.use_device(args.device)
    scipy_A = convection_diffusion_1d(args.size)
    A = mlx_csr_from_scipy(scipy_A)
    b_np = np.cos(np.linspace(0.0, 2.0, args.size)).astype(np.float32)
    b = mx.array(b_np, dtype=mx.float32)
    rhs_matrix = mx.array(
        np.vstack([np.roll(b_np, col) for col in range(args.rhs_cols)]).T,
        dtype=mx.float32,
    )
    x0 = mx.zeros((args.size,), dtype=mx.float32)

    setup_timing = time_result(
        lambda: preconditioners.ilu0(A),
        warmup=args.warmup,
        iters=args.iters,
        evaluator=force_eval,
    )
    M = preconditioners.ilu0(A)
    M_analyzed = preconditioners.ilu0(A, reuse_analysis=True)

    apply_vector = time_result(
        lambda: M(b),
        warmup=args.warmup,
        iters=args.iters,
        evaluator=force_eval,
    )
    apply_matrix = time_result(
        lambda: M(rhs_matrix),
        warmup=args.warmup,
        iters=args.iters,
        evaluator=force_eval,
    )
    analyzed_vector = time_result(
        lambda: M_analyzed(b),
        warmup=args.warmup,
        iters=args.iters,
        evaluator=force_eval,
    )
    analyzed_matrix = time_result(
        lambda: M_analyzed(rhs_matrix),
        warmup=args.warmup,
        iters=args.iters,
        evaluator=force_eval,
    )

    start = time.perf_counter_ns()
    x_base, info_base, residual_base, iterations_base = _native.csr_gmres(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        A.shape,
        rtol=args.rtol,
        atol=args.atol,
        restart=args.restart,
        maxiter=args.maxiter,
    )
    mx.eval(x_base, info_base, residual_base, iterations_base)
    gmres_base_ms = (time.perf_counter_ns() - start) / 1_000_000.0

    start = time.perf_counter_ns()
    x_ilu0, info_ilu0, residual_ilu0, iterations_ilu0 = _native.csr_gmres_ilu0(
        A.data,
        A.indices,
        A.indptr,
        b,
        x0,
        M.L.data,
        M.L.indices,
        M.L.indptr,
        M.U.data,
        M.U.indices,
        M.U.indptr,
        A.shape,
        rtol=args.rtol,
        atol=args.atol,
        restart=args.restart,
        maxiter=args.maxiter,
    )
    mx.eval(x_ilu0, info_ilu0, residual_ilu0, iterations_ilu0)
    gmres_ilu0_ms = (time.perf_counter_ns() - start) / 1_000_000.0

    spilu_start = time.perf_counter_ns()
    spilu = spla.spilu(
        scipy_A.tocsc(),
        drop_tol=0.0,
        fill_factor=1.0,
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
    )
    spilu_setup_ms = (time.perf_counter_ns() - spilu_start) / 1_000_000.0
    scipy_M = spla.LinearOperator(scipy_A.shape, matvec=spilu.solve, dtype=np.float32)
    scipy_start = time.perf_counter_ns()
    try:
        scipy_x, scipy_info = spla.gmres(
            scipy_A,
            b_np,
            M=scipy_M,
            rtol=args.rtol,
            atol=args.atol,
            restart=args.restart,
            maxiter=args.maxiter,
        )
    except TypeError:
        scipy_x, scipy_info = spla.gmres(
            scipy_A,
            b_np,
            M=scipy_M,
            tol=args.rtol,
            restart=args.restart,
            maxiter=args.maxiter,
        )
    force_scipy_eval(scipy_x)
    scipy_gmres_ms = (time.perf_counter_ns() - scipy_start) / 1_000_000.0

    result = {
        "matrix": sparse_matrix_metadata(A),
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "rtol": args.rtol,
        "atol": args.atol,
        "restart": args.restart,
        "maxiter": args.maxiter,
        "ilu0": {
            "setup_ms": setup_timing.as_dict(),
            "apply_vector_ms": apply_vector.as_dict(),
            "apply_matrix_ms": apply_matrix.as_dict(),
            "analyzed_apply_vector_ms": analyzed_vector.as_dict(),
            "analyzed_apply_matrix_ms": analyzed_matrix.as_dict(),
            "nnz_L": M.nnz_L,
            "nnz_U": M.nnz_U,
            "fill_ratio": float(M.nnz / max(A.nnz, 1)),
            "reuse_analysis_default": M.reuse_analysis,
        },
        "gmres": {
            "unpreconditioned": {
                "info": int(np.asarray(to_numpy(info_base)).item()),
                "iterations": int(np.asarray(to_numpy(iterations_base)).item()),
                "residual_norm": float(np.asarray(to_numpy(residual_base)).item()),
                "true_relative_residual": rel_residual(scipy_A, to_numpy(x_base), b_np),
                "time_ms": gmres_base_ms,
            },
            "ilu0": {
                "info": int(np.asarray(to_numpy(info_ilu0)).item()),
                "iterations": int(np.asarray(to_numpy(iterations_ilu0)).item()),
                "residual_norm": float(np.asarray(to_numpy(residual_ilu0)).item()),
                "true_relative_residual": rel_residual(scipy_A, to_numpy(x_ilu0), b_np),
                "time_ms": gmres_ilu0_ms,
            },
            "scipy_spilu": {
                "info": int(scipy_info),
                "setup_ms": spilu_setup_ms,
                "gmres_ms": scipy_gmres_ms,
                "true_relative_residual": rel_residual(scipy_A, scipy_x, b_np),
            },
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
