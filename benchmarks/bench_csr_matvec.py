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

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
import scipy.sparse

import mlx_sparse as ms


def bench(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        y = fn()
        mx.eval(y)

    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        mx.eval(y)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192 * 4)
    parser.add_argument("--cols", type=int, default=8192 * 4)
    parser.add_argument("--density", type=float, default=0.0001)
    parser.add_argument("--complex", action="store_true")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Do not materialize the dense MLX baseline matrix.",
    )
    args = parser.parse_args()

    ms.use_device(args.device)
    rng = np.random.default_rng(0)
    value_dtype = np.complex64 if args.complex else np.float32
    scipy_csr = scipy.sparse.random(
        args.rows,
        args.cols,
        density=args.density,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    scipy_csr.data = scipy_csr.data.astype(value_dtype)
    x_np = rng.normal(size=(args.cols,)).astype(np.float32)
    if args.complex:
        scipy_csr.data += 1j * rng.normal(size=scipy_csr.nnz).astype(np.float32)
        x_np = x_np.astype(np.complex64)
        x_np += 1j * rng.normal(size=(args.cols,)).astype(np.float32)

    csr = ms.csr_array(
        (
            mx.array(scipy_csr.data.astype(value_dtype)),
            mx.array(scipy_csr.indices.astype(np.int32)),
            mx.array(scipy_csr.indptr.astype(np.int32)),
        ),
        shape=scipy_csr.shape,
        sorted_indices=True,
        canonical=True,
    )
    x = mx.array(x_np)
    dense = (
        None if args.skip_dense else mx.array(scipy_csr.toarray().astype(value_dtype))
    )

    sparse_ms = bench(lambda: csr @ x, warmup=args.warmup, iters=args.iters)
    scipy_ms = bench_scipy(
        lambda: scipy_csr @ x_np,
        warmup=args.warmup,
        iters=args.iters,
    )
    dense_ms = (
        None
        if dense is None
        else bench(lambda: dense @ x, warmup=args.warmup, iters=args.iters)
    )

    print(
        {
            "backend": args.device,
            "shape": scipy_csr.shape,
            "nnz": int(scipy_csr.nnz),
            "density": args.density,
            "dtype": str(value_dtype),
            "csr_matvec_ms": sparse_ms,
            "scipy_csr_matvec_ms": scipy_ms,
            "speedup_vs_scipy": scipy_ms / sparse_ms if sparse_ms > 0.0 else None,
            "dense_matvec_ms": dense_ms,
            "effective_nnz_per_s": scipy_csr.nnz / (sparse_ms / 1000.0),
        }
    )


if __name__ == "__main__":
    main()
