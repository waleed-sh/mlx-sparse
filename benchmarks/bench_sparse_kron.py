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

"""Benchmark native sparse Kronecker products.

The benchmark uses native ``ms.random`` inputs, then times ``ms.kron`` across
COO/CSR/CSC output formats. SciPy is used only as optional context when it is
installed; it is never required to run the mlx-sparse measurements.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.Device(mx.cpu, 0))

import mlx_sparse as ms
from mlx_sparse._host import to_numpy

try:  # Optional comparison only.
    import scipy.sparse as sp
except Exception:  # pragma: no cover - SciPy is optional for benchmark users.
    sp = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark native ms.kron.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--densities", nargs="+", type=float, default=[0.005, 0.02])
    parser.add_argument("--formats", nargs="+", default=["coo", "csr", "csc"])
    parser.add_argument("--dtype", choices=["float32", "complex64"], default="float32")
    parser.add_argument("--index-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def dtype_from_name(name: str):
    return {"float32": mx.float32, "complex64": mx.complex64}[name]


def index_dtype_from_name(name: str):
    return {"int32": mx.int32, "int64": mx.int64}[name]


def set_device(name: str) -> str:
    if name == "gpu":
        ms.use_gpu()
        return "gpu"
    mx.set_default_device(mx.Device(mx.cpu, 0))
    return "cpu"


def eval_sparse(array) -> None:
    if isinstance(array, ms.COOArray):
        mx.eval(array.data, array.row, array.col)
    else:
        mx.eval(array.data, array.indices, array.indptr)


def make_input(n: int, density: float, fmt: str, dtype, index_dtype, seed: int):
    rng = np.random.default_rng(seed)
    nnz = int(round(float(density) * n * n))
    nnz = max(0, min(nnz, n * n))
    flat = rng.choice(n * n, size=nnz, replace=False)
    row = (flat // n).astype(np.int64 if index_dtype == mx.int64 else np.int32)
    col = (flat % n).astype(np.int64 if index_dtype == mx.int64 else np.int32)
    if dtype == mx.complex64:
        values = (
            rng.random(nnz, dtype=np.float32) + 1j * rng.random(nnz, dtype=np.float32)
        ).astype(np.complex64)
    else:
        values = rng.random(nnz, dtype=np.float32)
    coo = ms.coo_array(
        (
            mx.array(values, dtype=dtype),
            (mx.array(row, dtype=index_dtype), mx.array(col, dtype=index_dtype)),
        ),
        shape=(n, n),
        canonical=False,
    )
    if fmt == "coo":
        return coo
    if fmt == "csr":
        return coo.tocsr(canonical=True)
    if fmt == "csc":
        return coo.tocsc(canonical=True)
    raise ValueError(f"unsupported input format {fmt!r}")


def scipy_kron_ms(lhs, rhs, out_format: str, iters: int, warmup: int):
    if sp is None:
        return None
    lhs_sp = sp.csr_array(to_numpy(lhs.todense()))
    rhs_sp = sp.csr_array(to_numpy(rhs.todense()))
    timings = []
    for i in range(warmup + iters):
        start = time.perf_counter()
        out = sp.kron(lhs_sp, rhs_sp, format=out_format)
        _ = out.nnz
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        if i >= warmup:
            timings.append(elapsed_ms)
    return statistics.median(timings)


def time_mlx_kron(lhs, rhs, out_format: str, iters: int, warmup: int) -> float:
    timings = []
    for i in range(warmup + iters):
        start = time.perf_counter()
        out = ms.kron(lhs, rhs, format=out_format)
        eval_sparse(out)
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        if i >= warmup:
            timings.append(elapsed_ms)
    return statistics.median(timings)


def main() -> None:
    args = parse_args()
    device = set_device(args.device)
    dtype = dtype_from_name(args.dtype)
    index_dtype = index_dtype_from_name(args.index_dtype)
    rows = []

    for n in args.sizes:
        for density in args.densities:
            lhs = make_input(n, density, "csr", dtype, index_dtype, args.seed + n)
            rhs = make_input(n, density, "csc", dtype, index_dtype, args.seed + 17 + n)
            eval_sparse(lhs)
            eval_sparse(rhs)
            for out_format in args.formats:
                mlx_ms = time_mlx_kron(lhs, rhs, out_format, args.iters, args.warmup)
                scipy_ms = (
                    scipy_kron_ms(lhs, rhs, out_format, args.iters, args.warmup)
                    if device == "cpu"
                    else None
                )
                out = ms.kron(lhs, rhs, format=out_format)
                eval_sparse(out)
                rows.append(
                    {
                        "operation": "kron",
                        "n": n,
                        "density": density,
                        "lhs_format": "csr",
                        "rhs_format": "csc",
                        "out_format": out_format,
                        "dtype": args.dtype,
                        "index_dtype": args.index_dtype,
                        "device": device,
                        "lhs_nnz": lhs.nnz,
                        "rhs_nnz": rhs.nnz,
                        "out_nnz": out.nnz,
                        "mlx_median_ms": mlx_ms,
                        "scipy_median_ms": scipy_ms,
                    }
                )

    for row in rows:
        scipy_part = (
            " scipy=NA"
            if row["scipy_median_ms"] is None
            else f" scipy={row['scipy_median_ms']:.4f}ms"
        )
        print(
            "kron"
            f" n={row['n']}"
            f" density={row['density']}"
            f" out={row['out_format']}"
            f" dtype={row['dtype']}"
            f" index={row['index_dtype']}"
            f" device={row['device']}"
            f" nnz={row['lhs_nnz']}x{row['rhs_nnz']}->{row['out_nnz']}"
            f" mlx={row['mlx_median_ms']:.4f}ms"
            f"{scipy_part}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
