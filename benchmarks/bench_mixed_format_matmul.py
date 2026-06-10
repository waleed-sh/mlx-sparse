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

"""Benchmark mixed-format sparse-sparse matmul normalization paths.

The production mixed-format ``A @ B`` path normalizes the right-hand operand
with native format conversion, then dispatches to the left-hand format's native
same-format SpGEMM kernel. This benchmark measures that full path against:

* native RHS conversion alone, and
* the same native SpGEMM with the RHS pre-normalized before timing.

Those numbers are the baseline a future direct mixed-format kernel must beat
before it is worth adding and maintaining.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx_sparse as ms

FORMATS = ("coo", "csr", "csc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark mixed-format sparse-sparse matmul normalization."
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--densities", nargs="+", type=float, default=[0.002, 0.01])
    parser.add_argument("--dtype", choices=["float32", "complex64"], default="float32")
    parser.add_argument("--index-dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def dtype_from_name(name: str):
    return {"float32": mx.float32, "complex64": mx.complex64}[name]


def np_dtype_from_name(name: str):
    return {"float32": np.float32, "complex64": np.complex64}[name]


def index_dtype_from_name(name: str):
    return {"int32": mx.int32, "int64": mx.int64}[name]


def np_index_dtype_from_name(name: str):
    return {"int32": np.int32, "int64": np.int64}[name]


def set_device(name: str) -> str:
    if name == "gpu":
        ms.use_gpu()
        return "gpu"
    mx.set_default_device(mx.Device(mx.cpu, 0))
    return "cpu"


def eval_sparse(array) -> None:
    if isinstance(array, ms.COOArray):
        mx.eval(array.data, array.row, array.col)
    elif isinstance(array, (ms.CSRArray, ms.CSCArray)):
        mx.eval(array.data, array.indices, array.indptr)
    else:
        mx.eval(array)


def make_coo(n: int, density: float, dtype_name: str, index_dtype_name: str, seed: int):
    rng = np.random.default_rng(seed)
    nnz = int(round(float(density) * n * n))
    nnz = max(0, min(nnz, n * n))
    flat = rng.choice(n * n, size=nnz, replace=False)
    row = (flat // n).astype(np_index_dtype_from_name(index_dtype_name), copy=False)
    col = (flat % n).astype(np_index_dtype_from_name(index_dtype_name), copy=False)
    if dtype_name == "complex64":
        values = (
            rng.standard_normal(nnz, dtype=np.float32)
            + 1j * rng.standard_normal(nnz, dtype=np.float32)
        ).astype(np.complex64)
    else:
        values = rng.standard_normal(nnz).astype(np_dtype_from_name(dtype_name))
    return ms.coo_array(
        (
            mx.array(values, dtype=dtype_from_name(dtype_name)),
            (
                mx.array(row, dtype=index_dtype_from_name(index_dtype_name)),
                mx.array(col, dtype=index_dtype_from_name(index_dtype_name)),
            ),
        ),
        shape=(n, n),
        canonical=True,
    )


def as_format(array, format_name: str):
    if format_name == "coo":
        return array
    if format_name == "csr":
        return array.tocsr(canonical=True)
    if format_name == "csc":
        return array.tocsc(canonical=True)
    raise ValueError(format_name)


def normalize_rhs_for_lhs(lhs_format: str, rhs):
    if lhs_format == "coo":
        if isinstance(rhs, ms.COOArray):
            return rhs
        return rhs.tocoo(canonical=None if isinstance(rhs, ms.CSRArray) else False)
    if lhs_format == "csr":
        if isinstance(rhs, ms.CSRArray):
            return rhs
        return rhs.tocsr(canonical=True)
    if lhs_format == "csc":
        if isinstance(rhs, ms.CSCArray):
            return rhs
        return rhs.tocsc(canonical=True)
    raise ValueError(lhs_format)


def same_format_matmul(lhs_format: str, lhs, normalized_rhs):
    if lhs_format == "coo":
        return ms.coo_matmat(lhs, normalized_rhs)
    if lhs_format == "csr":
        return ms.csr_matmat(lhs, normalized_rhs)
    if lhs_format == "csc":
        return ms.csc_matmat(lhs, normalized_rhs)
    raise ValueError(lhs_format)


def time_call(fn, *, warmup: int, iters: int):
    samples = []
    for i in range(warmup + iters):
        start = time.perf_counter()
        out = fn()
        eval_sparse(out)
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        if i >= warmup:
            samples.append(elapsed_ms)
    return statistics.median(samples)


def main() -> None:
    args = parse_args()
    device = set_device(args.device)
    rows = []

    for n in args.sizes:
        for density in args.densities:
            lhs_base = make_coo(n, density, args.dtype, args.index_dtype, args.seed + n)
            rhs_base = make_coo(
                n, density, args.dtype, args.index_dtype, args.seed + 101 + n
            )
            eval_sparse(lhs_base)
            eval_sparse(rhs_base)

            for lhs_format in FORMATS:
                lhs = as_format(lhs_base, lhs_format)
                eval_sparse(lhs)
                for rhs_format in FORMATS:
                    if lhs_format == rhs_format:
                        continue
                    rhs = as_format(rhs_base, rhs_format)
                    eval_sparse(rhs)
                    normalized_rhs = normalize_rhs_for_lhs(lhs_format, rhs)
                    eval_sparse(normalized_rhs)

                    mixed_ms = time_call(
                        lambda lhs=lhs, rhs=rhs: lhs @ rhs,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    conversion_ms = time_call(
                        lambda lhs_format=lhs_format, rhs=rhs: normalize_rhs_for_lhs(
                            lhs_format, rhs
                        ),
                        warmup=args.warmup,
                        iters=args.iters,
                    )

                    def run_pre_normalized(
                        lhs_format=lhs_format,
                        lhs=lhs,
                        normalized_rhs=normalized_rhs,
                    ):
                        return same_format_matmul(lhs_format, lhs, normalized_rhs)

                    normalized_ms = time_call(
                        run_pre_normalized,
                        warmup=args.warmup,
                        iters=args.iters,
                    )

                    rows.append(
                        {
                            "n": n,
                            "density": density,
                            "lhs_format": lhs_format,
                            "rhs_format": rhs_format,
                            "dtype": args.dtype,
                            "index_dtype": args.index_dtype,
                            "device": device,
                            "lhs_nnz": lhs.nnz,
                            "rhs_nnz": rhs.nnz,
                            "normalized_rhs_nnz": normalized_rhs.nnz,
                            "mixed_operator_median_ms": mixed_ms,
                            "rhs_conversion_median_ms": conversion_ms,
                            "pre_normalized_matmul_median_ms": normalized_ms,
                            "direct_mixed_kernel_median_ms": None,
                        }
                    )

    for row in rows:
        print(
            "mixed-matmul"
            f" n={row['n']}"
            f" density={row['density']}"
            f" {row['lhs_format']}@{row['rhs_format']}"
            f" dtype={row['dtype']}"
            f" index={row['index_dtype']}"
            f" device={row['device']}"
            f" nnz={row['lhs_nnz']}x{row['rhs_nnz']}"
            f" mixed={row['mixed_operator_median_ms']:.4f}ms"
            f" convert={row['rhs_conversion_median_ms']:.4f}ms"
            f" normalized={row['pre_normalized_matmul_median_ms']:.4f}ms"
            " direct=NA"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
