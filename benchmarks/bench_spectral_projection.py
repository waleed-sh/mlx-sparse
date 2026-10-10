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
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import scipy.sparse as sp

import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse._host import to_numpy


def _csr(a):
    a = a.tocsr().astype(np.float32)
    a.sort_indices()
    return ms.csr_array(
        (mx.array(a.data), mx.array(a.indices), mx.array(a.indptr)),
        shape=a.shape,
        sorted_indices=True,
        canonical=True,
    )


def _eval(result):
    arrays = result if isinstance(result, tuple) else (result,)
    mx.eval(*(a for a in arrays if a is not None))


def _time(fn, repeats):
    for _ in range(3):
        _eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        _eval(fn())
        samples.append(1000 * (time.perf_counter() - start))
    return float(np.median(samples))


def _quality(a, result, routine):
    bound = float(np.sqrt(abs(a).sum(axis=0).max() * abs(a).sum(axis=1).max()))
    if routine == "eigs":
        values, vectors = (to_numpy(x) for x in result)
        residual = a @ vectors - vectors * values
        return {
            "relative_residual": float(np.linalg.norm(residual) / bound),
            "max_vector_norm_error": float(
                np.max(abs(np.linalg.norm(vectors, axis=0) - 1))
            ),
        }
    u, s, vh = (to_numpy(x) for x in result)
    return {
        "left_equation_relative_residual": float(
            np.linalg.norm(a @ vh.T - u * s) / bound
        ),
        "right_equation_relative_residual": float(
            np.linalg.norm(a.T @ u - vh.T * s) / bound
        ),
        "left_orthogonality_error": float(np.linalg.norm(u.T @ u - np.eye(len(s)))),
        "right_orthogonality_error": float(np.linalg.norm(vh @ vh.T - np.eye(len(s)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 2048, 16384])
    parser.add_argument("--ncv", type=int, default=24)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mx.set_default_device(mx.Device(getattr(mx, args.device), 0))
    records = []
    for n in args.sizes:
        diagonal = np.linspace(1, 8, n, dtype=np.float32)
        square = sp.diags(
            [np.full(n - 1, -0.25), diagonal, np.full(n - 1, 0.5)],
            [-1, 0, 1],
            shape=(n, n),
            dtype=np.float32,
            format="csr",
        )
        tall = sp.vstack([square, 0.5 * sp.eye(n, dtype=np.float32)], format="csr")
        wide = tall.T.tocsr()
        for routine, shape_name, a in [
            ("eigs", "square", square),
            ("svds", "tall", tall),
            ("svds", "wide", wide),
        ]:
            matrix = _csr(a)
            mx.eval(matrix.data, matrix.indices, matrix.indptr)
            kwargs = dict(k=args.k, ncv=args.ncv)
            fn = getattr(linalg, routine)
            vectors = fn(matrix, **kwargs)
            _eval(vectors)
            quality = _quality(a, vectors, routine)
            for requested in [True, False]:
                option = (
                    "return_eigenvectors"
                    if routine == "eigs"
                    else "return_singular_vectors"
                )
                milliseconds = _time(
                    lambda: fn(matrix, **kwargs, **{option: requested}), args.repeats
                )
                record = {
                    "routine": routine,
                    "shape": shape_name,
                    "rows": a.shape[0],
                    "cols": a.shape[1],
                    "nnz": a.nnz,
                    "ncv": args.ncv,
                    "k": args.k,
                    "vectors": requested,
                    "median_ms": milliseconds,
                    **quality,
                }
                records.append(record)
                print(
                    f"{routine} {shape_name} n={n} vectors={requested}: {milliseconds:.3f} ms",
                    flush=True,
                )
    result = {
        "package": str(Path(ms.__file__).resolve()),
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
        "device": args.device,
        "repeats": args.repeats,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
