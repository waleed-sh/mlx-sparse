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


import argparse
import json
import time
from statistics import median

import mlx.core as mx

import mlx_sparse as ms
from mlx_sparse import _native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    mx.set_default_device(mx.gpu)
    print(f"Native extension: {_native.extension().__file__}")
    print(f"Device: {mx.device_info()['device_name']}")
    results = []
    for rows, degree, width in [
        (1024, 32, 1),
        (8192, 64, 8),
        (4096, 128, 32),
        (65536, 32, 16),
        (4096, 256, 1),
    ]:
        a = ms.csr_array(
            (
                mx.full((rows * degree,), 1.0 / degree),
                mx.tile(mx.arange(degree, dtype=mx.int32), rows),
                mx.arange(rows + 1, dtype=mx.int32) * degree,
            ),
            shape=(rows, degree),
            canonical=True,
            sorted_indices=True,
        )
        rhs = mx.ones((degree, width))
        mx.eval(a.data, a.indices, a.indptr, rhs)
        for operation, call in [
            ("matmul", lambda: a @ rhs),
            ("row_sums", lambda: a.sum(axis=1)),
        ]:
            for _ in range(3):
                mx.eval(call())
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                mx.eval(call())
                times.append((time.perf_counter() - start) * 1000)
            results.append(
                dict(
                    operation=operation,
                    rows=rows,
                    degree=degree,
                    width=width,
                    median_ms=median(times),
                    samples_ms=times,
                )
            )
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    for row in results:
        print(
            f"{row['operation']:8} rows={row['rows']:6} degree={row['degree']:3} "
            f"width={row['width']:2}: {row['median_ms']:.4f} ms"
        )


if __name__ == "__main__":
    main()
