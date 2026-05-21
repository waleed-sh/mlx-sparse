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

import mlx.core as mx
import numpy as np

import mlx_sparse as ms


def laplacian_2d(n: int) -> ms.CSRArray:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def idx(i: int, j: int) -> int:
        return i * n + j

    for i in range(n):
        for j in range(n):
            center = idx(i, j)
            rows.append(center)
            cols.append(center)
            vals.append(4.0)
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= ni < n and 0 <= nj < n:
                    rows.append(center)
                    cols.append(idx(ni, nj))
                    vals.append(-1.0)

    data = mx.array(np.asarray(vals, dtype=np.float32))
    row = mx.array(np.asarray(rows, dtype=np.int32))
    col = mx.array(np.asarray(cols, dtype=np.int32))
    return ms.coo_array((data, (row, col)), shape=(n * n, n * n)).tocsr(canonical=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    ms.use_device(args.device)
    a = laplacian_2d(16)
    x = mx.array(np.ones((256,), dtype=np.float32))
    y = a @ x
    mx.eval(y)
    print(y)
