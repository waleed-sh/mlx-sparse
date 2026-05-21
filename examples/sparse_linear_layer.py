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


def block_sparse_weight(features: int, blocks: int) -> ms.CSRArray:
    if features % blocks != 0:
        raise ValueError("features must be divisible by blocks.")
    block = features // blocks
    rows = []
    cols = []
    values = []
    for b in range(blocks):
        start = b * block
        for i in range(block):
            row = start + i
            rows.append(row)
            cols.append(row)
            values.append(1.0)
            if i + 1 < block:
                rows.append(row)
                cols.append(row + 1)
                values.append(0.25)

    return ms.coo_array(
        (
            mx.array(np.asarray(values, dtype=np.float32)),
            (
                mx.array(np.asarray(rows, dtype=np.int32)),
                mx.array(np.asarray(cols, dtype=np.int32)),
            ),
        ),
        shape=(features, features),
    ).tocsr(canonical=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    ms.use_device(args.device)
    weight = block_sparse_weight(args.features, args.blocks)
    x = mx.random.normal((args.batch, args.features, 1))
    y = weight @ x
    mx.eval(y)
    print(y.shape)


if __name__ == "__main__":
    main()
