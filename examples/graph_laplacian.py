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


def cycle_graph_laplacian(n: int) -> ms.CSRArray:
    nodes = np.arange(n, dtype=np.int32)
    row = np.concatenate([nodes, nodes, nodes])
    col = np.concatenate([nodes, (nodes + 1) % n, (nodes - 1) % n]).astype(np.int32)
    data = np.concatenate(
        [
            np.full(n, 2.0, dtype=np.float32),
            np.full(n, -1.0, dtype=np.float32),
            np.full(n, -1.0, dtype=np.float32),
        ]
    )
    return ms.coo_array(
        (
            mx.array(data),
            (mx.array(row.astype(np.int32)), mx.array(col)),
        ),
        shape=(n, n),
    ).tocsr(canonical=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    ms.use_device(args.device)
    laplacian = cycle_graph_laplacian(args.nodes)
    signal = mx.sin(mx.arange(args.nodes, dtype=mx.float32) / args.nodes)
    response = laplacian @ signal
    mx.eval(response)
    print(response)


if __name__ == "__main__":
    main()
