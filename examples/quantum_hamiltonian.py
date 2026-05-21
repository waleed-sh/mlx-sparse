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


def transverse_field_ising(n_qubits: int, field: float) -> ms.CSRArray:
    dim = 1 << n_qubits
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    for state in range(dim):
        energy = 0.0
        for site in range(n_qubits - 1):
            z_i = 1 if state & (1 << site) else -1
            z_j = 1 if state & (1 << (site + 1)) else -1
            energy += -z_i * z_j
        rows.append(state)
        cols.append(state)
        values.append(energy)

        for site in range(n_qubits):
            rows.append(state)
            cols.append(state ^ (1 << site))
            values.append(-field)

    return ms.coo_array(
        (
            mx.array(np.asarray(values, dtype=np.float32)),
            (
                mx.array(np.asarray(rows, dtype=np.int32)),
                mx.array(np.asarray(cols, dtype=np.int32)),
            ),
        ),
        shape=(dim, dim),
    ).tocsr(canonical=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--field", type=float, default=0.7)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()

    ms.use_device(args.device)
    hamiltonian = transverse_field_ising(args.qubits, args.field)
    state = mx.ones((1 << args.qubits,), dtype=mx.float32)
    state = state / mx.sqrt(mx.sum(state * state))
    evolved_direction = hamiltonian @ state
    mx.eval(evolved_direction)
    print(evolved_direction)


if __name__ == "__main__":
    main()
