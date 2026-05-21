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

import mlx.core as mx
import numpy as np


def to_numpy(array: mx.array) -> np.ndarray:
    """Materialize an MLX array on the host."""

    if array.dtype == mx.bfloat16:
        array = array.astype(mx.float32)
    mx.eval(array)
    return np.array(array)


def to_mx(array: np.ndarray, *, dtype=None) -> mx.array:
    if dtype is None:
        return mx.array(array)
    return mx.array(array, dtype=dtype)
