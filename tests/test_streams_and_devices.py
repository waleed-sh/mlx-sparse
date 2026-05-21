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

import numpy as np

import mlx_sparse as ms
from mlx_sparse._host import to_numpy


def test_operations_accept_default_stream(mx):
    data = mx.array(np.array([1.0, 2.0], dtype=np.float32))
    indices = mx.array(np.array([0, 1], dtype=np.int32))
    indptr = mx.array(np.array([0, 2], dtype=np.int32))
    x = mx.array(np.array([5.0, 7.0], dtype=np.float32))

    csr = ms.csr_array((data, indices, indptr), shape=(1, 2))

    np.testing.assert_allclose(to_numpy(csr @ x), np.array([19.0], dtype=np.float32))
