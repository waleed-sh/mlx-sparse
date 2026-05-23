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


def test_issparse_and_todense(mx):
    coo = ms.coo_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            (
                mx.array(np.array([0], dtype=np.int32)),
                mx.array(np.array([1], dtype=np.int32)),
            ),
        ),
        shape=(2, 3),
    )

    assert ms.issparse(coo)
    assert ms.issparse(coo.tocsr())
    assert ms.todense(coo).shape == (2, 3)


def test_public_constructor_exports():
    for name in ("fromdense", "from_dense", "from_numpy", "from_scipy", "asarray"):
        assert name in ms.__all__
        assert callable(getattr(ms, name))


def test_public_batched_operation_exports():
    for name in ("csr_batched_matvec", "csr_batched_matmul"):
        assert name in ms.__all__
        assert callable(getattr(ms, name))
