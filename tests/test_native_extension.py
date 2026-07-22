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
import pytest

import mlx_sparse as ms
from mlx_sparse._host import to_numpy


@pytest.mark.native
def test_identity_like_native_smoke(mx):
    if not ms.is_available():
        pytest.skip("native extension is not built")

    x = mx.array(np.array([1.0, 2.0], dtype=np.float32))
    y = ms.identity_like(x)

    np.testing.assert_allclose(to_numpy(y), np.array([1.0, 2.0], dtype=np.float32))


@pytest.mark.native
def test_v006_native_symbols_are_exported():
    if not ms.is_available():
        pytest.skip("native extension is not built")

    import mlx_sparse._ext as ext

    required = (
        "random_coo_indices",
        "random_csr_indices",
        "random_csc_indices",
        "csr_add",
        "coo_kron",
        "coo_block",
        "coo_triangular",
        "csr_triangular",
        "csc_triangular",
        "csr_bicgstab",
        "csr_bicgstab_jacobi",
        "csr_bicgstab_ilu0",
        "csr_bicgstab_exact_lu",
        "csr_bicgstab_exact_cholesky",
    )
    missing = [name for name in required if not hasattr(ext, name)]
    assert not missing, f"native extension is missing v0.0.6b0 symbols: {missing}"

    capabilities = ext._compiled_capabilities()
    if capabilities["accelerate"]:
        assert hasattr(ext, "csr_bicgstab_exact_accelerate")
