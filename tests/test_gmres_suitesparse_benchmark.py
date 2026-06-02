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

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _gmres_suitesparse_benchmark():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "bench_gmres_suitesparse_normal_equations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bench_gmres_suitesparse_normal_equations_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suitesparse_gmres_benchmark_builds_normal_equation_case():
    bench = _gmres_suitesparse_benchmark()

    normal, rhs, metadata = bench.normal_equation_case("well1033")

    assert metadata["source_matrix"] == "well1033"
    assert metadata["source_shape"] == [1033, 320]
    assert metadata["normal_shape"] == [320, 320]
    assert normal.shape == (320, 320)
    assert rhs.shape == (320,)
    assert normal.nnz > 0
    assert np.isfinite(metadata["rhs_norm"])
    assert metadata["rhs_norm"] > 0.0


def test_suitesparse_gmres_benchmark_mlx_conversion_preserves_normal_matvec(mx):
    bench = _gmres_suitesparse_benchmark()
    normal, _rhs, _metadata = bench.normal_equation_case("well1033")
    mlx_normal = bench.to_mlx_csr(normal, device="cpu")
    vector = np.linspace(-1.0, 1.0, normal.shape[1], dtype=np.float32)

    got = bench.to_numpy(mlx_normal @ mx.array(vector))

    np.testing.assert_allclose(got, normal @ vector, rtol=2e-5, atol=2e-5)
