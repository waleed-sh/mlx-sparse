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
import pytest


def _direct_solver_benchmark():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "bench_native_cpu_direct_solvers.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bench_native_cpu_direct_solvers_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repeated_solve_benchmark():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "bench_native_cpu_repeated_solves.py"
    )
    spec = importlib.util.spec_from_file_location(
        "bench_native_cpu_repeated_solves_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_density_for_size_keeps_target_row_occupancy():
    bench = _direct_solver_benchmark()

    assert bench.density_for_size(512, target_nnz_per_row=16) == pytest.approx(16 / 512)
    assert bench.density_for_size(32, target_nnz_per_row=16) == pytest.approx(0.25)


def test_direct_banded_family_uses_density_derived_bandwidth():
    bench = _direct_solver_benchmark()
    rng = np.random.default_rng(0)

    narrow = bench.make_banded_spd(
        64,
        bench.density_for_size(64, target_nnz_per_row=4),
        rng,
    )
    wide = bench.make_banded_spd(
        64,
        bench.density_for_size(64, target_nnz_per_row=16),
        rng,
    )

    assert wide.nnz > narrow.nnz
    assert np.diff(wide.indptr).max() > np.diff(narrow.indptr).max()


def test_repeated_solve_benchmark_reuses_direct_density_policy():
    direct = _direct_solver_benchmark()
    repeated = _repeated_solve_benchmark()

    assert repeated.DEFAULT_NRHS == [1, 2, 4, 8, 16, 32]
    assert repeated.MAX_BENCHMARK_DIMENSION == direct.MAX_BENCHMARK_DIMENSION
    assert repeated.density_for_size(1024, target_nnz_per_row=16) == pytest.approx(
        direct.density_for_size(1024, target_nnz_per_row=16)
    )
