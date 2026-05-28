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
import scipy.sparse as sp

import mlx_sparse as ms


def _benchmark_utils():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_utils.py"
    spec = importlib.util.spec_from_file_location("benchmark_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_force_eval_forces_all_sparse_structural_buffers(mx, monkeypatch):
    helpers = _benchmark_utils()
    data = mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    indices = mx.array(np.array([0, 2, 1], dtype=np.int32))
    indptr = mx.array(np.array([0, 2, 3], dtype=np.int32))
    csr = ms.csr_array(
        (data, indices, indptr),
        shape=(2, 3),
        sorted_indices=True,
        canonical=True,
    )
    csc = csr.tocsc(canonical=True)
    coo = ms.coo_array(
        (
            data,
            (
                mx.array(np.array([0, 0, 1], dtype=np.int32)),
                mx.array(np.array([0, 2, 1], dtype=np.int32)),
            ),
        ),
        shape=(2, 3),
        canonical=True,
    )
    dense = mx.array(np.ones((2, 2), dtype=np.float32))

    calls = []
    monkeypatch.setattr(helpers.mx, "eval", lambda *args: calls.append(args))

    returned = helpers.force_eval((csr, csc, coo, dense))

    assert returned[0] is csr
    seen = {id(array) for call in calls for array in call}
    for array in (
        csr.data,
        csr.indices,
        csr.indptr,
        csc.data,
        csc.indices,
        csc.indptr,
        coo.data,
        coo.row,
        coo.col,
        dense,
    ):
        assert id(array) in seen


def test_sparse_matrix_metadata_reports_axis_statistics(mx):
    helpers = _benchmark_utils()
    csr = ms.csr_array(
        (
            mx.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
            mx.array(np.array([0, 2, 2, 3], dtype=np.int32)),
            mx.array(np.array([0, 2, 3, 4], dtype=np.int32)),
        ),
        shape=(3, 4),
        sorted_indices=True,
        canonical=True,
    )

    metadata = helpers.sparse_matrix_metadata(csr)

    assert metadata["format"] == "csr"
    assert metadata["shape"] == [3, 4]
    assert metadata["nnz"] == 4
    assert metadata["density"] == pytest.approx(4 / 12)
    assert metadata["dtype"] == "float32"
    assert metadata["index_dtype"] == "int32"
    assert metadata["row_lengths"]["min"] == 1
    assert metadata["row_lengths"]["max"] == 2
    assert metadata["row_lengths"]["empty"] == 0
    assert metadata["col_lengths"]["min"] == 0
    assert metadata["col_lengths"]["max"] == 2
    assert metadata["col_lengths"]["empty"] == 1


def test_time_result_warms_up_and_records_samples():
    helpers = _benchmark_utils()
    calls = []
    evaluated = []

    def fn():
        calls.append("fn")
        return len(calls)

    timing = helpers.time_result(
        fn,
        warmup=2,
        iters=3,
        evaluator=lambda value: evaluated.append(value),
    )

    assert len(calls) == 5
    assert evaluated == [1, 2, 3, 4, 5]
    assert len(timing.samples_ms) == 3
    assert timing.min_ms >= 0.0


def test_force_scipy_eval_accepts_sparse_dense_and_nested_results():
    helpers = _benchmark_utils()
    matrix = sp.csr_matrix(
        (
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 1, 2], dtype=np.int32),
        ),
        shape=(2, 2),
    )
    dense = np.ones((2,), dtype=np.float32)

    result = {"matrix": matrix, "dense": dense}

    assert helpers.force_scipy_eval(result) is result
    assert helpers.scipy_speedup(scipy_ms=4.0, native_ms=2.0) == 2.0


@pytest.mark.cpu_only
def test_cpu_runtime_metadata_records_configured_workers(mx, monkeypatch):
    helpers = _benchmark_utils()
    monkeypatch.setenv(helpers.CPU_THREADS_ENV, "3")

    metadata = helpers.cpu_runtime_metadata(warmup=1, iters=2)

    assert metadata["selected_mlx_device"]
    assert metadata["configured_worker_count"] == 3
    assert metadata["configured_worker_count_source"] == helpers.CPU_THREADS_ENV
    assert metadata["warmup_count"] == 1
    assert metadata["iteration_count"] == 2
    assert "native_extension_available" in metadata
    assert "metal_available" in metadata
    assert "accelerate_available" in metadata
