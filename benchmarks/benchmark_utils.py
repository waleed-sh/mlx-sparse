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

"""Shared utilities for fair mlx-sparse benchmark timing.

The most important helper in this file is :func:`force_eval`.  It evaluates
all structural buffers of sparse containers, not only their value buffer.  This
keeps dynamic sparse operations such as SpGEMM, conversions, and
canonicalization from looking artificially cheap in benchmark reports.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import numpy as np
import scipy.sparse as sp

import mlx_sparse as ms

CPU_THREADS_ENV = "MLX_SPARSE_CPU_THREADS"


@dataclass(frozen=True, slots=True)
class BenchmarkTiming:
    """Summary statistics for one timed benchmark operation."""

    samples_ms: tuple[float, ...]

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.samples_ms))

    @property
    def mean_ms(self) -> float:
        return float(statistics.fmean(self.samples_ms))

    @property
    def stdev_ms(self) -> float:
        if len(self.samples_ms) < 2:
            return 0.0
        return float(statistics.stdev(self.samples_ms))

    def as_dict(self, *, digits: int = 6) -> dict[str, Any]:
        return {
            "min_ms": round(self.min_ms, digits),
            "median_ms": round(self.median_ms, digits),
            "mean_ms": round(self.mean_ms, digits),
            "stdev_ms": round(self.stdev_ms, digits),
            "samples_ms": [round(value, digits) for value in self.samples_ms],
        }


def dtype_name(dtype) -> str:
    """Return a stable, compact dtype name for JSON benchmark reports."""

    return str(dtype).replace("mlx.core.", "")


def structural_arrays(result: Any) -> tuple[mx.array, ...]:
    """Return MLX arrays that must be evaluated for ``result`` to be real.

    Sparse containers contribute all structural buffers:

    * CSR/CSC: ``data``, ``indices``, and ``indptr``
    * COO: ``data``, ``row``, and ``col``

    Dense MLX results contribute the array itself.  Containers, mappings, and
    factorization dataclasses are walked recursively so benchmarks can time
    direct solver factor objects without hand-written evaluator lambdas.
    """

    arrays: list[mx.array] = []
    _collect_structural_arrays(result, arrays)
    return tuple(arrays)


def force_eval(result: Any) -> Any:
    """Evaluate an MLX result and return it unchanged."""

    arrays = structural_arrays(result)
    if arrays:
        mx.eval(*arrays)
    return result


def force_scipy_eval(result: Any) -> Any:
    """Touch eager SciPy/NumPy results so benchmark callbacks are symmetric.

    SciPy sparse operations are eager, but touching the structural buffers keeps
    the intent explicit and handles functions that return tuples/lists of
    sparse or dense results.
    """

    if result is None:
        return result
    if sp.issparse(result):
        _touch_scipy_sparse(result)
        return result
    if isinstance(result, np.ndarray | np.generic):
        np.asarray(result)
        return result
    if isinstance(result, Mapping):
        for value in result.values():
            force_scipy_eval(value)
        return result
    if isinstance(result, tuple | list):
        for value in result:
            force_scipy_eval(value)
        return result
    return result


def time_result(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    evaluator: Callable[[Any], Any] = force_eval,
) -> BenchmarkTiming:
    """Time ``fn`` after warmups, evaluating each returned result."""

    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}.")
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}.")

    for _ in range(warmup):
        evaluator(fn())

    samples: list[float] = []
    for _ in range(iters):
        start_ns = time.perf_counter_ns()
        evaluator(fn())
        end_ns = time.perf_counter_ns()
        samples.append((end_ns - start_ns) / 1_000_000.0)
    return BenchmarkTiming(samples_ms=tuple(samples))


def scipy_speedup(*, scipy_ms: float, native_ms: float) -> float | None:
    """Return ``scipy_ms / native_ms`` when the ratio is meaningful."""

    if native_ms <= 0.0:
        return None
    return float(scipy_ms / native_ms)


def sparse_matrix_metadata(matrix: Any) -> dict[str, Any]:
    """Return shape, sparsity, dtype, index dtype, and axis length stats."""

    shape = tuple(int(dim) for dim in matrix.shape)
    n_rows, n_cols = shape
    nnz = int(matrix.nnz)
    density = float(nnz / (n_rows * n_cols)) if n_rows and n_cols else 0.0

    if isinstance(matrix, ms.CSRArray):
        row_lengths = _compressed_lengths(matrix.indptr)
        col_lengths = _coordinate_counts(matrix.indices, n_cols)
        fmt = "csr"
    elif isinstance(matrix, ms.CSCArray):
        row_lengths = _coordinate_counts(matrix.indices, n_rows)
        col_lengths = _compressed_lengths(matrix.indptr)
        fmt = "csc"
    elif isinstance(matrix, ms.COOArray):
        row_lengths = _coordinate_counts(matrix.row, n_rows)
        col_lengths = _coordinate_counts(matrix.col, n_cols)
        fmt = "coo"
    else:
        raise TypeError(
            "sparse_matrix_metadata expects COOArray, CSRArray, or CSCArray, "
            f"got {type(matrix).__name__}."
        )

    return {
        "format": fmt,
        "shape": [n_rows, n_cols],
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": nnz,
        "density": density,
        "dtype": dtype_name(matrix.dtype),
        "index_dtype": dtype_name(matrix.index_dtype),
        "row_lengths": _length_stats(row_lengths),
        "col_lengths": _length_stats(col_lengths),
    }


def cpu_runtime_metadata(*, warmup: int, iters: int) -> dict[str, Any]:
    """Return CPU-only benchmark runtime metadata required by v0.0.4b1."""

    device = ms.use_cpu(require_available=True)
    hardware_threads = os.cpu_count()
    worker_count, worker_source = _configured_worker_count(hardware_threads)
    return {
        "native_extension_available": bool(ms.is_available()),
        "selected_mlx_device": str(_default_device_fallback(device)),
        "metal_available": bool(ms.capabilities.METAL),
        "accelerate_available": bool(ms.capabilities.ACCELERATE),
        "cpu_model": _cpu_model(),
        "hardware_core_count": _hardware_core_count(),
        "hardware_thread_count": hardware_threads,
        "configured_worker_count": worker_count,
        "configured_worker_count_source": worker_source,
        "cpu_threads_env": os.environ.get(CPU_THREADS_ENV),
        "warmup_count": int(warmup),
        "iteration_count": int(iters),
    }


def _collect_structural_arrays(result: Any, arrays: list[mx.array]) -> None:
    if result is None:
        return
    if isinstance(result, mx.array):
        arrays.append(result)
        return
    if isinstance(result, ms.CSRArray | ms.CSCArray):
        arrays.extend((result.data, result.indices, result.indptr))
        return
    if isinstance(result, ms.COOArray):
        arrays.extend((result.data, result.row, result.col))
        return
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        for field in dataclasses.fields(result):
            _collect_structural_arrays(getattr(result, field.name), arrays)
        return
    if isinstance(result, Mapping):
        for value in result.values():
            _collect_structural_arrays(value, arrays)
        return
    if isinstance(result, tuple | list):
        for value in result:
            _collect_structural_arrays(value, arrays)


def _touch_scipy_sparse(matrix: sp.spmatrix) -> None:
    _ = matrix.shape
    _ = matrix.nnz
    if hasattr(matrix, "data"):
        _ = matrix.data
    if hasattr(matrix, "indices"):
        _ = matrix.indices
    if hasattr(matrix, "indptr"):
        _ = matrix.indptr
    if hasattr(matrix, "row"):
        _ = matrix.row
    if hasattr(matrix, "col"):
        _ = matrix.col


def _to_numpy(array: mx.array) -> np.ndarray:
    if array.dtype == mx.bfloat16:
        array = array.astype(mx.float32)
    mx.eval(array)
    return np.asarray(array)


def _compressed_lengths(indptr: mx.array) -> np.ndarray:
    ptr = _to_numpy(indptr).astype(np.int64, copy=False)
    return np.diff(ptr)


def _coordinate_counts(coords: mx.array, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    values = _to_numpy(coords).astype(np.int64, copy=False)
    if values.size == 0:
        return np.zeros((length,), dtype=np.int64)
    return np.bincount(values, minlength=length).astype(np.int64, copy=False)


def _length_stats(lengths: np.ndarray) -> dict[str, Any]:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "empty": 0,
            "nonempty": 0,
        }
    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "std": float(lengths.std()),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "empty": int(np.count_nonzero(lengths == 0)),
        "nonempty": int(np.count_nonzero(lengths != 0)),
    }


def _default_device_fallback(device: mx.Device) -> mx.Device:
    getter = getattr(mx, "default_device", None)
    if getter is None:
        return device
    try:
        return getter()
    except Exception:
        return device


def _configured_worker_count(hardware_threads: int | None) -> tuple[int | None, str]:
    raw_value = os.environ.get(CPU_THREADS_ENV)
    if raw_value is None or raw_value.strip() == "":
        return hardware_threads, "hardware_thread_count_default"
    try:
        value = int(raw_value)
    except ValueError:
        return None, f"invalid_{CPU_THREADS_ENV}"
    if value <= 0:
        return None, f"invalid_{CPU_THREADS_ENV}"
    return value, CPU_THREADS_ENV


def _cpu_model() -> str | None:
    candidates: list[str | None] = []
    if platform.system().lower() == "darwin":
        candidates.append(_sysctl_string("machdep.cpu.brand_string"))
    candidates.extend(
        [
            platform.processor(),
            platform.machine(),
            platform.platform(),
        ]
    )
    for candidate in candidates:
        if candidate:
            stripped = candidate.strip()
            if stripped:
                return stripped
    return None


def _hardware_core_count() -> int | None:
    if platform.system().lower() == "darwin":
        for key in ("hw.physicalcpu", "hw.ncpu"):
            raw_value = _sysctl_string(key)
            if raw_value is None:
                continue
            try:
                value = int(raw_value)
            except ValueError:
                continue
            if value > 0:
                return value
    return os.cpu_count()


def _sysctl_string(key: str) -> str | None:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", key],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None
