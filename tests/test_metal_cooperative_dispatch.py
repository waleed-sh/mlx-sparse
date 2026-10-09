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

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse import _native


def test_native_grid_covers_integer_boundaries_without_wrapping(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "grid.cpp"
    source.write_text(r"""
#include "common/metal_dispatch.h"
#include <cassert>
#include <limits>

int main() {
  const size_t counts[] = {0, 1, 65535, 65536, 65537,
      (size_t(1) << 25) - 1, size_t(1) << 25, (size_t(1) << 25) + 1,
      (size_t(1) << 31) - 1, size_t(1) << 31, (size_t(1) << 31) + 1,
      (size_t(1) << 32) - 1, size_t(1) << 32, (size_t(1) << 32) + 1,
      (size_t(1) << 40) + 17,
      size_t(std::numeric_limits<uint32_t>::max()) * 65536};
  for (size_t count : counts) {
    const auto grid = mlx_sparse::cooperative_grid(count);
    assert(grid.groups_x > 0 && grid.groups_y > 0);
    assert(grid.groups_x * 128 <= std::numeric_limits<uint32_t>::max());
    assert(grid.groups_y <= std::numeric_limits<uint32_t>::max());
    assert(grid.groups_x * grid.groups_y >= count);
    if (count) {
      assert(grid.groups_x * grid.groups_y - count < grid.groups_y);
      for (size_t id : {size_t(0), count / 2, count - 1}) {
        const size_t x = id % grid.groups_x;
        const size_t y = id / grid.groups_x;
        assert(x < grid.groups_x && y < grid.groups_y);
        assert(y * grid.groups_x + x == id);
      }
    }
  }
  const size_t limit = std::numeric_limits<uint32_t>::max();
  assert(!mlx_sparse::needs_wide_dense_indices(limit, limit));
  assert(mlx_sparse::needs_wide_dense_indices(limit + 1, 1));
  assert(mlx_sparse::needs_wide_dense_indices(1, limit + 1));
  try {
    mlx_sparse::cooperative_grid(std::numeric_limits<size_t>::max());
    return 1;
  } catch (const std::overflow_error &) {
    return 0;
  }
}
""")
    binary = tmp_path / "grid"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(root / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


@pytest.mark.gpu
def test_compiled_cooperative_kernels_directly_include_wide_indexing(mx, tmp_path):
    import sys

    if sys.platform != "darwin":
        pytest.skip("Direct Metal execution requires macOS")
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("Clang compiler unavailable")
    extension = _native.extension()
    assert extension is not None
    metallib = Path(extension.__file__).parent / "mlx_sparse.metallib"
    source = Path(__file__).parent / "native" / "cooperative_metal.mm"
    binary = tmp_path / "cooperative_metal"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fobjc-arc",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(binary), str(metallib)], check=True, capture_output=True, text=True
    )


def _regular_csr(mx, rows, degree, dtype, index_dtype):
    return ms.csr_array(
        (
            mx.full((rows * degree,), 1.0 / degree, dtype=getattr(mx, dtype)),
            mx.tile(mx.arange(degree, dtype=getattr(mx, index_dtype)), rows),
            mx.arange(rows + 1, dtype=getattr(mx, index_dtype)) * degree,
        ),
        shape=(rows, degree),
        canonical=True,
        sorted_indices=True,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("width", [511, 512, 520])
def test_cooperative_matmul_crosses_flattened_thread_limit(mx, width):
    # The degree selects the native 128-thread cooperative path.
    # Outputs straddle 2**25, where the previous thread grid wrapped at 2**32.
    rows, degree = 65536, 32
    a = _regular_csr(mx, rows, degree, "float32", "int32")
    rhs = mx.broadcast_to(mx.arange(width, dtype=mx.float32)[None, :], (degree, width))
    out = _native.csr_matmul(a.data, a.indices, a.indptr, rhs, a.shape)
    error = mx.max(mx.abs(out - mx.arange(width, dtype=mx.float32)[None, :]))
    assert error.item() == 0.0


@pytest.mark.gpu
def test_cooperative_batched_product_crosses_flattened_thread_limit(mx):
    a = _regular_csr(mx, 32768, 32, "float32", "int32")
    rhs = mx.stack([mx.ones((32, 513)), mx.full((32, 513), 2.0)])
    out = ms.csr_batched_matmul(a, rhs)
    expected = mx.array([1.0, 2.0])[:, None, None]
    assert mx.max(mx.abs(out - expected)).item() == 0.0


@pytest.mark.gpu
@pytest.mark.parametrize("degree", [1, 32])
def test_padded_batched_grids_cover_scalar_and_cooperative_outputs(mx, degree):
    # This count needs several grid planes and padding in the final plane.
    a = _regular_csr(mx, 257, degree, "float32", "int64")
    rhs = mx.stack([mx.ones((degree, 259)), mx.full((degree, 259), 3.0)])
    out = ms.csr_batched_matmul(a, rhs)
    assert mx.max(mx.abs(out - mx.array([1.0, 3.0])[:, None, None])).item() == 0.0


_OPS = [
    "matvec",
    "matmul",
    "batched_matvec",
    "batched_matmul",
    "diagonal",
    "row_sums",
    "row_norms",
    "csc_transpose_matvec",
    "csc_diagonal",
    "csc_col_sums",
    "csc_col_norms",
]


@pytest.mark.gpu
@pytest.mark.parametrize("operation", _OPS)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "complex64"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_cooperative_kernels_preserve_dtypes_and_reductions(
    mx, scipy_sparse, to_numpy, operation, dtype, index_dtype
):
    n, degree = 71, 40
    dense = np.zeros((n, n), dtype=np.complex64 if dtype == "complex64" else np.float32)
    for row in range(n):
        for offset in range(degree):
            value = ((offset % 9) + 1) / 16
            dense[row, (row + offset) % n] = value + (
                0.125j if dtype == "complex64" else 0
            )
    sp = scipy_sparse.csr_array(dense)
    a = ms.csr_array(
        (
            mx.array(sp.data, dtype=getattr(mx, dtype)),
            mx.array(sp.indices, dtype=getattr(mx, index_dtype)),
            mx.array(sp.indptr, dtype=getattr(mx, index_dtype)),
        ),
        shape=sp.shape,
        canonical=True,
        sorted_indices=True,
    )
    x = (np.arange(n, dtype=np.float32) % 7) / 8
    rhs = np.stack([x, x + 0.25, x - 0.25], axis=1)
    x_mx, rhs_mx = mx.array(x, dtype=getattr(mx, dtype)), mx.array(
        rhs, dtype=getattr(mx, dtype)
    )
    if operation == "matvec":
        got, expected = a @ x_mx, dense @ x
    elif operation == "matmul":
        got, expected = a @ rhs_mx, dense @ rhs
    elif operation == "batched_matvec":
        got = ms.csr_batched_matvec(a, mx.stack([x_mx, x_mx * 2]))
        expected = np.stack([dense @ x, dense @ (x * 2)])
    elif operation == "batched_matmul":
        got = ms.csr_batched_matmul(a, mx.stack([rhs_mx, rhs_mx * 2]))
        expected = np.stack([dense @ rhs, dense @ (rhs * 2)])
    elif operation == "diagonal":
        got, expected = a.diagonal(), np.diag(dense)
    elif operation == "row_sums":
        got, expected = a.sum(axis=1), dense.sum(axis=1)
    elif operation == "row_norms":
        got, expected = ms.csr_row_norms(a), np.linalg.norm(dense, axis=1)
    else:
        csc = a.tocsc()
        if operation == "csc_transpose_matvec":
            got, expected = ms.csc_matvec_transpose(csc, x_mx), dense.T @ x
        elif operation == "csc_diagonal":
            got, expected = csc.diagonal(), np.diag(dense)
        elif operation == "csc_col_sums":
            got, expected = csc.sum(axis=0), dense.sum(axis=0)
        else:
            got, expected = csc.col_norms(), np.linalg.norm(dense, axis=0)
    tolerance = {"float32": 2e-5, "complex64": 2e-5, "float16": 2e-3, "bfloat16": 1e-2}[
        dtype
    ]
    np.testing.assert_allclose(to_numpy(got), expected, rtol=tolerance, atol=tolerance)
