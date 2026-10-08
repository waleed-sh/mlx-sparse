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

"""The GPU COO conversions are built from MLX array operations.

The kernel path ranks every entry against every other entry, so its cost grows
with ``nnz`` squared. The array-op path sorts the row-major keys instead. These
cells hold it to the kernel path's exact output: same values, same order, same
dtypes, including where duplicate coordinates make the order a choice.
"""

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse import _convert

VALUE_DTYPES = ["float32", "float16", "bfloat16"]
INDEX_DTYPES = ["int32", "int64"]


def _gpu_available(mx):
    try:
        return mx.is_available(mx.Device(mx.gpu, 0))
    except Exception:  # pragma: no cover - platform dependent
        return False


def _require_gpu(mx):
    if not _gpu_available(mx):
        pytest.skip("No Metal device available.")


def _coo(mx, data, row, col, shape, value_dtype, index_dtype):
    return ms.coo_array(
        (
            mx.array(data).astype(getattr(mx, value_dtype)),
            (
                mx.array(row).astype(getattr(mx, index_dtype)),
                mx.array(col).astype(getattr(mx, index_dtype)),
            ),
        ),
        shape=shape,
    )


def _buffers(mx, array):
    mx.eval(array.data, array.indices, array.indptr)
    return array


def _identical(mx, left, right):
    for name in ("data", "indices", "indptr"):
        lhs = getattr(left, name)
        rhs = getattr(right, name)
        assert lhs.dtype == rhs.dtype, name
        assert np.array_equal(
            np.array(lhs.astype(mx.float32) if name == "data" else lhs),
            np.array(rhs.astype(mx.float32) if name == "data" else rhs),
        ), name


def _random_coo(seed, n_rows, n_cols, nnz):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(nnz).astype(np.float32),
        rng.integers(0, n_rows, size=nnz).astype(np.int64),
        rng.integers(0, n_cols, size=nnz).astype(np.int64),
    )


@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("to_format", ["csr", "csc"])
def test_gpu_conversion_matches_the_kernel_path_exactly(
    mx, value_dtype, index_dtype, to_format
):
    _require_gpu(mx)
    data, row, col = _random_coo(0, 37, 41, 500)
    shape = (37, 41)

    with mx.stream(mx.cpu):
        coo = _coo(mx, data, row, col, shape, value_dtype, index_dtype)
        assert not _convert.can_use_array_ops(coo.data, shape)
        kernel = _buffers(mx, getattr(coo, "to" + to_format)())

    with mx.stream(mx.gpu):
        coo = _coo(mx, data, row, col, shape, value_dtype, index_dtype)
        assert _convert.can_use_array_ops(coo.data, shape)
        array_ops = _buffers(mx, getattr(coo, "to" + to_format)())

    _identical(mx, array_ops, kernel)


@pytest.mark.parametrize("to_format", ["csr", "csc"])
def test_duplicate_coordinates_keep_their_input_order(mx, to_format):
    _require_gpu(mx)
    # Three entries in cell (0, 1) and two in (1, 0). Which value lands first is
    # a choice, and the two paths have to make the same one.
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    row = np.array([0, 1, 0, 1, 0, 2], dtype=np.int32)
    col = np.array([1, 0, 1, 0, 1, 2], dtype=np.int32)
    shape = (3, 3)

    with mx.stream(mx.cpu):
        kernel = _buffers(
            mx,
            getattr(
                _coo(mx, data, row, col, shape, "float32", "int32"), "to" + to_format
            )(),
        )
    with mx.stream(mx.gpu):
        array_ops = _buffers(
            mx,
            getattr(
                _coo(mx, data, row, col, shape, "float32", "int32"), "to" + to_format
            )(),
        )

    _identical(mx, array_ops, kernel)
    assert array_ops.nnz == 6


@pytest.mark.parametrize(
    "shape,row,col",
    [
        ((4, 4), [], []),
        ((1, 1), [0], [0]),
        ((1, 6), [0, 0, 0], [5, 0, 3]),
        ((6, 1), [4, 0, 2], [0, 0, 0]),
        ((5, 5), [2, 2, 2, 2], [1, 1, 1, 1]),
        ((3, 3), [0, 0, 0, 0], [2, 1, 0, 2]),
    ],
)
@pytest.mark.parametrize("to_format", ["csr", "csc"])
def test_degenerate_shapes_match_the_kernel_path(mx, shape, row, col, to_format):
    _require_gpu(mx)
    data = np.arange(1, len(row) + 1, dtype=np.float32)
    row = np.array(row, dtype=np.int32)
    col = np.array(col, dtype=np.int32)

    with mx.stream(mx.cpu):
        kernel = _buffers(
            mx,
            getattr(
                _coo(mx, data, row, col, shape, "float32", "int32"), "to" + to_format
            )(),
        )
    with mx.stream(mx.gpu):
        array_ops = _buffers(
            mx,
            getattr(
                _coo(mx, data, row, col, shape, "float32", "int32"), "to" + to_format
            )(),
        )

    _identical(mx, array_ops, kernel)


def test_empty_dimension_falls_back_to_the_kernel_path(mx):
    # A zero-length axis has no linear key to sort, and the histogram would be
    # a scatter into an empty accumulator.
    assert not _convert.can_use_array_ops(mx.zeros((0,)), (0, 4))
    assert not _convert.can_use_array_ops(mx.zeros((0,)), (4, 0))
    coo = ms.coo_array(
        (
            mx.zeros((0,), dtype=mx.float32),
            (mx.zeros((0,), dtype=mx.int32), mx.zeros((0,), dtype=mx.int32)),
        ),
        shape=(0, 4),
    )
    csr = coo.tocsr()
    assert csr.nnz == 0
    assert np.array_equal(np.array(csr.indptr), np.zeros(1, dtype=np.int32))


def test_shape_whose_linear_key_overflows_int64_falls_back(mx):
    _require_gpu(mx)
    with mx.stream(mx.gpu):
        # row * n_cols would leave int64, so the sort key could not order
        # entries; the same shape one dimension smaller stays representable.
        assert not _convert.can_use_array_ops(mx.zeros((1,)), (2**32, 2**32))
        assert _convert.can_use_array_ops(mx.zeros((1,)), (2**20, 2**20))


def test_complex_values_keep_the_kernel_path(mx):
    # mx.take builds its vjp from a scatter, and the Metal scatter has no
    # complex64 support, so a complex conversion would convert but not
    # differentiate.
    _require_gpu(mx)
    assert not _convert.can_use_array_ops(mx.zeros((4,), dtype=mx.complex64), (4, 4))

    values = mx.array(np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64))
    row = mx.array(np.array([1, 0], dtype=np.int32))
    col = mx.array(np.array([0, 1], dtype=np.int32))

    def converted(data):
        return ms.coo_array((data, (row, col)), shape=(2, 2)).tocsr().data

    cotangent = mx.array(np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex64))
    _, vjp = mx.vjp(converted, (values,), (cotangent,))
    mx.eval(vjp[0])
    assert vjp[0].shape == (2,)


@pytest.mark.parametrize("to_format", ["csr", "csc"])
def test_conversion_declares_sorted_indices_it_actually_has(mx, to_format):
    _require_gpu(mx)
    data, row, col = _random_coo(1, 64, 64, 800)
    with mx.stream(mx.gpu):
        converted = _buffers(
            mx,
            getattr(
                _coo(mx, data, row, col, (64, 64), "float32", "int32"), "to" + to_format
            )(),
        )
    assert converted.sorted_indices
    indptr = np.array(converted.indptr)
    indices = np.array(converted.indices)
    for start, end in zip(indptr[:-1], indptr[1:]):
        assert np.all(np.diff(indices[start:end]) >= 0)


@pytest.mark.parametrize("to_format", ["csr", "csc"])
def test_conversion_at_a_size_the_ranking_kernel_cannot_reach(
    mx, scipy_sparse, to_format
):
    # Ranking is quadratic in nnz, so this many entries is minutes of GPU time
    # on the kernel path and milliseconds on the sort.
    _require_gpu(mx)
    nnz = 200_000
    n = 20_000
    data, row, col = _random_coo(2, n, n, nnz)
    with mx.stream(mx.gpu):
        coo = _coo(mx, data, row, col, (n, n), "float32", "int32")
        assert coo.nnz == nnz
        # SciPy's conversion sums duplicate coordinates, so ask for the same.
        converted = _buffers(mx, getattr(coo, "to" + to_format)(canonical=True))
    reference = getattr(
        scipy_sparse.coo_array((data, (row, col)), shape=(n, n)), "to" + to_format
    )()
    assert converted.nnz == reference.nnz
    np.testing.assert_array_equal(np.array(converted.indptr), reference.indptr)
    np.testing.assert_array_equal(np.array(converted.indices), reference.indices)
    np.testing.assert_allclose(
        np.array(converted.data), reference.data, rtol=1e-6, atol=1e-6
    )


def test_conversion_needs_no_host_synchronization(mx):
    # The point of doing this with array operations is that nothing has to come
    # back to the host, so the result is still unevaluated when it is returned.
    _require_gpu(mx)
    data, row, col = _random_coo(3, 32, 32, 100)
    with mx.stream(mx.gpu):
        coo = _coo(mx, data, row, col, (32, 32), "float32", "int32")
        mx.eval(coo.data, coo.row, coo.col)
        csr = coo.tocsr()
        dense_from_lazy = csr.todense()
        mx.eval(dense_from_lazy)
    reference = np.zeros((32, 32), dtype=np.float64)
    np.add.at(reference, (row, col), data)
    np.testing.assert_allclose(
        np.array(dense_from_lazy), reference, rtol=1e-5, atol=1e-5
    )
