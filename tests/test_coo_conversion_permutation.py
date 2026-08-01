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

"""``tocsr``/``tocsc`` can hand back the permutation they applied.

A caller who stores several arrays against one set of coordinates -- extra
value columns, a presence mask, an edge label -- has to reorder all of them the
same way the conversion reordered ``data``. Without the permutation the only
option is to redo the sort by hand and hope it ties duplicates identically.
"""

import numpy as np
import pytest

import mlx_sparse as ms

VALUE_DTYPES = ["float32", "float16", "bfloat16"]
STREAMS = ["cpu", "gpu"]


def _stream(mx, name):
    if name == "gpu":
        try:
            available = mx.is_available(mx.Device(mx.gpu, 0))
        except Exception:  # pragma: no cover - platform dependent
            available = False
        if not available:
            pytest.skip("No Metal device available.")
        return mx.stream(mx.gpu)
    return mx.stream(mx.cpu)


def _random_coo(seed, n_rows, n_cols, nnz, repeat_span=None):
    rng = np.random.default_rng(seed)
    high_rows = repeat_span if repeat_span else n_rows
    high_cols = repeat_span if repeat_span else n_cols
    return (
        rng.standard_normal(nnz).astype(np.float32),
        rng.integers(0, high_rows, size=nnz).astype(np.int32),
        rng.integers(0, high_cols, size=nnz).astype(np.int32),
    )


def _coo(mx, data, row, col, shape):
    return ms.coo_array((mx.array(data), (mx.array(row), mx.array(col))), shape=shape)


@pytest.mark.parametrize("stream_name", STREAMS)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("to_format", ["csr", "csc"])
def test_permutation_reproduces_the_converted_buffers(
    mx, stream_name, value_dtype, to_format
):
    data, row, col = _random_coo(0, 23, 29, 300)
    shape = (23, 29)
    with _stream(mx, stream_name):
        coo = ms.coo_array(
            (
                mx.array(data).astype(getattr(mx, value_dtype)),
                (mx.array(row), mx.array(col)),
            ),
            shape=shape,
        )
        converted, order = getattr(coo, "to" + to_format)(return_permutation=True)
        minor = coo.col if to_format == "csr" else coo.row
        gathered_data = mx.take(coo.data, order)
        gathered_indices = mx.take(minor, order)
        mx.eval(converted.data, converted.indices, gathered_data, gathered_indices)

    assert order.shape == (coo.nnz,)
    np.testing.assert_array_equal(
        np.array(converted.data.astype(mx.float32)),
        np.array(gathered_data.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(converted.indices), np.array(gathered_indices)
    )


@pytest.mark.parametrize("stream_name", STREAMS)
def test_parallel_columns_and_a_presence_mask_ride_the_same_permutation(
    mx, stream_name
):
    # One coordinate list, three arrays hanging off it. Duplicate coordinates
    # are included so a hand-rolled sort with a different tie-break would be
    # caught here.
    data, row, col = _random_coo(1, 12, 12, 200, repeat_span=6)
    weight = np.arange(row.size, dtype=np.float32) * 0.5
    present = (np.arange(row.size) % 3 != 0).astype(np.bool_)

    with _stream(mx, stream_name):
        coo = _coo(mx, data, row, col, (12, 12))
        csr, order = coo.tocsr(return_permutation=True)
        weight_csr = mx.take(mx.array(weight), order)
        present_csr = mx.take(mx.array(present), order)
        mx.eval(csr.data, csr.indices, csr.indptr, weight_csr, present_csr)

    reference = np.lexsort((np.arange(row.size), col, row))
    np.testing.assert_array_equal(np.array(order), reference)
    np.testing.assert_array_equal(np.array(weight_csr), weight[reference])
    np.testing.assert_array_equal(np.array(present_csr), present[reference])

    # Every stored entry still describes the coordinate it came from.
    indptr = np.array(csr.indptr)
    rows_of_entries = np.repeat(np.arange(12), np.diff(indptr))
    np.testing.assert_array_equal(rows_of_entries, row[reference])
    np.testing.assert_array_equal(np.array(csr.indices), col[reference])


@pytest.mark.parametrize("stream_name", STREAMS)
@pytest.mark.parametrize("to_format", ["csr", "csc"])
def test_permutation_is_the_same_order_the_plain_conversion_uses(
    mx, stream_name, to_format
):
    data, row, col = _random_coo(2, 9, 9, 120, repeat_span=4)
    with _stream(mx, stream_name):
        coo = _coo(mx, data, row, col, (9, 9))
        plain = getattr(coo, "to" + to_format)()
        with_order, order = getattr(coo, "to" + to_format)(return_permutation=True)
        mx.eval(plain.data, plain.indices, with_order.data, with_order.indices, order)
    np.testing.assert_array_equal(np.array(plain.data), np.array(with_order.data))
    np.testing.assert_array_equal(np.array(plain.indices), np.array(with_order.indices))
    assert sorted(np.array(order).tolist()) == list(range(coo.nnz))


@pytest.mark.parametrize("to_format", ["csr", "csc"])
def test_canonical_and_return_permutation_are_mutually_exclusive(mx, to_format):
    data, row, col = _random_coo(3, 5, 5, 20)
    coo = _coo(mx, data, row, col, (5, 5))
    with pytest.raises(ValueError, match="not a permutation"):
        getattr(coo, "to" + to_format)(canonical=True, return_permutation=True)


@pytest.mark.parametrize("stream_name", STREAMS)
def test_permutation_of_an_empty_matrix(mx, stream_name):
    with _stream(mx, stream_name):
        coo = ms.coo_array(
            (
                mx.zeros((0,), dtype=mx.float32),
                (mx.zeros((0,), dtype=mx.int32), mx.zeros((0,), dtype=mx.int32)),
            ),
            shape=(3, 3),
        )
        csr, order = coo.tocsr(return_permutation=True)
        mx.eval(csr.indptr, order)
    assert order.shape == (0,)
    assert csr.nnz == 0


def test_two_arrays_share_one_set_of_index_buffers(mx):
    # The permutation exists so that several value arrays can sit on one
    # converted structure. Building those arrays must not copy the structure.
    data, row, col = _random_coo(4, 16, 16, 150)
    coo = _coo(mx, data, row, col, (16, 16))
    csr, order = coo.tocsr(return_permutation=True)

    second_values = mx.take(mx.arange(row.size, dtype=mx.float32), order)
    second = ms.csr_array(
        (second_values, csr.indices, csr.indptr),
        shape=csr.shape,
        sorted_indices=True,
    )

    assert second.indices is csr.indices
    assert second.indptr is csr.indptr

    x = mx.ones((16,), dtype=mx.float32)
    mx.eval(csr @ x, second @ x)
    reference_order = np.lexsort((np.arange(row.size), col, row))
    dense_first = np.zeros((16, 16))
    dense_second = np.zeros((16, 16))
    np.add.at(dense_first, (row, col), data)
    np.add.at(
        dense_second,
        (row[reference_order], col[reference_order]),
        np.arange(row.size, dtype=np.float32)[reference_order],
    )
    np.testing.assert_allclose(
        np.array(csr @ x), dense_first.sum(axis=1), rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        np.array(second @ x), dense_second.sum(axis=1), rtol=1e-5, atol=1e-4
    )
