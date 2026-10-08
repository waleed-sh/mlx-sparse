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

"""Coordinates outside the declared shape must not reach the kernels.

``validate="full"`` rejects such coordinates before anything runs, but it is not
the default and it cannot be used inside ``mx.compile``, so the kernels see them.
The contract asserted here is that an entry whose row or column is outside the
declared shape contributes nothing: the result equals the result of passing the
remaining entries on their own, on either backend.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as ms

SHAPE = (4, 5)

# (label, row, col) where exactly the middle entry is out of the declared shape.
OUT_OF_RANGE_CASES = [
    ("row_negative_one", [0, -1, 3], [1, 2, 4]),
    ("row_negative_far", [0, -3, 3], [1, 2, 4]),
    ("row_equals_n_rows", [0, 4, 3], [1, 2, 4]),
    ("row_far_past_end", [0, 9999, 3], [1, 2, 4]),
    ("col_negative_one", [0, 2, 3], [1, -1, 4]),
    ("col_negative_far", [0, 2, 3], [1, -3, 4]),
    ("col_equals_n_cols", [0, 2, 3], [1, 5, 4]),
    ("col_far_past_end", [0, 2, 3], [1, 9999, 4]),
    ("both_out_of_range", [0, -7, 3], [1, 9999, 4]),
]

# The largest legal coordinate on each axis, which must keep working.
IN_RANGE_CASES = [
    ("last_row_and_col", [0, 3, 3], [1, 4, 4]),
    ("first_row_and_col", [0, 0, 3], [1, 0, 4]),
]

DATA = [1.0, 2.0, 3.0]

DEVICES = [pytest.param(mx.cpu, id="cpu"), pytest.param(mx.gpu, id="gpu")]


def _coo(row, col, data, *, dtype=mx.float32, shape=SHAPE):
    """Build a COO with the default validation, which does not inspect values."""
    return ms.coo_array(
        (
            mx.array(data, dtype=dtype),
            (mx.array(row, dtype=mx.int32), mx.array(col, dtype=mx.int32)),
        ),
        shape=shape,
    )


def _without_middle(row, col, data):
    """The same entries with the offending middle one removed."""
    return [row[0], row[2]], [col[0], col[2]], [data[0], data[2]]


def _csr_parts(array):
    mx.eval(array.data, array.indices, array.indptr)
    return (
        np.asarray(array.data),
        np.asarray(array.indices),
        np.asarray(array.indptr),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,row,col", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_tocsr_ignores_an_out_of_range_entry(device, label, row, col):
    """CSR conversion equals the conversion of the entries that are in range."""
    kept_row, kept_col, kept_data = _without_middle(row, col, DATA)
    with mx.stream(device):
        got_data, got_indices, got_indptr = _csr_parts(_coo(row, col, DATA).tocsr())
        want = _csr_parts(_coo(kept_row, kept_col, kept_data).tocsr())
    want_data, want_indices, want_indptr = want

    assert got_indptr.tolist() == want_indptr.tolist()
    kept = int(want_indptr[-1])
    assert got_data[:kept].tolist() == want_data[:kept].tolist()
    assert got_indices[:kept].tolist() == want_indices[:kept].tolist()
    # The slots past the last referenced entry are unreachable through indptr,
    # so they must not expose whatever happened to be in the allocation.
    assert got_data[kept:].tolist() == [0.0] * (len(DATA) - kept)
    assert got_indices[kept:].tolist() == [0] * (len(DATA) - kept)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,row,col", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_tocsc_ignores_an_out_of_range_entry(device, label, row, col):
    """CSC conversion equals the conversion of the entries that are in range."""
    kept_row, kept_col, kept_data = _without_middle(row, col, DATA)
    with mx.stream(device):
        got_data, got_indices, got_indptr = _csr_parts(_coo(row, col, DATA).tocsc())
        want = _csr_parts(_coo(kept_row, kept_col, kept_data).tocsc())
    want_data, want_indices, want_indptr = want

    assert got_indptr.tolist() == want_indptr.tolist()
    kept = int(want_indptr[-1])
    assert got_data[:kept].tolist() == want_data[:kept].tolist()
    assert got_indices[:kept].tolist() == want_indices[:kept].tolist()
    assert got_data[kept:].tolist() == [0.0] * (len(DATA) - kept)
    assert got_indices[kept:].tolist() == [0] * (len(DATA) - kept)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,row,col", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_todense_ignores_an_out_of_range_entry(device, label, row, col):
    """A dense materialization must not place the entry anywhere at all."""
    kept_row, kept_col, kept_data = _without_middle(row, col, DATA)
    with mx.stream(device):
        got = _coo(row, col, DATA).todense()
        want = _coo(kept_row, kept_col, kept_data).todense()
        mx.eval(got, want)
    assert np.asarray(got).tolist() == np.asarray(want).tolist()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,row,col", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_matmul_ignores_an_out_of_range_entry(device, label, row, col):
    """A product must not read a right-hand side row outside the operand."""
    kept_row, kept_col, kept_data = _without_middle(row, col, DATA)
    rhs_values = np.arange(SHAPE[1] * 2, dtype=np.float32).reshape(SHAPE[1], 2)
    with mx.stream(device):
        rhs = mx.array(rhs_values)
        got = _coo(row, col, DATA) @ rhs
        want = _coo(kept_row, kept_col, kept_data) @ rhs
        mx.eval(got, want)
    assert np.asarray(got).tolist() == np.asarray(want).tolist()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,row,col", IN_RANGE_CASES, ids=lambda v: v)
def test_in_range_coordinates_are_untouched(device, label, row, col):
    """The guard must not change any legal conversion, boundaries included."""
    expected = np.zeros(SHAPE, dtype=np.float32)
    for r, c, v in zip(row, col, DATA):
        expected[r, c] += v
    with mx.stream(device):
        coo = _coo(row, col, DATA)
        dense = coo.todense()
        csr_dense = coo.tocsr().todense()
        csc_dense = coo.tocsc().todense()
        mx.eval(dense, csr_dense, csc_dense)
    assert np.asarray(dense).tolist() == expected.tolist()
    assert np.asarray(csr_dense).tolist() == expected.tolist()
    assert np.asarray(csc_dense).tolist() == expected.tolist()


@pytest.mark.parametrize("label,row,col", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_cpu_and_gpu_agree_on_an_out_of_range_entry(label, row, col):
    """The two backends must not disagree about what the entry did."""
    with mx.stream(mx.cpu):
        cpu = _csr_parts(_coo(row, col, DATA).tocsr())
    with mx.stream(mx.gpu):
        gpu = _csr_parts(_coo(row, col, DATA).tocsr())
    for cpu_part, gpu_part in zip(cpu, gpu):
        assert cpu_part.tolist() == gpu_part.tolist()


@pytest.mark.parametrize("device", DEVICES)
def test_every_entry_out_of_range_gives_an_empty_result(device):
    """Nothing survives, and the pointer array still describes an empty matrix."""
    with mx.stream(device):
        data, indices, indptr = _csr_parts(_coo([-1, 4, 9999], [0, 1, 2], DATA).tocsr())
    assert indptr.tolist() == [0] * (SHAPE[0] + 1)
    assert data.tolist() == [0.0, 0.0, 0.0]
    assert indices.tolist() == [0, 0, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_complex64_takes_the_kernel_path_and_is_guarded_too(device):
    """complex64 declines the array-op conversion, so it exercises the kernel."""
    values = [1 + 1j, 2 + 2j, 3 + 3j]
    with mx.stream(device):
        got = _csr_parts(
            _coo([0, 9999, 3], [1, 2, 4], values, dtype=mx.complex64).tocsr()
        )
        want = _csr_parts(
            _coo([0, 3], [1, 4], [1 + 1j, 3 + 3j], dtype=mx.complex64).tocsr()
        )
    assert got[2].tolist() == want[2].tolist()
    kept = int(want[2][-1])
    assert got[0][:kept].tolist() == want[0][:kept].tolist()
    assert got[1][:kept].tolist() == want[1][:kept].tolist()


def test_the_gradient_agrees_with_the_conversion_it_differentiates():
    """A dropped entry cannot influence the output, so its gradient is zero."""

    def total(values):
        coo = ms.coo_array(
            (
                values,
                (
                    mx.array([0, -1, 3], dtype=mx.int32),
                    mx.array([1, 9999, 4], dtype=mx.int32),
                ),
            ),
            shape=SHAPE,
        )
        return mx.sum(coo.tocsr().data)

    grad = mx.grad(total)(mx.array(DATA, dtype=mx.float32))
    mx.eval(grad)
    assert np.asarray(grad).tolist() == [1.0, 0.0, 1.0]


def test_the_guard_does_not_introduce_a_host_synchronization():
    """A device-side guard must leave the conversion usable under mx.compile."""

    @mx.compile
    def convert(values, row, col):
        coo = ms.coo_array((values, (row, col)), shape=SHAPE)
        csr = coo.tocsr()
        return csr.data, csr.indptr

    data, indptr = convert(
        mx.array(DATA, dtype=mx.float32),
        mx.array([0, -1, 3], dtype=mx.int32),
        mx.array([1, 2, 4], dtype=mx.int32),
    )
    mx.eval(data, indptr)
    assert np.asarray(indptr).tolist() == [0, 1, 1, 1, 2]


@pytest.mark.parametrize("op", ["coo_to_csr", "coo_to_csc"])
@pytest.mark.parametrize("label,row,col", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_a_build_without_the_extension_answers_the_same(op, label, row, col):
    """The pure-array fallback has to agree, or the answer depends on the build."""
    from mlx_sparse import _fallback, _native

    data = mx.array(DATA, dtype=mx.float32)
    row_array = mx.array(row, dtype=mx.int32)
    col_array = mx.array(col, dtype=mx.int32)

    fallback = getattr(_fallback, op)(data, row_array, col_array, SHAPE)
    native_op = "coo_tocsr" if op == "coo_to_csr" else "coo_tocsc"
    with mx.stream(mx.cpu):
        native = getattr(_native, native_op)(data, row_array, col_array, SHAPE)
        mx.eval(*native)

    for from_fallback, from_extension in zip(fallback, native):
        assert np.asarray(from_fallback).tolist() == np.asarray(from_extension).tolist()


@pytest.mark.parametrize("label,row,col", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_a_build_without_the_extension_materializes_the_same(label, row, col):
    """The same agreement for the dense and product paths."""
    from mlx_sparse import _fallback

    data = mx.array(DATA, dtype=mx.float32)
    row_array = mx.array(row, dtype=mx.int32)
    col_array = mx.array(col, dtype=mx.int32)
    rhs = mx.array(np.arange(SHAPE[1] * 2, dtype=np.float32).reshape(SHAPE[1], 2))
    vector = mx.array(np.arange(SHAPE[1], dtype=np.float32))

    with mx.stream(mx.cpu):
        dense = _coo(row, col, DATA).todense()
        product = _coo(row, col, DATA) @ rhs
        mat_vector = _coo(row, col, DATA) @ vector
        mx.eval(dense, product, mat_vector)

    assert (
        np.asarray(_fallback.coo_todense(data, row_array, col_array, SHAPE)).tolist()
        == np.asarray(dense).tolist()
    )
    assert (
        np.asarray(
            _fallback.coo_matmul(data, row_array, col_array, rhs, SHAPE)
        ).tolist()
        == np.asarray(product).tolist()
    )
    assert (
        np.asarray(
            _fallback.coo_matvec(data, row_array, col_array, vector, SHAPE)
        ).tolist()
        == np.asarray(mat_vector).tolist()
    )


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("op", ["tocsr", "tocsc"])
def test_a_far_out_of_range_coordinate_does_not_corrupt_memory(device, op):
    """Run the conversion in its own process, because the failure is a crash."""
    coordinate = "row" if op == "tocsr" else "col"
    program = f"""
import mlx.core as mx
import mlx_sparse as ms

bad = [0, 9999, 3]
good = [1, 2, 4]
row, col = (bad, good) if {coordinate!r} == "row" else (good, bad)
with mx.stream(mx.{device}):
    for _ in range(64):
        coo = ms.coo_array(
            (
                mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
                (mx.array(row, dtype=mx.int32), mx.array(col, dtype=mx.int32)),
            ),
            shape={SHAPE},
        )
        out = coo.{op}()
        mx.eval(out.data, out.indices, out.indptr)
print("survived")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{op} on the {device} stream exited with {result.returncode} "
        f"(negative means a fatal signal): {result.stderr[-2000:]}"
    )
    assert "survived" in result.stdout
