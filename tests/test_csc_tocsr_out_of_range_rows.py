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

"""A CSC row index outside the declared shape must not reach the conversion.

``csc_tocsr`` indexes its output by the row index it reads out of ``indices``.
The conversion's own derivative already returns a zero gradient for a row
outside ``[0, n_rows)``, which is only correct if the conversion dropped that
entry, so the contract asserted here is the one the derivative was written
against: the entry contributes nothing, and the result is the result of
converting the entries that remain.
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
DATA = [1.0, 2.0, 3.0]
# One entry in each of the first three columns, so removing the middle entry is
# just a shorter indptr.
INDPTR = [0, 1, 2, 3, 3, 3]
KEPT_INDPTR = [0, 1, 1, 2, 2, 2]

OUT_OF_RANGE_CASES = [
    ("row_negative_one", [0, -1, 3]),
    ("row_negative_far", [0, -3, 3]),
    ("row_equals_n_rows", [0, 4, 3]),
    ("row_far_past_end", [0, 9999, 3]),
]

IN_RANGE_CASES = [
    ("last_row", [0, 3, 3]),
    ("first_row", [0, 0, 3]),
]

DEVICES = [pytest.param(mx.cpu, id="cpu"), pytest.param(mx.gpu, id="gpu")]


def _csc(indices, data=DATA, indptr=INDPTR, dtype=mx.int32):
    """Build a CSC with the default validation, which does not inspect values."""
    return ms.csc_array(
        (
            mx.array(data, dtype=mx.float32),
            mx.array(indices, dtype=dtype),
            mx.array(indptr, dtype=dtype),
        ),
        shape=SHAPE,
    )


def _parts(array):
    mx.eval(array.data, array.indices, array.indptr)
    return (
        np.asarray(array.data),
        np.asarray(array.indices),
        np.asarray(array.indptr),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,indices", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_tocsr_ignores_an_out_of_range_row(device, label, indices):
    """The conversion equals the conversion of the entries that are in range."""
    kept = [indices[0], indices[2]]
    with mx.stream(device):
        got_data, got_indices, got_indptr = _parts(_csc(indices).tocsr())
        want_data, want_indices, want_indptr = _parts(
            _csc(kept, data=[DATA[0], DATA[2]], indptr=KEPT_INDPTR).tocsr()
        )

    assert got_indptr.tolist() == want_indptr.tolist()
    # The pointer array must not still claim the dropped entry.
    assert int(got_indptr[-1]) == 2
    assert got_data[:2].tolist() == want_data[:2].tolist()
    assert got_indices[:2].tolist() == want_indices[:2].tolist()
    # The slot past the last one any row points at is unreachable through
    # indptr, so it must not expose whatever the allocation came with.
    assert got_data[2:].tolist() == [0.0]
    assert got_indices[2:].tolist() == [0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,indices", IN_RANGE_CASES, ids=lambda v: v)
def test_in_range_rows_are_untouched(device, label, indices):
    """The guard must not change a legal conversion, boundaries included."""
    expected = np.zeros(SHAPE, dtype=np.float32)
    for col, (row, value) in enumerate(zip(indices, DATA)):
        expected[row, col] += value
    with mx.stream(device):
        dense = _csc(indices).tocsr().todense()
        mx.eval(dense)
    assert np.asarray(dense).tolist() == expected.tolist()


@pytest.mark.parametrize("label,indices", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_cpu_and_gpu_agree_on_an_out_of_range_row(label, indices):
    """The two backends must not disagree about what the entry did."""
    with mx.stream(mx.cpu):
        cpu = _parts(_csc(indices).tocsr())
    with mx.stream(mx.gpu):
        gpu = _parts(_csc(indices).tocsr())
    for cpu_part, gpu_part in zip(cpu, gpu):
        assert cpu_part.tolist() == gpu_part.tolist()


@pytest.mark.parametrize("device", DEVICES)
def test_every_row_out_of_range_gives_an_empty_result(device):
    """Nothing survives, and the pointer array describes an empty matrix."""
    with mx.stream(device):
        data, indices, indptr = _parts(_csc([-1, 4, 9999]).tocsr())
    assert indptr.tolist() == [0] * (SHAPE[0] + 1)
    assert data.tolist() == [0.0, 0.0, 0.0]
    assert indices.tolist() == [0, 0, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_a_row_above_int32_range_is_out_of_range(device):
    """A 64-bit row must be compared at full width.

    Casting it to ``int`` first folds ``2**32 + 1`` onto ``1``, which is a
    perfectly ordinary row number, so a narrowing check lets it through.
    """
    with mx.stream(device):
        data, indices, indptr = _parts(
            _csc([0, 2**32 + 1, 3], dtype=mx.int64).tocsr()
        )
    assert int(indptr[-1]) == 2
    assert data[:2].tolist() == [1.0, 3.0]


@pytest.mark.parametrize("label,indices", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_the_gradient_agrees_with_the_conversion_it_differentiates(label, indices):
    """A dropped entry cannot influence the output, so its gradient is zero.

    This is the relation the conversion was violating: the derivative already
    answered zero here while the forward pass was still placing the entry.
    """

    def total(values):
        array = ms.csc_array(
            (
                values,
                mx.array(indices, dtype=mx.int32),
                mx.array(INDPTR, dtype=mx.int32),
            ),
            shape=SHAPE,
        )
        return mx.sum(array.tocsr().data)

    grad = mx.grad(total)(mx.array(DATA, dtype=mx.float32))
    mx.eval(grad)
    assert np.asarray(grad).tolist() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize("label,indices", OUT_OF_RANGE_CASES, ids=lambda v: v)
def test_a_build_without_the_extension_answers_the_same(label, indices):
    """The fallback already drops these entries; the extension must agree."""
    from mlx_sparse import _fallback, _native

    data = mx.array(DATA, dtype=mx.float32)
    index_array = mx.array(indices, dtype=mx.int32)
    indptr_array = mx.array(INDPTR, dtype=mx.int32)

    fallback = _fallback.csc_to_csr(data, index_array, indptr_array, SHAPE)
    with mx.stream(mx.cpu):
        native = _native.csc_tocsr(data, index_array, indptr_array, SHAPE)
        mx.eval(*native)

    for from_fallback, from_extension in zip(fallback, native):
        assert np.asarray(from_fallback).tolist() == np.asarray(from_extension).tolist()


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_a_far_out_of_range_row_does_not_corrupt_memory(device):
    """Run in its own process: the failure this pins is an intermittent fault.

    The write lands outside the allocation every time, but only faults when the
    heap happens to put an unmapped page in the way, so the loop repeats.
    """
    program = f"""
import mlx.core as mx
import mlx_sparse as ms

with mx.stream(mx.{device}):
    for _ in range(256):
        array = ms.csc_array(
            (
                mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
                mx.array([0, 9999, 3], dtype=mx.int32),
                mx.array({INDPTR}, dtype=mx.int32),
            ),
            shape={SHAPE},
        )
        out = array.tocsr()
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
        f"csc_tocsr on the {device} stream exited with {result.returncode} "
        f"(negative means a fatal signal): {result.stderr[-2000:]}"
    )
    assert "survived" in result.stdout
