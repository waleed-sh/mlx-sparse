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

"""A stored index outside the declared shape must not reach the kernels.

``validate="full"`` rejects one before anything runs, but it is not the default
and it reads the indices back to the host, so the conversions and the dense
materializations see them. Each of those has a statically known output shape and
so cannot afford a check of its own; what they can do is decline to write
outside their buffers. The contract is the one the COO conversions already
follow: an entry whose stored index is outside the shape contributes nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as ms

SHAPE = (3, 4)
DATA = [1.0, 2.0, 3.0]
# One entry per row, so dropping the middle one is a shorter indptr.
CSR_INDPTR = [0, 1, 2, 3]
KEPT_INDPTR = [0, 1, 1, 2]

OUT_OF_RANGE = [
    ("negative_one", [0, -1, 3]),
    ("negative_far", [0, -5, 3]),
    ("equals_bound", [0, 4, 3]),
    ("far_past_end", [0, 9999, 3]),
]

DEVICES = [pytest.param(mx.cpu, id="cpu"), pytest.param(mx.gpu, id="gpu")]


def _csr(indices, data=DATA, indptr=CSR_INDPTR, dtype=mx.int32, shape=SHAPE):
    return ms.csr_array(
        (
            mx.array(data, dtype=mx.float32),
            mx.array(indices, dtype=dtype),
            mx.array(indptr, dtype=dtype),
        ),
        shape=shape,
    )


def _csc(indices, data=DATA, indptr=(0, 1, 2, 3, 3), shape=(4, 4)):
    return ms.csc_array(
        (
            mx.array(data, dtype=mx.float32),
            mx.array(indices, dtype=mx.int32),
            mx.array(list(indptr), dtype=mx.int32),
        ),
        shape=shape,
    )


def _parts(array):
    mx.eval(array.data, array.indices, array.indptr)
    return (
        np.asarray(array.data),
        np.asarray(array.indices),
        np.asarray(array.indptr),
    )


# --- csr_tocsc: the third copy of the counting sort ------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,indices", OUT_OF_RANGE, ids=lambda v: v)
def test_tocsc_ignores_an_out_of_range_column(device, label, indices):
    """Equals the conversion of the entries that are in range."""
    kept = [indices[0], indices[2]]
    with mx.stream(device):
        got_data, got_indices, got_indptr = _parts(_csr(indices).tocsc())
        want_data, want_indices, want_indptr = _parts(
            _csr(kept, data=[DATA[0], DATA[2]], indptr=KEPT_INDPTR).tocsc()
        )

    assert got_indptr.tolist() == want_indptr.tolist()
    # The pointer array must not still count the entry that was never placed.
    assert int(got_indptr[-1]) == 2
    assert got_data[:2].tolist() == want_data[:2].tolist()
    assert got_indices[:2].tolist() == want_indices[:2].tolist()
    assert got_data[2:].tolist() == [0.0]
    assert got_indices[2:].tolist() == [0]


@pytest.mark.parametrize("label,indices", OUT_OF_RANGE, ids=lambda v: v)
def test_tocsc_cpu_and_gpu_agree(label, indices):
    with mx.stream(mx.cpu):
        cpu = _parts(_csr(indices).tocsc())
    with mx.stream(mx.gpu):
        gpu = _parts(_csr(indices).tocsc())
    for left, right in zip(cpu, gpu):
        assert left.tolist() == right.tolist()


@pytest.mark.parametrize("device", DEVICES)
def test_tocsc_with_every_column_out_of_range(device):
    with mx.stream(device):
        data, indices, indptr = _parts(_csr([-1, 4, 9999]).tocsc())
    assert indptr.tolist() == [0] * (SHAPE[1] + 1)
    assert data.tolist() == [0.0, 0.0, 0.0]
    assert indices.tolist() == [0, 0, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_tocsc_column_above_int32_range(device):
    """A 64-bit index must be compared at full width, not cast to int first."""
    with mx.stream(device):
        data, _, indptr = _parts(_csr([0, 2**32 + 1, 3], dtype=mx.int64).tocsc())
    assert int(indptr[-1]) == 2
    assert data[:2].tolist() == [1.0, 3.0]


# --- the dense materializations --------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,indices", OUT_OF_RANGE, ids=lambda v: v)
def test_csr_todense_ignores_an_out_of_range_column(device, label, indices):
    kept = [indices[0], indices[2]]
    with mx.stream(device):
        got = _csr(indices).todense()
        want = _csr(kept, data=[DATA[0], DATA[2]], indptr=KEPT_INDPTR).todense()
        mx.eval(got, want)
    assert np.asarray(got).tolist() == np.asarray(want).tolist()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,indices", OUT_OF_RANGE, ids=lambda v: v)
def test_csc_todense_ignores_an_out_of_range_row(device, label, indices):
    """The Metal kernel already checked this; the CPU path did not."""
    kept = [indices[0], indices[2]]
    with mx.stream(device):
        got = _csc(indices).todense()
        want = _csc(kept, data=[DATA[0], DATA[2]], indptr=(0, 1, 1, 2, 2)).todense()
        mx.eval(got, want)
    assert np.asarray(got).tolist() == np.asarray(want).tolist()


@pytest.mark.parametrize("label,indices", OUT_OF_RANGE, ids=lambda v: v)
def test_todense_cpu_and_gpu_agree(label, indices):
    """The two backends disagreed here before, on the same input."""
    with mx.stream(mx.cpu):
        csr_cpu = np.asarray(_csr(indices).todense())
        csc_cpu = np.asarray(_csc(indices).todense())
    with mx.stream(mx.gpu):
        csr_gpu = np.asarray(_csr(indices).todense())
        csc_gpu = np.asarray(_csc(indices).todense())
    assert csr_cpu.tolist() == csr_gpu.tolist()
    assert csc_cpu.tolist() == csc_gpu.tolist()


# --- the guard must not change anything legal ------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("indices", [[0, 1, 3], [0, 0, 0], [3, 3, 3]])
def test_in_range_indices_are_untouched(device, indices):
    """Including the largest legal column on both operations."""
    expected = np.zeros(SHAPE, dtype=np.float32)
    for row, (col, value) in enumerate(zip(indices, DATA)):
        expected[row, col] += value
    with mx.stream(device):
        matrix = _csr(indices)
        dense = matrix.todense()
        through_csc = matrix.tocsc().todense()
        mx.eval(dense, through_csc)
    assert np.asarray(dense).tolist() == expected.tolist()
    assert np.asarray(through_csc).tolist() == expected.tolist()


@pytest.mark.parametrize("device", DEVICES)
def test_a_realistic_matrix_round_trips_unchanged(device):
    """A guard that changed a legal answer would show up here."""
    import scipy.sparse as sp

    reference = sp.random(30, 40, density=0.2, format="csr", random_state=3)
    reference.sort_indices()
    with mx.stream(device):
        matrix = ms.csr_array(
            (
                mx.array(reference.data.astype(np.float32)),
                mx.array(reference.indices.astype(np.int32)),
                mx.array(reference.indptr.astype(np.int32)),
            ),
            shape=(30, 40),
        )
        dense = matrix.todense()
        via_csc = matrix.tocsc().todense()
        mx.eval(dense, via_csc)
    assert np.allclose(np.asarray(dense), reference.toarray(), atol=1e-6)
    assert np.allclose(np.asarray(via_csc), reference.toarray(), atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("op", ["tocsc", "todense"])
def test_a_far_out_of_range_index_does_not_corrupt_memory(device, op):
    """Its own process: the write lands outside the buffer every time, but only
    faults when the heap happens to put an unmapped page in the way."""
    program = f"""
import mlx.core as mx
import mlx_sparse as ms

with mx.stream(mx.{device}):
    for _ in range(256):
        A = ms.csr_array(
            (
                mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
                mx.array([0, 9999, 3], dtype=mx.int32),
                mx.array({CSR_INDPTR}, dtype=mx.int32),
            ),
            shape={SHAPE},
        )
        out = A.{op}()
        mx.eval(out) if {op!r} == "todense" else mx.eval(out.data, out.indptr)
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
