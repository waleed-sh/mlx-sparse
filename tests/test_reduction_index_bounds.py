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

"""The reductions that scatter into their output by a stored index.

Seven of them do: the COO row and column sums and norms, and the compressed
reductions that run along the minor axis. Their siblings along the major axis
walk indptr and write out[row], never using a stored index as an address, so
they are safe by construction rather than by check and are not covered here.

Two things are pinned. An index outside the declared shape contributes nothing,
which is the contract the conversions already follow. And the check happens in
the index type: an index above INT_MAX must not be narrowed to int first, which
folds it back into range and produces an answer that looks ordinary.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse import _fallback, _native

SHAPE = (4, 4)
DATA = [1.0, 2.0, 3.0]
GOOD = [0, 2]
KEPT_DATA = [DATA[0], DATA[2]]
# One stored entry per row (or column), so dropping the middle one is visible.
CSR_INDPTR = [0, 1, 2, 3, 3]
CSR_KEPT_INDPTR = [0, 1, 1, 2, 2]

OUT_OF_RANGE = [
    ("negative_one", -1),
    ("negative_far", -5),
    ("equals_bound", 4),
    ("far_past_end", 9999),
]

# Folds to 1 under static_cast<int>, which is inside a four-row shape.
ABOVE_INT_MAX = 2**32 + 1
# Far enough out that the write leaves the mapped region every time.
FAR = 2_147_483_000

DEVICES = [pytest.param(mx.cpu, id="cpu"), pytest.param(mx.gpu, id="gpu")]
DTYPES = [
    pytest.param(mx.float32, id="float32"),
    pytest.param(mx.float16, id="float16"),
    pytest.param(mx.bfloat16, id="bfloat16"),
    pytest.param(mx.complex64, id="complex64"),
]


def _coo(row, col, data=DATA, dtype=mx.float32, index_dtype=mx.int32,
         canonical=None):
    return ms.coo_array(
        (
            mx.array(data, dtype=dtype),
            (mx.array(list(row), dtype=index_dtype),
             mx.array(list(col), dtype=index_dtype)),
        ),
        shape=SHAPE,
        canonical=canonical,
    )


def _csc(indices, data=DATA, indptr=CSR_INDPTR, dtype=mx.float32,
         index_dtype=mx.int32):
    return ms.csc_array(
        (
            mx.array(data, dtype=dtype),
            mx.array(list(indices), dtype=index_dtype),
            mx.array(list(indptr), dtype=index_dtype),
        ),
        shape=SHAPE,
    )


def _csr(indices, data=DATA, indptr=CSR_INDPTR, dtype=mx.float32,
         index_dtype=mx.int32):
    return ms.csr_array(
        (
            mx.array(data, dtype=dtype),
            mx.array(list(indices), dtype=index_dtype),
            mx.array(list(indptr), dtype=index_dtype),
        ),
        shape=SHAPE,
    )


def _values(array):
    mx.eval(array)
    return np.asarray(array.astype(mx.float32)).ravel().tolist()


# Each entry places the bad index on the axis the reduction scatters along.
# ``canonical=True`` is what reaches the two COO norms kernels: without it the
# operation routes through tocsr(canonical=True) and never calls them, which is
# how a first reading of these mistook the route for the code.
OPS = {
    "coo_row_sums": lambda bad, **kw: _coo([0, bad, 2], [0, 1, 2], **kw).row_sums(),
    "coo_col_sums": lambda bad, **kw: _coo([0, 1, 2], [0, bad, 2], **kw).col_sums(),
    "coo_row_norms": lambda bad, **kw: _coo(
        [0, bad, 2], [0, 1, 2], canonical=True, **kw).row_norms(),
    "coo_col_norms": lambda bad, **kw: _coo(
        [0, 1, 2], [0, bad, 2], canonical=True, **kw).col_norms(),
    "csc_row_sums": lambda bad, **kw: _csc([0, bad, 2], **kw).row_sums(),
    "csc_row_norms": lambda bad, **kw: _csc([0, bad, 2], **kw).row_norms(),
    "csr_col_sums": lambda bad, **kw: _csr([0, bad, 2], **kw).col_sums(),
}

KEPT = {
    "coo_row_sums": lambda **kw: _coo(GOOD, [0, 2], data=KEPT_DATA, **kw).row_sums(),
    "coo_col_sums": lambda **kw: _coo([0, 2], GOOD, data=KEPT_DATA, **kw).col_sums(),
    "coo_row_norms": lambda **kw: _coo(
        GOOD, [0, 2], data=KEPT_DATA, canonical=True, **kw).row_norms(),
    "coo_col_norms": lambda **kw: _coo(
        [0, 2], GOOD, data=KEPT_DATA, canonical=True, **kw).col_norms(),
    "csc_row_sums": lambda **kw: _csc(
        GOOD, data=KEPT_DATA, indptr=CSR_KEPT_INDPTR, **kw).row_sums(),
    "csc_row_norms": lambda **kw: _csc(
        GOOD, data=KEPT_DATA, indptr=CSR_KEPT_INDPTR, **kw).row_norms(),
    "csr_col_sums": lambda **kw: _csr(
        GOOD, data=KEPT_DATA, indptr=CSR_KEPT_INDPTR, **kw).col_sums(),
}

NAMES = sorted(OPS)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,bad", OUT_OF_RANGE, ids=lambda v: v)
def test_a_reduction_ignores_an_out_of_range_index(name, device, label, bad):
    """Equals the reduction over the entries that are in range."""
    with mx.stream(device):
        got = _values(OPS[name](bad))
        want = _values(KEPT[name]())
    assert got == want


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("label,bad", OUT_OF_RANGE, ids=lambda v: v)
def test_a_reduction_agrees_between_cpu_and_gpu(name, label, bad):
    with mx.stream(mx.cpu):
        cpu = _values(OPS[name](bad))
    with mx.stream(mx.gpu):
        gpu = _values(OPS[name](bad))
    assert cpu == gpu


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_a_reduction_compares_the_index_at_full_width(name, device):
    """An index above INT_MAX must not be narrowed before the comparison.

    Narrowed, 2**32 + 1 becomes 1, which is a legal row of this shape, so the
    entry is accumulated into it and the result is indistinguishable from the
    answer for a matrix that really did store it there.
    """
    with mx.stream(device):
        got = _values(OPS[name](ABOVE_INT_MAX, index_dtype=mx.int64))
        want = _values(KEPT[name](index_dtype=mx.int64))
    assert got == want


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_reduction_ignores_an_out_of_range_index_for_every_dtype(
    name, device, dtype
):
    """The accumulator, and on the GPU the kernel, differ by value dtype."""
    if "norms" in name and dtype in (mx.float16, mx.bfloat16):
        pytest.skip("the norms accumulate in float64 and take float32 input")
    with mx.stream(device):
        got = _values(OPS[name](9999, dtype=dtype))
        want = _values(KEPT[name](dtype=dtype))
    assert got == want


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_every_index_out_of_range_reduces_to_nothing(name, device):
    with mx.stream(device):
        if name.startswith("coo_row"):
            matrix = _coo([9999, 4, -1], [0, 1, 2], canonical=True)
            got = _values(matrix.row_norms() if "norms" in name
                          else matrix.row_sums())
        elif name.startswith("coo_col"):
            matrix = _coo([0, 1, 2], [9999, 4, -1], canonical=True)
            got = _values(matrix.col_norms() if "norms" in name
                          else matrix.col_sums())
        elif name.startswith("csc"):
            matrix = _csc([9999, 4, -1])
            got = _values(matrix.row_norms() if "norms" in name
                          else matrix.row_sums())
        else:
            got = _values(_csr([9999, 4, -1]).col_sums())
    assert got == [0.0] * len(got)


# --- the negative control: a guard must not change a legal answer ----------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [0, 1, 3])
def test_a_legal_index_is_untouched(name, device, bad):
    """Including the largest legal index, which the check must still accept."""
    with mx.stream(device):
        got = _values(OPS[name](bad))
    if "norms" in name:
        expected_total = sum(v * v for v in DATA)
        assert sum(v * v for v in got) == pytest.approx(expected_total, rel=1e-5)
    else:
        assert sum(got) == pytest.approx(sum(DATA), rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_a_realistic_reduction_is_unchanged(device):
    import scipy.sparse as sp

    reference = sp.random(30, 24, density=0.25, format="csr", random_state=5)
    reference.sort_indices()
    with mx.stream(device):
        matrix = ms.csr_array(
            (
                mx.array(reference.data.astype(np.float32)),
                mx.array(reference.indices.astype(np.int32)),
                mx.array(reference.indptr.astype(np.int32)),
            ),
            shape=(30, 24),
        )
        got = _values(matrix.col_sums())
    assert np.allclose(got, np.asarray(reference.sum(axis=0)).ravel(), atol=1e-5)


# --- the COO reductions must agree with the conversion spelling ------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,bad", OUT_OF_RANGE, ids=lambda v: v)
def test_a_coo_reduction_agrees_with_the_conversion(device, label, bad):
    """The bad index is on the OTHER axis from the one being reduced.

    The conversion drops such an entry because it is outside the shape, so a
    reduction that kept it would make two spellings of one computation
    disagree.
    """
    with mx.stream(device):
        direct = _values(_coo([0, 1, 2], [0, bad, 2]).row_sums())
        converted = _values(_coo([0, 1, 2], [0, bad, 2]).tocsr().row_sums())
        direct_col = _values(_coo([0, bad, 2], [0, 1, 2]).col_sums())
        converted_col = _values(_coo([0, bad, 2], [0, 1, 2]).tocsc().col_sums())
    assert direct == converted
    assert direct_col == converted_col
    # Agreement alone would also hold if both sides kept the entry, so pin the
    # value the two of them have to agree ON.
    assert direct == [DATA[0], 0.0, DATA[2], 0.0]
    assert direct_col == [DATA[0], 0.0, DATA[2], 0.0]


@pytest.mark.parametrize("label,bad", OUT_OF_RANGE, ids=lambda v: v)
def test_a_coo_reduction_does_not_depend_on_the_value_dtype(label, bad):
    """coo_row_sums takes a conversion detour on the GPU off float32.

    The two branches have to answer the same, or the result depends on the
    dtype rather than on the matrix.
    """
    results = []
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        with mx.stream(mx.gpu):
            results.append(_values(_coo([0, 1, 2], [0, bad, 2], dtype=dtype).row_sums()))
    assert results[0] == results[1] == results[2]
    # And they have to agree on the right answer: if neither branch checked,
    # all three would agree on the wrong one.
    assert results[0] == [DATA[0], 0.0, DATA[2], 0.0]


# --- the extension-less build has to answer the same -----------------------


@pytest.mark.parametrize("label,bad", OUT_OF_RANGE, ids=lambda v: v)
def test_a_build_without_the_extension_answers_the_same(label, bad):
    """Otherwise the answer depends on how the package was built."""
    data = mx.array(DATA, dtype=mx.float32)
    row = mx.array([0, bad, 2], dtype=mx.int32)
    col = mx.array([0, 1, 2], dtype=mx.int32)
    good_col = mx.array([0, 1, 2], dtype=mx.int32)
    indptr = mx.array(CSR_INDPTR, dtype=mx.int32)

    with mx.stream(mx.cpu):
        pairs = [
            (_fallback.coo_row_sums(data, row, good_col, SHAPE),
             _native.coo_row_sums(data, row, good_col, SHAPE)),
            (_fallback.coo_col_sums(data, col, row, SHAPE),
             _native.coo_col_sums(data, col, row, SHAPE)),
            (_fallback.csc_row_sums(data, row, indptr, SHAPE),
             _native.csc_row_sums(data, row, indptr, SHAPE)),
            (_fallback.csr_col_sums(data, row, indptr, SHAPE),
             _native.csr_col_sums(data, row, indptr, SHAPE)),
            (_fallback.csc_row_norms(data, row, indptr, SHAPE),
             _native.csc_row_norms(data, row, indptr, SHAPE)),
        ]
        mx.eval(*[v for pair in pairs for v in pair])
    for from_fallback, from_extension in pairs:
        assert np.asarray(from_fallback).tolist() == pytest.approx(
            np.asarray(from_extension).tolist(), rel=1e-6
        )


# --- the routing fact that made a first reading of these look clean --------


def test_the_coo_norms_reach_their_kernel_only_when_told_it_is_canonical():
    """Pinned because it is how a first audit mistook the route for the code.

    Without the hint the operation converts first, and the conversion's own
    check hides whether the kernel has one. The two paths must now agree, which
    is the point, but the routing itself is worth stating.
    """
    with mx.stream(mx.gpu):
        via_conversion = _values(_coo([0, 9999, 2], [0, 1, 2]).row_norms())
        via_kernel = _values(
            _coo([0, 9999, 2], [0, 1, 2], canonical=True).row_norms()
        )
    assert via_conversion == via_kernel
    assert not _coo([0, 9999, 2], [0, 1, 2]).has_canonical_format
    assert _coo([0, 9999, 2], [0, 1, 2], canonical=True).has_canonical_format


# --- memory safety, which a returned value cannot show ---------------------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_a_far_out_of_range_index_does_not_corrupt_memory(name, device):
    """Its own process: an index this far out leaves the mapped region."""
    program = f"""
import numpy as np
import mlx.core as mx
import mlx_sparse as ms

SHAPE = {SHAPE!r}
DATA = {DATA!r}
INDPTR = {CSR_INDPTR!r}
bad = {FAR}

def coo(row, col, canonical=None):
    return ms.coo_array(
        (mx.array(DATA, dtype=mx.float32),
         (mx.array(row, dtype=mx.int64), mx.array(col, dtype=mx.int64))),
        shape=SHAPE, canonical=canonical)

def csc(indices):
    return ms.csc_array(
        (mx.array(DATA, dtype=mx.float32), mx.array(indices, dtype=mx.int64),
         mx.array(INDPTR, dtype=mx.int64)), shape=SHAPE)

def csr(indices):
    return ms.csr_array(
        (mx.array(DATA, dtype=mx.float32), mx.array(indices, dtype=mx.int64),
         mx.array(INDPTR, dtype=mx.int64)), shape=SHAPE)

OPS = {{
 "coo_row_sums":  lambda: coo([0, bad, 2], [0, 1, 2]).row_sums(),
 "coo_col_sums":  lambda: coo([0, 1, 2], [0, bad, 2]).col_sums(),
 "coo_row_norms": lambda: coo([0, bad, 2], [0, 1, 2], True).row_norms(),
 "coo_col_norms": lambda: coo([0, 1, 2], [0, bad, 2], True).col_norms(),
 "csc_row_sums":  lambda: csc([0, bad, 2]).row_sums(),
 "csc_row_norms": lambda: csc([0, bad, 2]).row_norms(),
 "csr_col_sums":  lambda: csr([0, bad, 2]).col_sums(),
}}

with mx.stream(mx.{device}):
    for _ in range(64):
        out = OPS[{name!r}]()
        mx.eval(out)
print("survived", np.asarray(out).ravel().tolist())
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and os.path.isdir(p))
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{name} on the {device} stream exited with {result.returncode} "
        f"(negative means a fatal signal): {result.stderr[-2000:]}"
    )
    assert "survived" in result.stdout
