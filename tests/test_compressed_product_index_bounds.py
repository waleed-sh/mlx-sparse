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

"""The products must not follow a stored index outside the declared shape.

Every one of these kernels needs exactly one number, and it is always the minor
dimension: columns for CSR, rows for CSC. The index addresses the minor axis, so
that is the bound, and the same predicate serves all of them.

Same contract as the conversions and the dense materializations: an entry whose
stored index is outside the shape contributes nothing. ``validate="full"``
remains the place that raises.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as ms
import mlx_sparse._native as native

CSR_SHAPE = (3, 4)
CSC_SHAPE = (4, 4)
DATA = [1.0, 2.0, 3.0]
# One entry per row (or column), so dropping the middle one is a shorter indptr.
CSR_INDPTR = [0, 1, 2, 3]
CSR_KEPT_INDPTR = [0, 1, 1, 2]
CSC_INDPTR = [0, 1, 2, 3, 3]
CSC_KEPT_INDPTR = [0, 1, 1, 2, 2]
KEPT_DATA = [DATA[0], DATA[2]]

OUT_OF_RANGE = [
    ("negative_one", [0, -1, 3]),
    ("negative_far", [0, -5, 3]),
    ("equals_bound", [0, 4, 3]),
    ("far_past_end", [0, 9999, 3]),
]

# Far enough out that the write or read leaves the mapped region every time. A
# smaller index lands inside a page that happens to be mapped and looks
# harmless, which is why the moderate cases above are not the whole battery.
FAR = 2_147_483_000

DEVICES = [pytest.param(mx.cpu, id="cpu"), pytest.param(mx.gpu, id="gpu")]
DTYPES = [
    pytest.param(mx.float32, id="float32"),
    pytest.param(mx.float16, id="float16"),
    pytest.param(mx.bfloat16, id="bfloat16"),
    pytest.param(mx.complex64, id="complex64"),
]


def _csr(indices, data=DATA, indptr=CSR_INDPTR, dtype=mx.float32,
         index_dtype=mx.int32, shape=CSR_SHAPE):
    return ms.csr_array(
        (
            mx.array(data, dtype=dtype),
            mx.array(list(indices), dtype=index_dtype),
            mx.array(list(indptr), dtype=index_dtype),
        ),
        shape=shape,
    )


def _csc(indices, data=DATA, indptr=CSC_INDPTR, dtype=mx.float32,
         index_dtype=mx.int32, shape=CSC_SHAPE):
    return ms.csc_array(
        (
            mx.array(data, dtype=dtype),
            mx.array(list(indices), dtype=index_dtype),
            mx.array(list(indptr), dtype=index_dtype),
        ),
        shape=shape,
    )


def _ones(shape, dtype):
    return mx.ones(shape, dtype=dtype)


# Each entry builds the matrix from a list of stored indices and applies the
# product. Four of these read an operand by the stored index and four write an
# output by it; the last three are the batched variants.
def _csr_op(fn):
    def build(indices, dtype, indptr=CSR_INDPTR, data=DATA, index_dtype=mx.int32):
        return fn(_csr(indices, data=data, indptr=indptr, dtype=dtype,
                       index_dtype=index_dtype), dtype)

    return build


def _csc_op(fn):
    def build(indices, dtype, indptr=CSC_INDPTR, data=DATA, index_dtype=mx.int32):
        return fn(_csc(indices, data=data, indptr=indptr, dtype=dtype,
                       index_dtype=index_dtype), dtype)

    return build


OPS = {
    # reads an operand by the stored index
    "csr_matvec": _csr_op(lambda a, dt: ms.csr_matvec(a, _ones((4,), dt))),
    "csc_matvec_transpose": _csc_op(
        lambda a, dt: ms.csc_matvec_transpose(a, _ones((4,), dt))
    ),
    "csr_matmul": _csr_op(lambda a, dt: ms.csr_matmul(a, _ones((4, 2), dt))),
    "csc_matmul_transpose": _csc_op(
        lambda a, dt: native.csc_matmul_transpose(
            a.data, a.indices, a.indptr, _ones((4, 2), dt), CSC_SHAPE
        )
    ),
    # writes an output by the stored index
    "csc_matvec": _csc_op(lambda a, dt: ms.csc_matvec(a, _ones((4,), dt))),
    "csr_matvec_transpose": _csr_op(
        lambda a, dt: ms.csr_matvec_transpose(a, _ones((3,), dt))
    ),
    "csc_matmul": _csc_op(lambda a, dt: ms.csc_matmul(a, _ones((4, 2), dt))),
    "csr_matmul_transpose": _csr_op(
        lambda a, dt: native.csr_matmul_transpose(
            a.data, a.indices, a.indptr, _ones((3, 2), dt), CSR_SHAPE
        )
    ),
    # the batched variants
    "csr_batched_matvec": _csr_op(
        lambda a, dt: ms.csr_batched_matvec(a, _ones((2, 4), dt))
    ),
    "csr_batched_matmul": _csr_op(
        lambda a, dt: ms.csr_batched_matmul(a, _ones((2, 4, 2), dt))
    ),
    "csc_batched_matmul": _csc_op(
        lambda a, dt: ms.csc_batched_matmul(a, _ones((2, 4, 2), dt))
    ),
}

CSC_OPS = {
    "csc_matvec_transpose",
    "csc_matmul_transpose",
    "csc_matvec",
    "csc_matmul",
    "csc_batched_matmul",
}
NAMES = sorted(OPS)


def _kept_kwargs(name):
    """The same matrix with only the in-range entries stored."""
    if name in CSC_OPS:
        return {"indptr": CSC_KEPT_INDPTR, "data": KEPT_DATA}
    return {"indptr": CSR_KEPT_INDPTR, "data": KEPT_DATA}


def _values(array):
    mx.eval(array)
    return np.asarray(array.astype(mx.float32)).ravel().tolist()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,indices", OUT_OF_RANGE, ids=lambda v: v)
def test_a_product_ignores_an_out_of_range_index(name, device, label, indices):
    """Equals the product of the entries that are in range."""
    kept = [indices[0], indices[2]]
    with mx.stream(device):
        got = _values(OPS[name](indices, mx.float32))
        want = _values(OPS[name](kept, mx.float32, **_kept_kwargs(name)))
    assert got == want


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("label,indices", OUT_OF_RANGE, ids=lambda v: v)
def test_a_product_agrees_between_cpu_and_gpu(name, label, indices):
    """These disagreed on the same input, and one of them was unstable."""
    with mx.stream(mx.cpu):
        cpu = _values(OPS[name](indices, mx.float32))
    with mx.stream(mx.gpu):
        gpu = _values(OPS[name](indices, mx.float32))
    assert cpu == gpu


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_product_ignores_an_out_of_range_index_for_every_dtype(
    name, device, dtype
):
    """The accumulator differs by value dtype, and so does the loop it lives in."""
    indices = [0, 9999, 3]
    with mx.stream(device):
        got = _values(OPS[name](indices, dtype))
        want = _values(OPS[name]([0, 3], dtype, **_kept_kwargs(name)))
    assert got == want


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_a_product_compares_the_index_at_full_width(name, device):
    """A 64-bit index above INT_MAX must not be cast to int before comparing."""
    indices = [0, 2**32 + 1, 3]
    with mx.stream(device):
        got = _values(OPS[name](indices, mx.float32, index_dtype=mx.int64))
        want = _values(
            OPS[name]([0, 3], mx.float32, index_dtype=mx.int64,
                      **_kept_kwargs(name))
        )
    assert got == want


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_a_product_with_every_index_out_of_range(name, device):
    """Nothing is in range, so the product is the product of an empty matrix."""
    with mx.stream(device):
        got = _values(OPS[name]([-1, 4, 9999], mx.float32))
    assert got == [0.0] * len(got)


# Every dense operand above is all ones, so each stored value contributes once
# per output column and once per batch. The product of those two is how many
# times the value shows up in the total.
COPIES = {
    "csr_matvec": 1,
    "csc_matvec_transpose": 1,
    "csr_matmul": 2,
    "csc_matmul_transpose": 2,
    "csc_matvec": 1,
    "csr_matvec_transpose": 1,
    "csc_matmul": 2,
    "csr_matmul_transpose": 2,
    "csr_batched_matvec": 2,
    "csr_batched_matmul": 4,
    "csc_batched_matmul": 4,
}


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("indices", [[0, 1, 3], [0, 0, 0], [3, 3, 3]])
def test_a_legal_index_is_untouched(name, device, indices):
    """Including the largest legal index, which the guard must still accept."""
    with mx.stream(device):
        got = _values(OPS[name](indices, mx.float32))
    assert sum(got) == pytest.approx(sum(DATA) * COPIES[name], rel=1e-6)


# --- the paths a three-by-four matrix never reaches -------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rhs_cols", [1, 2, 3, 4, 8, 16, 17])
def test_matmul_guards_every_specialized_rhs_width(device, rhs_cols):
    """The matmuls dispatch on the RHS width, so each width is its own loop."""
    with mx.stream(device):
        got = _values(ms.csr_matmul(_csr([0, 9999, 3]), mx.ones((4, rhs_cols))))
        want = _values(
            ms.csr_matmul(
                _csr([0, 3], data=KEPT_DATA, indptr=CSR_KEPT_INDPTR),
                mx.ones((4, rhs_cols)),
            )
        )
        batched_got = _values(
            ms.csr_batched_matmul(_csr([0, 9999, 3]), mx.ones((2, 4, rhs_cols)))
        )
        batched_want = _values(
            ms.csr_batched_matmul(
                _csr([0, 3], data=KEPT_DATA, indptr=CSR_KEPT_INDPTR),
                mx.ones((2, 4, rhs_cols)),
            )
        )
    assert got == want
    assert batched_got == batched_want


def _wide_csr(bad, rows=4096):
    """Wide enough that the CPU implementations take their parallel path."""
    data = np.arange(1, rows + 1, dtype=np.float32)
    indices = np.arange(rows, dtype=np.int32) % 4
    indices[rows // 2] = bad
    return ms.csr_array(
        (
            mx.array(data),
            mx.array(indices),
            mx.array(np.arange(rows + 1, dtype=np.int32)),
        ),
        shape=(rows, 4),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_the_parallel_path_guards_too(device):
    """A one-row matrix runs serially; the range-splitting path is separate."""
    rows = 4096
    bad_matrix = _wide_csr(9999, rows)
    kept = _wide_csr(0, rows)
    kept_data = np.asarray(kept.data)
    kept_data[rows // 2] = 0.0
    kept = ms.csr_array(
        (mx.array(kept_data), kept.indices, kept.indptr), shape=(rows, 4)
    )
    with mx.stream(device):
        got = _values(ms.csr_matvec_transpose(bad_matrix, mx.ones((rows,))))
        want = _values(ms.csr_matvec_transpose(kept, mx.ones((rows,))))
        got_mm = _values(
            native.csr_matmul_transpose(
                bad_matrix.data, bad_matrix.indices, bad_matrix.indptr,
                mx.ones((rows, 3)), (rows, 4),
            )
        )
        want_mm = _values(
            native.csr_matmul_transpose(
                kept.data, kept.indices, kept.indptr, mx.ones((rows, 3)),
                (rows, 4),
            )
        )
    assert got == want
    assert got_mm == want_mm


@pytest.mark.parametrize("device", DEVICES)
def test_a_realistic_product_is_unchanged(device):
    """A guard that changed a legal answer would show up here."""
    import scipy.sparse as sp

    reference = sp.random(40, 25, density=0.2, format="csr", random_state=11)
    reference.sort_indices()
    vector = np.arange(25, dtype=np.float32)
    dense = np.arange(25 * 3, dtype=np.float32).reshape(25, 3)
    with mx.stream(device):
        matrix = ms.csr_array(
            (
                mx.array(reference.data.astype(np.float32)),
                mx.array(reference.indices.astype(np.int32)),
                mx.array(reference.indptr.astype(np.int32)),
            ),
            shape=(40, 25),
        )
        product = ms.csr_matvec(matrix, mx.array(vector))
        matmul = ms.csr_matmul(matrix, mx.array(dense))
        transposed = ms.csr_matvec_transpose(matrix, mx.ones((40,)))
        mx.eval(product, matmul, transposed)
    assert np.allclose(np.asarray(product), reference @ vector, atol=1e-5)
    assert np.allclose(np.asarray(matmul), reference @ dense, atol=1e-4)
    assert np.allclose(
        np.asarray(transposed), reference.T @ np.ones(40, dtype=np.float32),
        atol=1e-5,
    )


# --- the transpose the non-float32 GPU products are lowered through --------


def _parts(array):
    mx.eval(array.data, array.indices, array.indptr)
    return (
        np.asarray(array.data).tolist(),
        np.asarray(array.indices).tolist(),
        np.asarray(array.indptr).tolist(),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,indices", OUT_OF_RANGE, ids=lambda v: v)
def test_transpose_ignores_an_out_of_range_column(device, label, indices):
    """A fourth copy of the same counting sort, reached by csr_matvec_transpose
    and csr_matmul_transpose whenever the value dtype is not float32."""
    kept = [indices[0], indices[2]]
    with mx.stream(device):
        data, out_indices, indptr = _parts(_csr(indices).T)
        want_data, want_indices, want_indptr = _parts(
            _csr(kept, data=KEPT_DATA, indptr=CSR_KEPT_INDPTR).T
        )
    assert indptr == want_indptr
    # The pointer array must not still count the entry that was never placed.
    assert indptr[-1] == 2
    assert data[:2] == want_data[:2]
    assert out_indices[:2] == want_indices[:2]
    assert data[2:] == [0.0]
    assert out_indices[2:] == [0]


@pytest.mark.parametrize("label,indices", OUT_OF_RANGE, ids=lambda v: v)
def test_transpose_agrees_between_cpu_and_gpu(label, indices):
    with mx.stream(mx.cpu):
        cpu = _parts(_csr(indices).T)
    with mx.stream(mx.gpu):
        gpu = _parts(_csr(indices).T)
    assert cpu == gpu


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_transposed_products_are_guarded_for_every_dtype(device, dtype):
    """These lower through the transpose off float32, so a guard in their own
    kernel is not enough on its own."""
    with mx.stream(device):
        got = _values(ms.csr_matvec_transpose(
            _csr([0, 9999, 3], dtype=dtype), _ones((3,), dtype)))
        want = _values(ms.csr_matvec_transpose(
            _csr([0, 3], data=KEPT_DATA, indptr=CSR_KEPT_INDPTR, dtype=dtype),
            _ones((3,), dtype)))
    assert got == want


# --- memory safety, which is the part a returned value cannot show ----------


CHILD_PROGRAM = """
import numpy as np
import mlx.core as mx
import mlx_sparse as ms
import mlx_sparse._native as native

CSR_SHAPE = {csr_shape!r}
CSC_SHAPE = {csc_shape!r}

def _csr(indices, data, indptr):
    return ms.csr_array(
        (mx.array(data, dtype=mx.float32),
         mx.array(indices, dtype=mx.int64),
         mx.array(indptr, dtype=mx.int64)),
        shape=CSR_SHAPE)

def _csc(indices, data, indptr):
    return ms.csc_array(
        (mx.array(data, dtype=mx.float32),
         mx.array(indices, dtype=mx.int64),
         mx.array(indptr, dtype=mx.int64)),
        shape=CSC_SHAPE)

OPS = {{
 "csr_matvec": lambda a: ms.csr_matvec(a, mx.ones((4,))),
 "csc_matvec_transpose": lambda a: ms.csc_matvec_transpose(a, mx.ones((4,))),
 "csr_matmul": lambda a: ms.csr_matmul(a, mx.ones((4, 2))),
 "csc_matmul_transpose": lambda a: native.csc_matmul_transpose(
     a.data, a.indices, a.indptr, mx.ones((4, 2)), CSC_SHAPE),
 "csc_matvec": lambda a: ms.csc_matvec(a, mx.ones((4,))),
 "csr_matvec_transpose": lambda a: ms.csr_matvec_transpose(a, mx.ones((3,))),
 "csc_matmul": lambda a: ms.csc_matmul(a, mx.ones((4, 2))),
 "csr_matmul_transpose": lambda a: native.csr_matmul_transpose(
     a.data, a.indices, a.indptr, mx.ones((3, 2)), CSR_SHAPE),
 "csr_batched_matvec": lambda a: ms.csr_batched_matvec(a, mx.ones((2, 4))),
 "csr_batched_matmul": lambda a: ms.csr_batched_matmul(a, mx.ones((2, 4, 2))),
 "csc_batched_matmul": lambda a: ms.csc_batched_matmul(a, mx.ones((2, 4, 2))),
}}

name = {name!r}
bad = {bad!r}
is_csc = {is_csc!r}
build = _csc if is_csc else _csr
indptr = {indptr!r}
kept_indptr = {kept_indptr!r}

with mx.stream(mx.{device}):
    for _ in range(64):
        out = OPS[name](build([0, bad, 3], [1.0, 2.0, 3.0], indptr))
        mx.eval(out)
    kept = OPS[name](build([0, 3], [1.0, 3.0], kept_indptr))
    mx.eval(kept)
print("survived", np.asarray(out).ravel().tolist() == np.asarray(kept).ravel().tolist())
"""


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_a_far_out_of_range_index_does_not_corrupt_memory(name, device):
    """Its own process: an index this far out leaves the mapped region, and the
    process dies from a signal that no assertion inside it could report."""
    is_csc = name in CSC_OPS
    program = CHILD_PROGRAM.format(
        csr_shape=CSR_SHAPE,
        csc_shape=CSC_SHAPE,
        name=name,
        bad=FAR,
        is_csc=is_csc,
        device=device,
        indptr=CSC_INDPTR if is_csc else CSR_INDPTR,
        kept_indptr=CSC_KEPT_INDPTR if is_csc else CSR_KEPT_INDPTR,
    )
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
    assert "survived True" in result.stdout, result.stdout


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("bad", [9999, FAR])
def test_the_transpose_does_not_corrupt_memory(device, bad):
    """This one faulted at 9999, which none of the products managed."""
    program = f"""
import numpy as np
import mlx.core as mx
import mlx_sparse as ms

with mx.stream(mx.{device}):
    for _ in range(64):
        A = ms.csr_array(
            (mx.array({DATA!r}, dtype=mx.float32),
             mx.array([0, {bad}, 3], dtype=mx.int64),
             mx.array({CSR_INDPTR!r}, dtype=mx.int64)),
            shape={CSR_SHAPE!r})
        T = A.T
        mx.eval(T.data, T.indices, T.indptr)
print("survived", np.asarray(T.indptr).tolist())
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
        f"the transpose on the {device} stream exited with {result.returncode} "
        f"(negative means a fatal signal): {result.stderr[-2000:]}"
    )
    assert "survived [0, 1, 1, 1, 2]" in result.stdout, result.stdout
