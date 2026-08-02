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

"""The derivatives of the products, for an entry the forward pass drops.

These four are the data VJPs of csr_matvec, csr_matmul, csc_matmul and
coo_matmul. Each of those forward passes now declines to follow a stored index
outside the declared shape, which fixes what the derivative has to be: if the
output does not depend on an entry's value, the gradient with respect to that
value is zero.

The file states it that way round on purpose. The COO conversion's derivative
already returned zero for a dropped entry, and that was the argument that its
forward was wrong. Here the forward is right and the derivative was wrong, which
is the same invariant read from the other end.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as ms

SHAPE = (4, 4)
DATA = [1.0, 2.0, 3.0]
KEPT_DATA = [DATA[0], 0.0, DATA[2]]
INDPTR = [0, 1, 2, 3, 3]
GOOD = [0, 1, 2]

OUT_OF_RANGE = [
    ("negative_one", -1),
    ("negative_far", -5),
    ("equals_bound", 4),
    ("far_past_end", 9999),
]

ABOVE_INT_MAX = 2**32 + 1
FAR = 2_147_483_000

DEVICES = [pytest.param(mx.cpu, id="cpu"), pytest.param(mx.gpu, id="gpu")]
DTYPES = [
    pytest.param(mx.float32, id="float32"),
    pytest.param(mx.float16, id="float16"),
    pytest.param(mx.bfloat16, id="bfloat16"),
]


def _csr(values, indices, index_dtype):
    return ms.csr_array(
        (values, mx.array(list(indices), dtype=index_dtype),
         mx.array(INDPTR, dtype=index_dtype)),
        shape=SHAPE,
    )


def _csc(values, indices, index_dtype):
    return ms.csc_array(
        (values, mx.array(list(indices), dtype=index_dtype),
         mx.array(INDPTR, dtype=index_dtype)),
        shape=SHAPE,
    )


def _coo(values, indices, index_dtype):
    return ms.coo_array(
        (values, (mx.array(GOOD, dtype=index_dtype),
                  mx.array(list(indices), dtype=index_dtype))),
        shape=SHAPE,
    )


# Each entry is the gradient of sum(op(A)) with respect to A's stored values,
# which is exactly what the data VJP computes.
BUILDERS = {
    "csr_matvec_data_vjp": (
        _csr, lambda a, dt: ms.csr_matvec(a, mx.ones((SHAPE[1],), dtype=dt))),
    "csr_matmul_data_vjp": (
        _csr, lambda a, dt: ms.csr_matmul(a, mx.ones((SHAPE[1], 2), dtype=dt))),
    "csc_matmul_data_vjp": (
        _csc, lambda a, dt: ms.csc_matmul(a, mx.ones((SHAPE[1], 2), dtype=dt))),
    "coo_matmul_data_vjp": (
        _coo, lambda a, dt: ms.coo_matmul(a, mx.ones((SHAPE[1], 2), dtype=dt))),
}

NAMES = sorted(BUILDERS)


def _gradient(name, indices, dtype=mx.float32, index_dtype=mx.int32,
              data=DATA):
    build, apply_op = BUILDERS[name]

    def loss(values):
        return apply_op(build(values, indices, index_dtype), dtype).sum()

    grad = mx.grad(loss)(mx.array(data, dtype=dtype))
    mx.eval(grad)
    return np.asarray(grad.astype(mx.float32)).tolist()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,bad", OUT_OF_RANGE, ids=lambda v: v)
def test_the_gradient_of_a_dropped_entry_is_zero(name, device, label, bad):
    """The output does not depend on it, so the derivative cannot either."""
    with mx.stream(device):
        grad = _gradient(name, [0, bad, 2])
    assert grad[1] == 0.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,bad", OUT_OF_RANGE, ids=lambda v: v)
def test_the_gradient_matches_the_forward_that_drops_the_entry(
    name, device, label, bad
):
    """Equals the gradient for a matrix whose middle value is genuinely zero.

    This is the claim the rest of the file rests on: the derivative has to
    describe the forward pass that actually ran, not the one that would have
    run if the index had been legal.
    """
    with mx.stream(device):
        got = _gradient(name, [0, bad, 2])
        # Same structure, with the dropped entry contributing nothing.
        want = _gradient(name, [0, 0, 2], data=KEPT_DATA)
    assert got[0] == want[0]
    assert got[2] == want[2]
    assert got[1] == 0.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("label,bad", OUT_OF_RANGE, ids=lambda v: v)
def test_the_gradient_agrees_between_cpu_and_gpu(name, label, bad):
    with mx.stream(mx.cpu):
        cpu = _gradient(name, [0, bad, 2])
    with mx.stream(mx.gpu):
        gpu = _gradient(name, [0, bad, 2])
    assert cpu == gpu
    # Agreement alone would also hold if neither backend checked, so pin the
    # value the two of them have to agree ON.
    assert cpu[1] == 0.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_the_gradient_compares_the_index_at_full_width(name, device):
    """2**32 + 1 narrows to 1, a legal index of this shape."""
    with mx.stream(device):
        grad = _gradient(name, [0, ABOVE_INT_MAX, 2], index_dtype=mx.int64)
    assert grad[1] == 0.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_gradient_of_a_dropped_entry_is_zero_for_every_dtype(
    name, device, dtype
):
    with mx.stream(device):
        grad = _gradient(name, [0, 9999, 2], dtype=dtype)
    assert grad[1] == 0.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_every_index_out_of_range_gives_a_zero_gradient(name, device):
    with mx.stream(device):
        grad = _gradient(name, [-1, 4, 9999])
    assert grad == [0.0, 0.0, 0.0]


# --- the negative control --------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("indices", [[0, 1, 2], [0, 0, 0], [3, 3, 3]])
def test_a_legal_index_gives_the_ordinary_gradient(name, device, indices):
    """Including the largest legal index, which the check must still accept."""
    expected = 1.0 if name == "csr_matvec_data_vjp" else 2.0
    with mx.stream(device):
        grad = _gradient(name, indices)
    assert grad == [expected, expected, expected]


@pytest.mark.parametrize("device", DEVICES)
def test_a_realistic_gradient_is_unchanged(device):
    """A check that changed a legal answer would show up here."""
    import scipy.sparse as sp

    reference = sp.random(20, 16, density=0.3, format="csr", random_state=7)
    reference.sort_indices()
    rhs = np.ones((16,), dtype=np.float32)
    indices = mx.array(reference.indices.astype(np.int32))
    indptr = mx.array(reference.indptr.astype(np.int32))

    def loss(values):
        matrix = ms.csr_array((values, indices, indptr), shape=(20, 16))
        return ms.csr_matvec(matrix, mx.array(rhs)).sum()

    with mx.stream(device):
        grad = mx.grad(loss)(mx.array(reference.data.astype(np.float32)))
        mx.eval(grad)
    # d/d data[p] of sum(A @ 1) is 1 for every stored entry.
    assert np.allclose(np.asarray(grad), np.ones(reference.nnz), atol=1e-6)


# --- an operand wide enough for the defect to be visible -------------------

# A four-by-four shape is not enough to see this. The out-of-range read lands
# in a page the allocator has already zeroed, so the gradient comes back as 0.0
# and looks correct. Measured: at width 4 even an index of 100000 reads zeros;
# at width 64 the same index returns -1.2e-20, at width 1024 it returns
# -6.2e+34, and at width 4096 the process dies. The cells above pin the
# contract; this one is the one that fails when the check is absent.
WIDE = 1024
WIDE_BAD = 100_000
WIDE_INDPTR = [0, 1, 2, 3] + [3] * (WIDE - 3)


def _wide_gradient(name, bad, device):
    indptr = mx.array(WIDE_INDPTR, dtype=mx.int64)
    good = mx.array([0, 1, 2], dtype=mx.int64)
    indices = mx.array([0, bad, 2], dtype=mx.int64)

    def loss(values):
        # The dense operand has to be sized for THIS shape, not the narrow one.
        if name == "csr_matvec_data_vjp":
            matrix = ms.csr_array((values, indices, indptr), shape=(WIDE, WIDE))
            return ms.csr_matvec(matrix, mx.ones((WIDE,))).sum()
        if name == "csr_matmul_data_vjp":
            matrix = ms.csr_array((values, indices, indptr), shape=(WIDE, WIDE))
            return ms.csr_matmul(matrix, mx.ones((WIDE, 2))).sum()
        if name == "csc_matmul_data_vjp":
            matrix = ms.csc_array((values, indices, indptr), shape=(WIDE, WIDE))
            return ms.csc_matmul(matrix, mx.ones((WIDE, 2))).sum()
        matrix = ms.coo_array((values, (good, indices)), shape=(WIDE, WIDE))
        return ms.coo_matmul(matrix, mx.ones((WIDE, 2))).sum()

    with mx.stream(device):
        grad = mx.grad(loss)(mx.array(DATA, dtype=mx.float32))
        mx.eval(grad)
    return np.asarray(grad).tolist()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_a_wide_operand_gives_a_zero_gradient_too(name, device):
    """Repeated because a narrow operand hides this: the read has to leave the
    zeroed region before a missing check shows up in the answer."""
    grad = _wide_gradient(name, WIDE_BAD, device)
    assert grad[1] == 0.0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_a_wide_operand_gives_the_same_answer_every_run(name, device):
    """Without the check this returned a different number on each run, which
    is the signature of reading memory nobody wrote."""
    seen = {tuple(_wide_gradient(name, WIDE_BAD, device)) for _ in range(6)}
    assert len(seen) == 1, f"{len(seen)} distinct results from identical runs"


@pytest.mark.parametrize("name", NAMES)
def test_a_wide_operand_agrees_between_cpu_and_gpu(name):
    cpu = _wide_gradient(name, WIDE_BAD, mx.cpu)
    gpu = _wide_gradient(name, WIDE_BAD, mx.gpu)
    assert cpu == gpu
    assert cpu[1] == 0.0


# --- memory safety ---------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_a_far_out_of_range_index_does_not_corrupt_memory(name, device):
    """Its own process. These read outside their operands rather than write
    outside their output, so the damage is a fault or a wrong number rather
    than a corrupted heap -- which is a weaker failure than the forward
    kernels had, and is stated that way rather than overclaimed."""
    program = f"""
import numpy as np
import mlx.core as mx
import mlx_sparse as ms

SHAPE = {SHAPE!r}
INDPTR = {INDPTR!r}
GOOD = {GOOD!r}
bad = {FAR}
name = {name!r}

def csr(values, indices):
    return ms.csr_array((values, mx.array(indices, dtype=mx.int64),
                         mx.array(INDPTR, dtype=mx.int64)), shape=SHAPE)

def csc(values, indices):
    return ms.csc_array((values, mx.array(indices, dtype=mx.int64),
                         mx.array(INDPTR, dtype=mx.int64)), shape=SHAPE)

def coo(values, indices):
    return ms.coo_array((values, (mx.array(GOOD, dtype=mx.int64),
                                  mx.array(indices, dtype=mx.int64))),
                        shape=SHAPE)

BUILDERS = {{
 "csr_matvec_data_vjp": (csr, lambda a: ms.csr_matvec(a, mx.ones((SHAPE[1],)))),
 "csr_matmul_data_vjp": (csr, lambda a: ms.csr_matmul(a, mx.ones((SHAPE[1], 2)))),
 "csc_matmul_data_vjp": (csc, lambda a: ms.csc_matmul(a, mx.ones((SHAPE[1], 2)))),
 "coo_matmul_data_vjp": (coo, lambda a: ms.coo_matmul(a, mx.ones((SHAPE[1], 2)))),
}}
build, apply_op = BUILDERS[name]

def loss(values):
    return apply_op(build(values, [0, bad, 2])).sum()

with mx.stream(mx.{device}):
    for _ in range(64):
        g = mx.grad(loss)(mx.array({DATA!r}, dtype=mx.float32))
        mx.eval(g)
print("survived", np.asarray(g).tolist())
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
    assert "0.0" in result.stdout.split("survived")[1]
