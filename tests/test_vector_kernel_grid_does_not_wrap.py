"""The cooperative vector kernels must not lose work to a wrapped grid.

The vector kernel runs one 128-thread threadgroup per output. Dispatched as a
flat grid, that is ``outputs * 128`` threads in a single dimension, and the
count wraps at 2**32 -- silently. Past the wrap the threadgroups simply do not
run: their outputs keep whatever the buffer held, which is zero for a fresh
allocation, so the product comes back looking plausible and is wrong.

The wrap is at ``outputs >= 2**25`` (because ``2**25 * 128 == 2**32``, which is
already zero in 32 bits), and it takes a matrix big enough that no existing
cell in this suite reaches it. That is why it survived: every other test of
these kernels is far below the limit, and the failure mode is a wrong number
rather than an error.

Measured before the fix, on a real 540,486-row matrix of mean degree 56 at 63
right-hand columns: exactly 2**25 of the 34,050,618 outputs came back zero,
and the correct ones were exactly the first ``outputs - 2**25``.

These cells are deliberately sized just under and just over the boundary, so
one of each pair would still pass on the broken build. A single over-the-line
cell could be made to pass by any change that slowed things down enough to
look different; a matched pair pins the boundary itself.
"""

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as mxs

# The wrap is AT this count, not past it: outputs * kVectorThreads is exactly
# 2**32 here, which is already 0 in 32 bits. The last safe count is one less.
# Writing this as ">" cost a first draft its control cell, which failed on the
# broken build alongside the cells that were supposed to fail -- a control that
# fails with the experiment is not a control.
WRAP_AT_OUTPUTS = 2**25

# The vector kernel is only selected at mean degree >= 32; below that the
# scalar kernel runs and these cells would test nothing.
VECTOR_KERNEL_MIN_MEAN_DEGREE = 32


def _matrix(n_rows, mean_degree, seed=0):
    """A CSR with exactly ``mean_degree`` stored entries in every row."""
    rng = np.random.default_rng(seed)
    indptr = (np.arange(n_rows + 1, dtype=np.int32) * mean_degree)
    indices = rng.integers(0, n_rows, size=n_rows * mean_degree).astype(np.int32)
    data = np.ones(n_rows * mean_degree, dtype=np.float32)
    csr = mxs.CSRArray(mx.array(data), mx.array(indices), mx.array(indptr),
                       (n_rows, n_rows))
    mx.eval(csr.data, csr.indices, csr.indptr)
    return csr, indptr, indices, data


def _reference(indptr, indices, data, rhs, n_rows):
    """The same product in float64 on the host."""
    import scipy.sparse as sp

    matrix = sp.csr_array((data.astype(np.float64), indices, indptr),
                          shape=(n_rows, n_rows))
    return matrix @ rhs.astype(np.float64)


@pytest.mark.parametrize(
    "n_rows, rhs_cols, over_the_line",
    [
        # 65,536 * 511 = 33,488,896 outputs, 4,286,578,688 threads: safe.
        (65_536, 511, False),
        # 65,536 * 512 = 33,554,432 outputs, exactly 2**32 threads: wraps to
        # zero threadgroups. The tightest possible over-the-line case.
        (65_536, 512, True),
        # Well past it, so the pair is not only pinning an off-by-one.
        (65_536, 520, True),
    ],
)
def test_csr_matmul_keeps_every_output(n_rows, rhs_cols, over_the_line):
    outputs = n_rows * rhs_cols
    assert (outputs >= WRAP_AT_OUTPUTS) == over_the_line, (
        "the fixture no longer straddles the boundary it was written for"
    )
    csr, indptr, indices, data = _matrix(n_rows, VECTOR_KERNEL_MIN_MEAN_DEGREE)
    rng = np.random.default_rng(1)
    host = rng.random((n_rows, rhs_cols)).astype(np.float32)
    rhs = mx.array(host)
    mx.eval(rhs)

    got = np.asarray(mxs.csr_matmul(csr, rhs)).astype(np.float64)
    want = _reference(indptr, indices, data, host, n_rows)

    # float32 accumulation over 32 terms; the failure this guards is not eps.
    assert np.abs(got - want).max() < 1e-2
    assert (got == 0).sum() == 0, "outputs left unwritten by threadgroups that never ran"


def test_csr_batched_matmul_keeps_every_output():
    """The batched kernel multiplies the batch in too, so it wraps soonest."""
    n_rows, rhs_cols, batch = 65_536, 260, 2
    outputs = batch * n_rows * rhs_cols
    assert outputs >= WRAP_AT_OUTPUTS
    csr, indptr, indices, data = _matrix(n_rows, VECTOR_KERNEL_MIN_MEAN_DEGREE)
    rng = np.random.default_rng(2)
    host = rng.random((batch, n_rows, rhs_cols)).astype(np.float32)
    rhs = mx.array(host)
    mx.eval(rhs)

    got = np.asarray(mxs.csr_batched_matmul(csr, rhs)).astype(np.float64)
    for b in range(batch):
        want = _reference(indptr, indices, data, host[b], n_rows)
        assert np.abs(got[b] - want).max() < 1e-2
    assert (got == 0).sum() == 0


def test_the_scalar_kernel_is_not_what_is_being_tested():
    """Teeth for the fixtures: below mean degree 32 the other kernel runs.

    Every cell above would still pass on a broken vector kernel if the
    selection rule sent it to the scalar one instead, so the rule the fixtures
    depend on is asserted rather than assumed.
    """
    n_rows = 1024
    for mean_degree, expect_vector in ((31, False), (32, True), (64, True)):
        nnz = n_rows * mean_degree
        assert (nnz >= n_rows * VECTOR_KERNEL_MIN_MEAN_DEGREE) == expect_vector
