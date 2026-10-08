"""``csr_matmul`` picks its kernel by right-hand width as well as by density.

The cooperative kernel hands a whole 128-thread threadgroup to one output.
That pays when there is nothing else to fill the machine with, and a second
right-hand column is already something else: the scalar kernel gets
``n_rows * rhs_cols`` independent threads and reuses each row's indices across
the columns, while the cooperative one re-reads them per column and adds a tree
reduction on top of that. The selection rule looked only at mean degree.

Measured on coPapersDBLP (540,486 rows, 30.5M entries, mean degree 56.4 -- the
one real graph in the corpus above the density threshold), median of 11, GPU,
each timing load-clean, cooperative against scalar:

    columns   cooperative    scalar
          1       3.438 ms   2.646 ms
          2       4.358 ms   2.377 ms
          4       8.378 ms   1.570 ms   5.3x
          8      16.306 ms   2.001 ms   8.2x
         32      64.765 ms   5.205 ms  12.4x
         62     126.264 ms  10.762 ms  11.7x

A synthetic sweep over mean degree 4 to 256 at two sizes agrees at every width
above one: the scalar kernel wins by 1.8x to 33x, in every cell, in both runs.

⚠️ The one-column case is deliberately left alone. Its cells are 0.5-3 ms and
they did not reproduce: the same synthetic cell came out at 0.55x in one run
and 1.61x in the next, and the real-graph one-column cell has a median/minimum
of 1.95, which is noise rather than a measurement. Two runs disagreeing by 3x
is a story about the machine, so the rule changes only where the evidence
holds, and one column keeps the behaviour it had.

HOW THIS IS TESTED WITHOUT A STOPWATCH
--------------------------------------
A timing assertion in a unit suite is a flake waiting to happen, and the first
thing anyone does with a flaky performance cell is delete it. The two kernels
are instead told apart by their arithmetic: the scalar one accumulates a row
serially, the cooperative one reduces across 128 lanes in a tree. In float32
those orders round differently, so a row built to expose the difference names
the kernel that ran, exactly and on any machine.
"""

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as mxs

# Every row holds this many entries, which is over the density threshold, so
# density alone would select the cooperative kernel.
DENSE_ROW = 256

# One huge value and many ones. Added serially, every 1.0 falls off the end of
# the float32 significand and the sum stays at the huge value. Added as a tree,
# the ones meet each other first and their total survives the final addition.
HUGE = np.float32(1e8)
SERIAL_ORDER = np.float32(1e8)


@pytest.fixture
def matrix():
    """Square, every row holding DENSE_ROW entries at columns 0..DENSE_ROW-1."""
    n = 512
    indptr = np.arange(n + 1, dtype=np.int32) * DENSE_ROW
    indices = np.tile(np.arange(DENSE_ROW, dtype=np.int32), n)
    data = np.ones(n * DENSE_ROW, dtype=np.float32)
    csr = mxs.CSRArray(mx.array(data), mx.array(indices), mx.array(indptr),
                       (n, n))
    mx.eval(csr.data, csr.indices, csr.indptr)
    return csr, n


def _rhs(n, cols):
    host = np.ones((n, cols), dtype=np.float32)
    host[0, :] = HUGE
    out = mx.array(host)
    mx.eval(out)
    return out


def _first_output(csr, rhs):
    return np.asarray(mxs.csr_matmul(csr, rhs))[0, 0]


class TestTheKernelIsChosenByWidth:
    @pytest.mark.parametrize("cols", [2, 3, 4, 8, 64])
    def test_more_than_one_column_uses_the_scalar_kernel(self, matrix, cols):
        csr, n = matrix
        got = _first_output(csr, _rhs(n, cols))
        assert got == SERIAL_ORDER, (
            f"{cols} columns summed in tree order ({got}), so the cooperative "
            "kernel ran"
        )

    def test_one_column_is_unchanged(self, matrix):
        """Left as it was, because its measurements did not reproduce.

        This cell exists to make that a decision rather than an oversight: it
        fails if someone extends the width rule to one column without new
        evidence, which is exactly the change the docstring declines to make.
        """
        csr, n = matrix
        got = _first_output(csr, _rhs(n, 1))
        assert got != SERIAL_ORDER, (
            "one column no longer takes the cooperative kernel; that may well "
            "be right, but it needs measurements that reproduce"
        )


class TestTheFingerprintIsReal:
    """Teeth for the method. If both orders agreed, every cell above would pass
    on any build, including one that ignores width entirely."""

    def test_the_two_summation_orders_disagree(self, matrix):
        csr, n = matrix
        one = _first_output(csr, _rhs(n, 1))
        many = _first_output(csr, _rhs(n, 8))
        assert one != many, (
            "the fixture no longer separates the kernels; every cell in this "
            "module is vacuous until it does"
        )

    def test_both_answers_are_correct(self, matrix):
        """Neither kernel is wrong -- they round differently, and both are
        within float32's tolerance of the exact sum. The rule being changed is
        about speed, so this pins that nothing about the answer moved."""
        csr, n = matrix
        exact = float(HUGE) + (DENSE_ROW - 1)
        for cols in (1, 8):
            got = float(_first_output(csr, _rhs(n, cols)))
            assert abs(got - exact) <= 256.0

    def test_a_sparse_row_still_takes_the_scalar_kernel(self):
        """Below the density threshold nothing changes, at any width."""
        n, per = 512, 8
        indptr = np.arange(n + 1, dtype=np.int32) * per
        indices = np.tile(np.arange(per, dtype=np.int32), n)
        csr = mxs.CSRArray(mx.ones((n * per,), dtype=mx.float32),
                           mx.array(indices), mx.array(indptr), (n, n))
        mx.eval(csr.data)
        for cols in (1, 8):
            host = np.ones((n, cols), dtype=np.float32)
            host[0, :] = HUGE
            rhs = mx.array(host)
            mx.eval(rhs)
            assert np.asarray(mxs.csr_matmul(csr, rhs))[0, 0] == SERIAL_ORDER
