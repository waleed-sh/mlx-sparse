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

"""Regression: the default Krylov start vector must not collapse on constant row sums.

A constant vector is an exact eigenvector of any matrix whose rows sum to the same
value, which covers two input families the solvers here are routinely pointed at: graph
Laplacians (``L @ 1 == 0``) and row-stochastic transition matrices (``P @ 1 == 1``).
Starting Lanczos or Arnoldi from a pure ``ones`` vector therefore built a
one-dimensional Krylov space, and every solver returned ``k`` copies of that one
eigenpair, silently and regardless of ``which``.

The tell is the rank of the returned basis, not the residual. Those ``k`` copies are a
genuine eigenpair, so the residual ``||A v - lambda v||`` is 0 and reports success; what
is wrong is that the ``k`` vectors are all the same vector. These tests assert the
returned basis has full rank and orthonormal columns, which is the property the bug
violated and which does not depend on a near-degenerate spectrum being resolved.

Eigenvalue tolerances here are single-shot Lanczos in float32 with no restart, on
graphs whose extreme spectrum is clustered, so they are convergence-grade rather than
tight.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh as sp_eigsh

import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse.linalg.utils.spectral import start_vector

pytestmark = pytest.mark.native


def _csr(A):
    A = A.tocsr().astype(np.float32)
    A.sort_indices()
    return ms.csr_array(
        (
            mx.array(A.data),
            mx.array(A.indices.astype(np.int32)),
            mx.array(A.indptr.astype(np.int32)),
        ),
        shape=A.shape,
        canonical=True,
        validate="full",
    )


def _path_adj(n):
    o = np.ones(n - 1)
    return sp.diags([o, o], [-1, 1], format="csr")


def _cycle_adj(n):
    A = _path_adj(n).tolil()
    A[0, n - 1] = 1.0
    A[n - 1, 0] = 1.0
    return A.tocsr()


def _grid_adj(m):
    # Cartesian product of two paths -> m x m grid-graph adjacency.
    P = _path_adj(m)
    Im = sp.identity(m, format="csr")
    return (sp.kron(P, Im) + sp.kron(Im, P)).tocsr()


def _lap(adj):
    return sp.csgraph.laplacian(adj.tocsr()).tocsr()


def _row_stochastic(adj):
    adj = adj.tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    return (sp.diags(1.0 / degree) @ adj).tocsr()


def _assert_basis_is_independent(vectors, k, context):
    """Assert ``k`` returned Krylov vectors span ``k`` dimensions rather than one."""
    V = np.array(vectors).astype(np.float64)
    # eigsh and eigs return the vectors as columns; svds returns v^T with them as rows.
    if V.shape[0] == k and V.shape[1] != k:
        V = V.T
    assert V.shape[1] == k, f"{context}: expected {k} vectors, got shape {V.shape}"

    rank = np.linalg.matrix_rank(V, tol=1e-6)
    assert rank == k, (
        f"{context}: returned basis has rank {rank}, expected {k}. "
        "A rank below k means the same eigenpair came back k times."
    )
    gram = V.T @ V
    off_diagonal = np.abs(gram - np.diag(np.diag(gram))).max()
    assert off_diagonal < 1e-4, f"{context}: columns not orthogonal, {off_diagonal:.2e}"


@pytest.mark.parametrize("graph", ["path500", "cycle500", "grid30"])
@pytest.mark.parametrize("which", ["LM", "SM"])
def test_eigsh_on_a_laplacian_returns_k_distinct_eigenpairs(graph, which):
    adj = {
        "path500": lambda: _path_adj(500),
        "cycle500": lambda: _cycle_adj(500),
        "grid30": lambda: _grid_adj(30),
    }[graph]()
    L = _lap(adj)

    w, v = linalg.eigsh(_csr(L), k=4, which=which, ncv=200)

    _assert_basis_is_independent(v, 4, f"eigsh({graph}, which={which})")
    assert (
        len(set(np.round(np.array(w), 6))) > 1
    ), f"eigsh({graph}, which={which}) returned one repeated eigenvalue: {np.array(w)}"


@pytest.mark.parametrize("graph", ["path500", "cycle500", "grid30"])
def test_eigsh_laplacian_lm_reaches_the_true_top_of_the_spectrum(graph):
    adj = {
        "path500": lambda: _path_adj(500),
        "cycle500": lambda: _cycle_adj(500),
        "grid30": lambda: _grid_adj(30),
    }[graph]()
    L = _lap(adj)
    ref = np.sort(
        sp_eigsh(L.astype(np.float64), k=4, which="LM", return_eigenvectors=False)
    )

    w, _ = linalg.eigsh(_csr(L), k=4, which="LM", ncv=40)
    got = np.sort(np.array(w))

    relative_error = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    assert (
        relative_error < 3.0e-2
    ), f"{graph}: LM relerr {relative_error:.2e}, got {got}"


def test_eigsh_laplacian_sm_still_recovers_the_constant_null_vector():
    """The ones term has to survive: a Laplacian's smallest eigenvector is constant."""
    L = _lap(_path_adj(500))

    w, v = linalg.eigsh(_csr(L), k=4, which="SM", ncv=200)

    w = np.array(w)
    V = np.array(v).astype(np.float64)
    smallest = V[:, int(np.argmin(w))]
    ones = np.ones(V.shape[0])
    overlap = abs(smallest @ ones) / (np.linalg.norm(smallest) * np.linalg.norm(ones))

    assert float(np.min(w)) == pytest.approx(0.0, abs=1e-4)
    assert overlap > 0.99, f"SM eigenvector is not the constant vector, cos={overlap}"


def test_eigsh_laplacian_lm_tightens_with_ncv():
    """The residual is convergence-limited, not the collapse: bigger ncv, tighter."""
    L = _lap(_path_adj(500))
    ref = np.sort(
        sp_eigsh(L.astype(np.float64), k=4, which="LM", return_eigenvectors=False)
    )

    previous = None
    for ncv in (40, 100, 200):
        w, _ = linalg.eigsh(_csr(L), k=4, which="LM", ncv=ncv)
        relative_error = np.max(np.abs(np.sort(np.array(w)) - ref)) / np.max(
            np.abs(ref)
        )
        if previous is not None:
            assert (
                relative_error <= previous + 1e-9
            ), f"ncv={ncv} did not tighten: {relative_error} vs {previous}"
        previous = relative_error
    assert previous < 1.0e-3, f"path500 LM should reach 1e-3 by ncv=200, got {previous}"


def test_svds_on_a_laplacian_returns_k_distinct_singular_vectors():
    """``svds`` shares the start vector, so it collapsed on the same input."""
    L = _lap(_path_adj(300))

    _, s, vt = linalg.svds(_csr(L), k=4)

    _assert_basis_is_independent(vt, 4, "svds(path300 laplacian)")
    assert float(np.max(np.array(s))) > 1.0, f"svds collapsed to ~zero: {np.array(s)}"


def test_eigs_on_a_row_stochastic_matrix_returns_k_distinct_eigenpairs():
    """``P @ 1 == 1`` makes the constant vector an exact eigenvector of a walk matrix."""
    P = _row_stochastic(_path_adj(200))

    w, v = linalg.eigs(_csr(P), k=4, which="LM", ncv=60)

    _assert_basis_is_independent(v, 4, "eigs(path200 row-stochastic)")
    # The trivial stationary eigenpair is lambda=1; returning it four times was the bug.
    assert (
        np.sum(np.isclose(np.array(w), 1.0, atol=1e-3)) <= 1
    ), f"eigs returned the trivial eigenvalue more than once: {np.array(w)}"


def _spd(n=200):
    d = np.linspace(1.0, 50.0, n)
    off = sp.random(n, n, density=0.01, random_state=1) * 0.05
    return (sp.diags(d) + off + off.T).tocsr()


def test_generic_spd_lm_unchanged():
    A = _spd()
    ref = np.sort(
        sp_eigsh(A.astype(np.float64), k=4, which="LM", return_eigenvectors=False)
    )

    w, _ = linalg.eigsh(_csr(A), k=4, which="LM", ncv=100)

    relative_error = np.max(np.abs(np.sort(np.array(w)) - ref)) / np.max(np.abs(ref))
    assert relative_error < 1.0e-3, f"generic SPD LM regressed: {relative_error:.2e}"


def test_default_start_vector_ignores_the_global_random_stream():
    """The fixed key is what keeps ``v0=None`` reproducible, so pin that, not a value."""
    mx.random.seed(0)
    first = np.array(start_vector(None, n=64))
    mx.random.seed(12345)
    mx.eval(mx.random.normal(shape=(1000,)))  # advance the global stream
    second = np.array(start_vector(None, n=64))

    np.testing.assert_array_equal(first, second)


def test_default_start_vector_does_not_consume_the_global_random_stream():
    mx.random.seed(7)
    expected = np.array(mx.random.normal(shape=(4,)))
    mx.random.seed(7)
    start_vector(None, n=64)
    actual = np.array(mx.random.normal(shape=(4,)))

    np.testing.assert_array_equal(expected, actual)


def test_default_start_vector_keeps_its_constant_component_as_n_grows():
    """A pure random start would dilute the constant overlap to ``~1/sqrt(n)``."""
    for n in (100, 10_000):
        v = np.array(start_vector(None, n=n)).astype(np.float64)
        ones = np.ones(n)
        overlap = abs(v @ ones) / (np.linalg.norm(v) * np.linalg.norm(ones))
        assert overlap > 0.5, f"n={n}: constant overlap decayed to {overlap:.4f}"


def test_eigsh_default_start_is_reproducible_across_calls():
    a = _csr(_spd())

    w1, _ = linalg.eigsh(a, k=4, which="LM", ncv=60)
    w2, _ = linalg.eigsh(a, k=4, which="LM", ncv=60)

    np.testing.assert_array_equal(np.array(w1), np.array(w2))


def test_explicit_v0_still_honored():
    A = _spd()
    ref = np.sort(
        sp_eigsh(A.astype(np.float64), k=4, which="LM", return_eigenvectors=False)
    )

    w, _ = linalg.eigsh(
        _csr(A), k=4, which="LM", ncv=100, v0=mx.ones((A.shape[0],), dtype=mx.float32)
    )

    relative_error = np.max(np.abs(np.sort(np.array(w)) - ref)) / np.max(np.abs(ref))
    assert relative_error < 1.0e-3, f"explicit v0=ones LM wrong: {relative_error:.2e}"


def test_wrong_shape_v0_raises():
    a = _csr(_spd(50))
    with pytest.raises(ValueError):
        linalg.eigsh(a, k=4, which="LM", v0=mx.ones((7,), dtype=mx.float32))
