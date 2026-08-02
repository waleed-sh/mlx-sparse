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

"""``eigsh`` reports how far off its answer is, at the DEFAULT ``ncv``.

One ``ncv``-bounded Ritz extraction is not a convergence loop, so accuracy is
governed entirely by ``ncv`` and the default is frequently too small. Every
existing ``eigsh`` cell in this suite passes an explicit ``ncv``, which is the
one thing a caller does not do, so nothing here observed what the default
returns. These cells call ``eigsh`` the way its signature invites.
"""

import warnings

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse.linalg.utils.spectral import (
    RITZ_BACKWARD_ERROR_TOLERANCE,
    UnconvergedRitzWarning,
    as_csr,
    float32_csr,
    matrix_inf_norm,
    ritz_backward_error,
)


def _path_laplacian(n):
    """Laplacian of a path graph: tridiagonal, and its spectrum is closed form."""
    dense = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        dense[i, i + 1] = -1.0
        dense[i + 1, i] = -1.0
    for i in range(n):
        dense[i, i] = -dense[i].sum()
    return dense


def _star_laplacian(n):
    """Laplacian of a star graph: eigenvalues 0, 1 with multiplicity n-2, and n.

    The largest is isolated, so a small Krylov basis is enough for it.
    """
    dense = np.zeros((n, n), dtype=np.float32)
    for i in range(1, n):
        dense[0, i] = -1.0
        dense[i, 0] = -1.0
    for i in range(n):
        dense[i, i] = -dense[i].sum()
    return dense


def _csr(mx, dense):
    rows, cols = np.nonzero(dense)
    return ms.coo_array(
        (
            mx.array(dense[rows, cols].astype(np.float32)),
            (mx.array(rows.astype(np.int32)), mx.array(cols.astype(np.int32))),
        ),
        shape=dense.shape,
    ).tocsr(canonical=True)


def test_default_ncv_warns_and_the_warning_reports_the_backward_error(mx):
    dense = _path_laplacian(500)
    A = _csr(mx, dense)
    with pytest.warns(UnconvergedRitzWarning) as record:
        values, vectors = ms.linalg.eigsh(A, k=4, which="LM")
    mx.eval(values, vectors)
    message = str(record[0].message)
    assert "backward error" in message
    assert "ncv" in message


def test_the_reported_error_tracks_how_wrong_the_eigenvalues_actually_are(mx):
    # The warning is only worth anything if the number it reports moves with the
    # real error. Both are measured here against a dense eigendecomposition.
    dense = _path_laplacian(400)
    truth = np.linalg.eigvalsh(dense.astype(np.float64))[-3:]
    A = _csr(mx, dense)
    csr = float32_csr(as_csr(A))

    measured = []
    for ncv in (None, 20, 60, 150):
        kwargs = {} if ncv is None else {"ncv": ncv}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnconvergedRitzWarning)
            values, vectors = ms.linalg.eigsh(A, k=3, which="LM", **kwargs)
        mx.eval(values, vectors)
        backward = float(ritz_backward_error(csr, values, vectors))
        got = np.sort(np.array(values).astype(np.float64))
        value_error = np.abs(got - truth).max() / np.abs(truth).max()
        measured.append((ncv, backward, value_error))

    # Both columns fall together, and the reported error never claims accuracy
    # the eigenvalues do not have.
    backwards = [row[1] for row in measured]
    value_errors = [row[2] for row in measured]
    assert backwards[0] > 10 * backwards[-1], measured
    assert value_errors[0] > 10 * value_errors[-1], measured
    for _, backward, value_error in measured:
        assert value_error <= 20 * backward + 1e-6, measured

    # The default is the bad end of that range, so the cell cannot pass by
    # every row being equally converged.
    assert measured[0][1] > RITZ_BACKWARD_ERROR_TOLERANCE, measured
    assert measured[0][2] > 1e-3, measured


def test_default_ncv_is_wrong_enough_to_matter_on_a_laplacian(mx):
    dense = _path_laplacian(500)
    truth = np.linalg.eigvalsh(dense.astype(np.float64))
    A = _csr(mx, dense)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnconvergedRitzWarning)
        values = ms.linalg.eigsh(A, k=1, which="LM", return_eigenvectors=False)
    mx.eval(values)
    got = float(np.array(values)[0])
    # Not a tolerance dressed up as a check: this asserts the default is BAD,
    # and it is the cell that fails if the default ever becomes good.
    assert abs(got - truth[-1]) / truth[-1] > 1e-3, (got, truth[-1])


def test_a_well_separated_spectrum_converges_at_the_default_and_does_not_warn(mx):
    # The negative control the warning needs: a star graph Laplacian has an
    # isolated largest eigenvalue n, so the default basis is ample and nothing
    # should be reported. A check that fired unconditionally would look just as
    # green without this cell.
    n = 300
    A = _csr(mx, _star_laplacian(n))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnconvergedRitzWarning)
        values, vectors = ms.linalg.eigsh(A, k=2, which="LM")
    mx.eval(values, vectors)
    got = np.sort(np.array(values).astype(np.float64))
    np.testing.assert_allclose(got, [1.0, float(n)], rtol=1e-5)
    csr = float32_csr(as_csr(A))
    assert (
        float(ritz_backward_error(csr, values, vectors)) < RITZ_BACKWARD_ERROR_TOLERANCE
    )


def test_the_small_end_of_a_laplacian_spectrum_is_measured_too(mx):
    # which="SA" is what a caller wanting a Fiedler value asks for, and no cell
    # in this suite exercised it.
    dense = _path_laplacian(300)
    A = _csr(mx, dense)
    with pytest.warns(UnconvergedRitzWarning):
        values, vectors = ms.linalg.eigsh(A, k=3, which="SA")
    mx.eval(values, vectors)
    truth = np.linalg.eigvalsh(dense.astype(np.float64))[:3]
    got = np.sort(np.array(values).astype(np.float64))
    assert np.abs(got - truth).max() > 1e-4, (got, truth)


def test_warning_fires_even_when_only_eigenvalues_are_requested(mx):
    dense = _path_laplacian(400)
    A = _csr(mx, dense)
    with pytest.warns(UnconvergedRitzWarning):
        values = ms.linalg.eigsh(A, k=3, which="LM", return_eigenvectors=False)
    mx.eval(values)
    assert values.shape == (3,)


def test_infinity_norm_matches_a_dense_reference(mx):
    dense = _path_laplacian(50)
    csr = float32_csr(as_csr(_csr(mx, dense)))
    expected = np.abs(dense).sum(axis=1).max()
    assert float(matrix_inf_norm(csr)) == pytest.approx(expected, rel=1e-6)


def test_backward_error_of_an_exact_eigenpair_is_zero(mx):
    # A diagonal matrix has known eigenpairs, so the instrument can be checked
    # against a case where the answer is not in doubt.
    dense = np.diag(np.array([3.0, 1.0, 2.0], dtype=np.float32))
    csr = float32_csr(as_csr(_csr(mx, dense)))
    values = mx.array(np.array([3.0], dtype=np.float32))
    vectors = mx.array(np.array([[1.0], [0.0], [0.0]], dtype=np.float32))
    assert float(ritz_backward_error(csr, values, vectors)) == pytest.approx(0.0)

    wrong = mx.array(np.array([2.0], dtype=np.float32))
    assert float(ritz_backward_error(csr, wrong, vectors)) > 0.1


def test_an_all_zero_matrix_does_not_divide_by_zero(mx):
    zeros = ms.csr_array(
        (
            mx.zeros((0,), dtype=mx.float32),
            mx.zeros((0,), dtype=mx.int32),
            mx.zeros((5,), dtype=mx.int32),
        ),
        shape=(4, 4),
    )
    csr = float32_csr(as_csr(zeros))
    assert float(matrix_inf_norm(csr)) == pytest.approx(1.0)
    values = mx.zeros((1,), dtype=mx.float32)
    vectors = mx.zeros((4, 1), dtype=mx.float32)
    assert np.isfinite(float(ritz_backward_error(csr, values, vectors)))
