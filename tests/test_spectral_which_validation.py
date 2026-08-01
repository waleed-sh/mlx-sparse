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

"""An unrecognized ``which`` must be rejected rather than answered.

Ritz selection sorts ascending for the small-end selectors and descending for
everything else, so before this was validated any unrecognized selector fell through
to the descending branch and returned the largest-algebraic pairs. A typo, or a
selector that is valid in SciPy but not implemented here, produced a plausible wrong
answer instead of an error.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import scipy.sparse as sp

import mlx_sparse as ms
from mlx_sparse import linalg

pytestmark = pytest.mark.native


def _indefinite_symmetric():
    """Symmetric with a dominant NEGATIVE eigenvalue, so LM and LA differ."""
    eigenvalues = np.array([-50.0, -3.0, 1.0, 2.0, 4.0, 30.0], dtype=np.float32)
    basis, _ = np.linalg.qr(np.random.default_rng(0).random((6, 6)))
    return (basis @ np.diag(eigenvalues) @ basis.T).astype(np.float32)


def _csr(dense):
    a = sp.csr_array(dense).tocsr()
    return ms.csr_array(
        (
            mx.array(a.data.astype(np.float32)),
            mx.array(a.indices.astype(np.int32)),
            mx.array(a.indptr.astype(np.int32)),
        ),
        shape=a.shape,
        sorted_indices=True,
        canonical=True,
    )


@pytest.mark.parametrize("which", ["XX", "BE", "typo", "", "L", "LMM"])
def test_eigsh_rejects_an_unknown_selector(which):
    a = _csr(_indefinite_symmetric())
    with pytest.raises(ValueError, match="eigsh which must be one of"):
        linalg.eigsh(a, k=3, which=which, ncv=6)


@pytest.mark.parametrize("which", ["XX", "BE", "LA", "SA"])
def test_eigs_rejects_a_selector_it_does_not_implement(which):
    """``LA`` and ``SA`` are eigsh's vocabulary; eigs documents LM, SM, LR, SR."""
    a = _csr(_indefinite_symmetric())
    with pytest.raises(ValueError, match="eigs which must be one of"):
        linalg.eigs(a, k=3, which=which, ncv=6)


@pytest.mark.parametrize("which", ["XX", "LA", "LR", "BE"])
def test_svds_rejects_a_selector_it_does_not_implement(which):
    a = _csr(_indefinite_symmetric())
    with pytest.raises(ValueError, match="svds which must be one of"):
        linalg.svds(a, k=2, which=which)


def test_which_rejection_names_the_accepted_selectors():
    a = _csr(_indefinite_symmetric())
    with pytest.raises(ValueError) as excinfo:
        linalg.eigsh(a, k=3, which="BE", ncv=6)
    message = str(excinfo.value)
    for selector in ("LM", "SM", "LA", "SA"):
        assert selector in message
    assert "'BE'" in message


def test_non_string_which_is_rejected():
    a = _csr(_indefinite_symmetric())
    with pytest.raises(ValueError, match="which must be a string"):
        linalg.eigsh(a, k=3, which=0, ncv=6)


@pytest.mark.parametrize("which", ["lm", "sm", "la", "sa"])
def test_eigsh_still_accepts_lowercase_selectors(which):
    a = _csr(_indefinite_symmetric())
    values = linalg.eigsh(a, k=3, which=which, ncv=6, return_eigenvectors=False)
    assert np.array(values).shape == (3,)


@pytest.mark.parametrize(
    "which,expected",
    [
        ("LM", [-50.0, 4.0, 30.0]),
        ("LA", [2.0, 4.0, 30.0]),
        ("SM", [-3.0, 1.0, 2.0]),
        ("SA", [-50.0, -3.0, 1.0]),
    ],
)
def test_each_accepted_selector_picks_its_own_end_of_the_spectrum(which, expected):
    """Pins that the four selectors are genuinely distinct on an indefinite matrix.

    Without this, rejecting unknown selectors could coexist with the accepted ones all
    resolving to the same branch, which is the bug one level up.
    """
    a = _csr(_indefinite_symmetric())

    values = np.sort(
        np.array(linalg.eigsh(a, k=3, which=which, ncv=6, return_eigenvectors=False))
    )

    np.testing.assert_allclose(values, expected, rtol=1e-4, atol=1e-3)
