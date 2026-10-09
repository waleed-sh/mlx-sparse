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

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse import _native, linalg
from mlx_sparse.linalg import preconditioners
from mlx_sparse.linalg.utils.spectral import start_vector


def _path_laplacian(n):
    a = np.diag(np.r_[1.0, np.full(n - 2, 2.0), 1.0])
    a += np.diag(np.full(n - 1, -1.0), 1)
    a += np.diag(np.full(n - 1, -1.0), -1)
    return a.astype(np.float32)


def _csr(mx, scipy_sparse, dense, index_dtype="int32"):
    a = scipy_sparse.csr_array(dense)
    return ms.csr_array(
        (
            mx.array(a.data),
            mx.array(a.indices, dtype=getattr(mx, index_dtype)),
            mx.array(a.indptr, dtype=getattr(mx, index_dtype)),
        ),
        shape=a.shape,
        canonical=True,
        sorted_indices=True,
    )


@pytest.mark.parametrize("routine", ["eigsh", "eigs", "svds"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_default_start_explores_constant_row_sum_matrix(
    mx, scipy_sparse, to_numpy, routine, index_dtype
):
    dense = _path_laplacian(12)
    a = _csr(mx, scipy_sparse, dense, index_dtype)
    if routine == "svds":
        _, values, vh = linalg.svds(a, k=3, ncv=12)
        vectors = to_numpy(vh).T
    else:
        values, vectors = getattr(linalg, routine)(a, k=3, ncv=12)
        vectors = to_numpy(vectors)
    expected = np.linalg.eigvalsh(dense.astype(np.float64))[-3:]
    np.testing.assert_allclose(np.sort(to_numpy(values)), expected, atol=3e-5)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=3e-5)
    if routine != "eigs":
        np.testing.assert_allclose(
            dense @ vectors, vectors * to_numpy(values), atol=3e-5
        )


def test_default_start_is_repeatable_and_preserves_global_random_stream(mx, to_numpy):
    mx.random.seed(731)
    expected = to_numpy(mx.random.normal((16,)))
    mx.random.seed(731)
    first = start_vector(None, n=32)
    second = start_vector(None, n=32)
    actual = to_numpy(mx.random.normal((16,)))
    np.testing.assert_array_equal(to_numpy(first), to_numpy(second))
    np.testing.assert_array_equal(actual, expected)
    assert np.ptp(to_numpy(first)) > 0.5
    mx.random.seed(917)
    np.testing.assert_array_equal(to_numpy(start_vector(None, n=32)), to_numpy(first))


@pytest.mark.parametrize("routine", ["eigsh", "eigs", "svds"])
def test_default_extraction_is_repeatable(mx, scipy_sparse, to_numpy, routine):
    a = _csr(mx, scipy_sparse, _path_laplacian(12))
    kwargs = dict(k=3, ncv=12)
    if routine == "svds":
        kwargs["return_singular_vectors"] = False
    else:
        kwargs["return_eigenvectors"] = False
    first = to_numpy(getattr(linalg, routine)(a, **kwargs))
    second = to_numpy(getattr(linalg, routine)(a, **kwargs))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("routine", ["eigsh", "eigs", "svds"])
@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_explicit_start_direction_is_independent_of_scale(
    mx, scipy_sparse, to_numpy, routine, scale
):
    dense = np.diag(np.arange(1, 7, dtype=np.float32))
    a = _csr(mx, scipy_sparse, dense)
    v0 = np.arange(1, 7, dtype=np.float32)
    kwargs = dict(k=2, ncv=6, v0=mx.array(v0 * scale))
    if routine == "svds":
        values = linalg.svds(a, return_singular_vectors=False, **kwargs)
    else:
        values = getattr(linalg, routine)(a, return_eigenvectors=False, **kwargs)
    np.testing.assert_allclose(np.sort(to_numpy(values)), [5.0, 6.0], atol=3e-5)


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_lanczos_preserves_scaled_user_start(mx, scipy_sparse, to_numpy, scale):
    a = _csr(mx, scipy_sparse, np.diag(np.arange(1, 7, dtype=np.float32)))
    v0 = np.arange(1, 7, dtype=np.float32)
    _, _, basis = linalg.lanczos(a, k=1, v0=mx.array(v0 * scale))
    np.testing.assert_allclose(
        to_numpy(basis)[:, 0], v0 / np.linalg.norm(v0), atol=1e-6
    )


@pytest.mark.parametrize("routine", ["lanczos", "eigsh", "eigs", "svds"])
def test_strided_mixed_sign_start_is_preserved(mx, scipy_sparse, to_numpy, routine):
    a = _csr(mx, scipy_sparse, np.diag(np.arange(1, 7, dtype=np.float32)))
    expected = np.array([1, -2, 3, -4, 5, -6], dtype=np.float32)
    storage = np.empty(12, dtype=np.float32)
    storage[::2], storage[1::2] = expected, 0
    v0 = mx.array(storage)[::2]
    if routine == "lanczos":
        _, _, basis = linalg.lanczos(a, k=1, v0=v0)
        np.testing.assert_allclose(
            to_numpy(basis)[:, 0], expected / np.linalg.norm(expected), atol=1e-6
        )
    elif routine == "svds":
        values = linalg.svds(a, k=2, ncv=6, v0=v0, return_singular_vectors=False)
        np.testing.assert_allclose(np.sort(to_numpy(values)), [5, 6], atol=3e-5)
    else:
        values = getattr(linalg, routine)(
            a, k=2, ncv=6, v0=v0, return_eigenvectors=False
        )
        np.testing.assert_allclose(np.sort(to_numpy(values)), [5, 6], atol=3e-5)


@pytest.mark.parametrize("routine", ["eigsh", "eigs", "svds"])
@pytest.mark.parametrize("matrix", ["identity", "coordinate_start", "constant_start"])
def test_invariant_start_continues_with_independent_directions(
    mx, scipy_sparse, to_numpy, routine, matrix
):
    if matrix == "identity":
        dense, v0 = np.eye(8, dtype=np.float32), None
    elif matrix == "coordinate_start":
        dense = np.diag(np.arange(1, 9, dtype=np.float32))
        v0 = mx.array([1.0] + [0.0] * 7)
    else:
        dense, v0 = _path_laplacian(8), mx.ones((8,))
    a = _csr(mx, scipy_sparse, dense)
    if routine == "svds":
        u, values, vh = linalg.svds(a, k=3, ncv=8, v0=v0)
        vectors = to_numpy(vh).T
        np.testing.assert_allclose(to_numpy(u).T @ to_numpy(u), np.eye(3), atol=3e-5)
    else:
        values, vectors = getattr(linalg, routine)(a, k=3, ncv=8, v0=v0)
        vectors = to_numpy(vectors)
    expected = np.linalg.eigvalsh(dense.astype(np.float64))[-3:]
    np.testing.assert_allclose(np.sort(to_numpy(values)), expected, atol=3e-5)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=3e-5)
    if routine != "eigs":
        np.testing.assert_allclose(
            dense @ vectors, vectors * to_numpy(values), atol=3e-5
        )


@pytest.mark.parametrize("routine", ["eigsh", "eigs", "svds"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_invariant_blocks_preserve_spectral_multiplicity(
    mx, scipy_sparse, to_numpy, routine, index_dtype
):
    dense = np.diag(np.array([1, 1, 2, 2, 3, 3], dtype=np.float32))
    a = _csr(mx, scipy_sparse, dense, index_dtype)
    if routine == "svds":
        u, values, vh = linalg.svds(a, k=3, ncv=6)
        vectors = to_numpy(vh).T
        np.testing.assert_allclose(to_numpy(u).T @ to_numpy(u), np.eye(3), atol=3e-5)
    else:
        values, vectors = getattr(linalg, routine)(a, k=3, ncv=6)
        vectors = to_numpy(vectors)
    np.testing.assert_allclose(np.sort(to_numpy(values)), [2, 3, 3], atol=3e-5)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=3e-5)
    if routine != "eigs":
        np.testing.assert_allclose(
            dense @ vectors, vectors * to_numpy(values), atol=3e-5
        )


@pytest.mark.parametrize("routine", ["eigsh", "eigs", "svds"])
def test_nearly_invariant_start_is_not_normalized_into_roundoff(
    mx, scipy_sparse, to_numpy, routine
):
    dense = np.diag(np.arange(1, 9, dtype=np.float32))
    a = _csr(mx, scipy_sparse, dense)
    v0 = mx.array(np.r_[1.0, np.full(7, 1e-10)].astype(np.float32))
    kwargs = dict(k=3, ncv=8, v0=v0)
    if routine == "svds":
        values = linalg.svds(a, return_singular_vectors=False, **kwargs)
    else:
        values = getattr(linalg, routine)(a, return_eigenvectors=False, **kwargs)
    np.testing.assert_allclose(np.sort(to_numpy(values)), [6, 7, 8], atol=5e-5)


@pytest.mark.parametrize(
    "routine", ["csr_lanczos", "csr_arnoldi", "csr_normal_lanczos"]
)
def test_low_level_factorizations_keep_early_breakdown(
    mx, scipy_sparse, to_numpy, routine
):
    a = _csr(mx, scipy_sparse, np.eye(6, dtype=np.float32))
    args = [a.data, a.indices, a.indptr, mx.ones((6,)), 6, 6, 4]
    if routine == "csr_lanczos":
        args.append(True)
    outputs = getattr(_native.extension(), routine)(*args)
    mx.eval(*outputs)
    assert int(outputs[-1].item()) == 1
    basis = to_numpy(outputs[-2])
    np.testing.assert_allclose(basis[:, 0], np.full(6, 1 / np.sqrt(6)), atol=1e-6)
    np.testing.assert_array_equal(basis[:, 1:], 0)


@pytest.mark.parametrize("routine", ["csr_eigsh", "csr_eigs", "csr_svds"])
def test_native_extraction_rejects_empty_basis(mx, scipy_sparse, routine):
    a = _csr(mx, scipy_sparse, np.eye(6, dtype=np.float32))
    with pytest.raises(RuntimeError, match="Krylov basis dimension 0"):
        getattr(_native.extension(), routine)(
            a.data, a.indices, a.indptr, mx.zeros((6,)), 6, 6, 3, 6, "LM"
        )


def test_gmres_keeps_happy_breakdown_with_invariant_residual(
    mx, scipy_sparse, to_numpy
):
    a = _csr(mx, scipy_sparse, 2 * np.eye(6, dtype=np.float32))
    b = mx.ones((6,))
    x, status = linalg.gmres(a, b, restart=4, maxiter=8)
    assert status == 0
    np.testing.assert_allclose(to_numpy(x), np.full(6, 0.5), atol=1e-6)


@pytest.mark.parametrize("explicit", [False, True])
def test_rectangular_svds_continues_normal_operator_invariant_start(
    mx, scipy_sparse, to_numpy, explicit
):
    dense = np.vstack([_path_laplacian(8) + np.eye(8), 2 * np.eye(8)]).astype(
        np.float32
    )
    a = _csr(mx, scipy_sparse, dense, "int64")
    u, values, vh = linalg.svds(a, k=3, ncv=8, v0=mx.ones((8,)) if explicit else None)
    u, values, vh = to_numpy(u), to_numpy(values), to_numpy(vh)
    np.testing.assert_allclose(
        np.sort(values), np.linalg.svd(dense, compute_uv=False)[:3][::-1], atol=3e-5
    )
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=3e-5)
    np.testing.assert_allclose(vh @ vh.T, np.eye(3), atol=3e-5)
    np.testing.assert_allclose(dense @ vh.T, u * values, atol=3e-5)
    np.testing.assert_allclose(dense.T @ u, vh.T * values, atol=3e-5)


@pytest.mark.parametrize("explicit", [False, True])
def test_stochastic_matrix_retains_repeated_nontrivial_eigenvalues(
    mx, scipy_sparse, to_numpy, explicit
):
    dense = (0.5 * np.eye(8) + np.full((8, 8), 0.5 / 8)).astype(np.float32)
    a = _csr(mx, scipy_sparse, dense)
    values = linalg.eigs(
        a, k=3, ncv=8, v0=mx.ones((8,)) if explicit else None, return_eigenvectors=False
    )
    np.testing.assert_allclose(np.sort(to_numpy(values)), [0.5, 0.5, 1.0], atol=3e-5)


def test_smallest_laplacian_pairs_include_constant_null_vector(
    mx, scipy_sparse, to_numpy
):
    dense = _path_laplacian(12)
    values, vectors = linalg.eigsh(
        _csr(mx, scipy_sparse, dense), k=3, ncv=12, which="SM"
    )
    values, vectors = to_numpy(values), to_numpy(vectors)
    np.testing.assert_allclose(
        np.sort(values), np.linalg.eigvalsh(dense.astype(np.float64))[:3], atol=3e-5
    )
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=3e-5)
    np.testing.assert_allclose(dense @ vectors, vectors * values, atol=3e-5)
    assert abs(
        np.dot(vectors[:, np.argmin(np.abs(values))], np.ones(12) / np.sqrt(12))
    ) == pytest.approx(1.0, abs=3e-5)


def test_larger_basis_improves_smallest_ritz_value(mx, scipy_sparse, to_numpy):
    dense = _path_laplacian(12) + 0.25 * np.eye(12, dtype=np.float32)
    a = _csr(mx, scipy_sparse, dense)
    errors = []
    for ncv in [4, 8, 12]:
        value = linalg.eigsh(a, k=1, which="SM", ncv=ncv, return_eigenvectors=False)
        errors.append(abs(float(to_numpy(value)[0]) - 0.25))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 3e-5


@pytest.mark.parametrize("routine", ["eigsh", "eigs", "svds"])
@pytest.mark.parametrize("scale", [1e-15, 1e15])
def test_spectral_breakdown_threshold_is_relative_to_operator_scale(
    mx, scipy_sparse, to_numpy, routine, scale
):
    dense = np.diag(np.arange(1, 7, dtype=np.float32)) * scale
    a = _csr(mx, scipy_sparse, dense)
    if routine == "svds":
        u, values, vh = linalg.svds(a, k=2, ncv=6)
        np.testing.assert_allclose(to_numpy(u).T @ to_numpy(u), np.eye(2), atol=3e-5)
        np.testing.assert_allclose(
            (dense / scale) @ to_numpy(vh).T,
            to_numpy(u) * (to_numpy(values) / scale),
            atol=3e-5,
        )
    else:
        values = getattr(linalg, routine)(a, k=2, ncv=6, return_eigenvectors=False)
    np.testing.assert_allclose(np.sort(to_numpy(values)) / scale, [5, 6], atol=3e-5)


@pytest.mark.parametrize("routine", ["lanczos", "eigsh", "eigs", "svds"])
@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf])
def test_invalid_user_starts_are_rejected(mx, scipy_sparse, routine, bad):
    a = _csr(mx, scipy_sparse, np.eye(4, dtype=np.float32))
    with pytest.raises(ValueError):
        getattr(linalg, routine)(a, k=1, v0=mx.full((4,), bad))


def test_chebyshev_estimate_explores_constant_row_sum_matrix(mx, scipy_sparse):
    dense = _path_laplacian(12) + np.eye(12, dtype=np.float32)
    a = _csr(mx, scipy_sparse, dense)
    m = preconditioners.chebyshev(a, estimate=True)
    assert m.spectral_info["estimate_steps"] == 12
    assert m.spectral_info["ritz_max"] == pytest.approx(
        np.linalg.eigvalsh(dense.astype(np.float64))[-1], abs=3e-5
    )
