"""Residual and subspace checks for native Arnoldi and singular triplets."""

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse import linalg


def _matrix(mx, scipy_sparse, dense, index_dtype="int32", format="csr"):
    sparse = scipy_sparse.csr_array(dense.astype(np.float32))
    result = ms.csr_array(
        (
            mx.array(sparse.data),
            mx.array(sparse.indices, dtype=getattr(mx, index_dtype)),
            mx.array(sparse.indptr, dtype=getattr(mx, index_dtype)),
        ),
        shape=sparse.shape,
        canonical=True,
        sorted_indices=True,
    )
    return result if format == "csr" else getattr(result, f"to{format}")(canonical=True)


def _start(mx, n, kind):
    if kind == "default":
        return None
    if kind == "coordinate":
        return mx.array(np.eye(n, dtype=np.float32)[0])
    if kind == "constant":
        return mx.ones((n,))
    values = np.arange(1, n + 1, dtype=np.float32) * (-1.0) ** np.arange(n)
    storage = np.zeros(2 * n, dtype=np.float32)
    storage[::2] = values * (1e-30 if kind == "tiny" else 1e30)
    return mx.array(storage)[::2]


@pytest.mark.parametrize("start", ["default", "coordinate", "constant", "tiny", "huge"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_eigs_reconstructs_ritz_vectors(mx, scipy_sparse, to_numpy, start, index_dtype):
    dense = np.diag(np.arange(1, 7, dtype=np.float32))
    a = _matrix(mx, scipy_sparse, dense, index_dtype)
    values, vectors = linalg.eigs(a, k=2, ncv=6, v0=_start(mx, 6, start))
    values, vectors = to_numpy(values), to_numpy(vectors)
    np.testing.assert_allclose(np.sort(values), [5, 6], atol=2e-5)
    np.testing.assert_allclose(dense @ vectors, vectors * values, atol=2e-5)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=0), 1, atol=2e-6)


@pytest.mark.parametrize(
    "which, expected",
    [("LM", [4j, -4j]), ("SM", [1j, -1j]), ("LR", [3, 2]), ("SR", [-3, -2])],
)
@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("start", ["default", "coordinate", "constant"])
def test_eigs_complex_spectrum_and_selection(
    mx, scipy_sparse, to_numpy, which, expected, format, start
):
    dense = np.diag(np.array([3, 2, -2, -3, 0, 0, 0, 0], dtype=np.float32))
    dense[4:6, 4:6] = [[0, -4], [4, 0]]
    dense[6:8, 6:8] = [[0, -1], [1, 0]]
    a = _matrix(mx, scipy_sparse, dense, format=format)
    kwargs = dict(k=2, ncv=8, which=which, v0=_start(mx, 8, start))
    values, vectors = linalg.eigs(a, **kwargs)
    assert values.dtype == mx.complex64
    assert vectors.dtype == mx.complex64
    values, vectors = to_numpy(values), to_numpy(vectors)
    np.testing.assert_allclose(
        np.sort_complex(values), np.sort_complex(expected), atol=3e-5
    )
    np.testing.assert_allclose(dense @ vectors, vectors * values, atol=3e-5)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(2), atol=3e-5)
    only_values = linalg.eigs(a, **kwargs, return_eigenvectors=False)
    np.testing.assert_allclose(to_numpy(only_values), values, atol=3e-5)


@pytest.mark.parametrize("scale", [1e-15, 1.0, 1e15])
def test_eigs_nonnormal_vectors_and_projected_residual(
    mx, scipy_sparse, to_numpy, scale
):
    dense = np.diag(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    dense += np.diag(np.full(5, 0.5, dtype=np.float32), 1)
    a = _matrix(mx, scipy_sparse, dense * scale)
    values, vectors = linalg.eigs(a, k=2, ncv=6)
    values, vectors = to_numpy(values) / scale, to_numpy(vectors)
    np.testing.assert_allclose(np.sort_complex(values), [5, 6], atol=4e-5)
    np.testing.assert_allclose(dense @ vectors, vectors * values, atol=4e-5)
    assert abs(np.vdot(vectors[:, 0], vectors[:, 1])) > 0.1
    values, vectors = linalg.eigs(a, k=2, ncv=4)
    values, vectors = to_numpy(values) / scale, to_numpy(vectors)
    q, _ = np.linalg.qr(vectors)
    residual = dense @ vectors - vectors * values
    np.testing.assert_allclose(q.conj().T @ residual, 0, atol=4e-5)


def test_eigs_single_member_of_conjugate_pair(mx, scipy_sparse, to_numpy):
    dense = np.array(
        [[0, -2, 0, 0], [2, 0, 0, 0], [0, 0, 0, -0.5], [0, 0, 0.5, 0]], dtype=np.float32
    )
    a = _matrix(mx, scipy_sparse, dense)
    values, vectors = linalg.eigs(a, k=1, ncv=4)
    values, vectors = to_numpy(values), to_numpy(vectors)
    np.testing.assert_allclose(abs(values), [2], atol=1e-5)
    np.testing.assert_allclose(dense @ vectors, vectors * values, atol=1e-5)


def test_eigs_complex_nonnormal_reconstruction(mx, scipy_sparse, to_numpy):
    dense = np.zeros((6, 6), dtype=np.float32)
    dense[0, 0], dense[1, 1] = 3, -3
    dense[2:4, 2:4] = [[0, -4], [4, 0]]
    dense[4:6, 4:6] = [[0, -1], [1, 0]]
    dense[:2, 2:] = 0.5
    dense[2:4, 4:] = [[0.3, -0.2], [0.1, 0.4]]
    a = _matrix(mx, scipy_sparse, dense)
    values, vectors = linalg.eigs(a, k=4, ncv=6, v0=_start(mx, 6, "coordinate"))
    values, vectors = to_numpy(values), to_numpy(vectors)
    np.testing.assert_allclose(np.sort(abs(values)), [3, 3, 4, 4], atol=3e-5)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=0), 1, atol=3e-6)
    np.testing.assert_allclose(dense @ vectors, vectors * values, atol=3e-5)


def test_eigs_zero_matrix_has_normalized_vectors(mx, scipy_sparse, to_numpy):
    a = _matrix(mx, scipy_sparse, np.zeros((6, 6), dtype=np.float32))
    values, vectors = linalg.eigs(a, k=3, ncv=6)
    np.testing.assert_array_equal(to_numpy(values), 0)
    vectors = to_numpy(vectors)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(3), atol=3e-6)


def _svd_example(shape, kind):
    m, n = shape
    dense = np.zeros(shape, dtype=np.float32)
    values = [0, 0, 1, 2, 3, 4] if kind == "rank_deficient" else [1, 1, 2, 2, 3, 4]
    if kind != "zero":
        dense[np.arange(6), np.arange(6)] = values
    if kind == "rotated_rank_deficient":
        rng = np.random.default_rng(123)
        u = np.linalg.qr(rng.normal(size=(m, m)))[0]
        v = np.linalg.qr(rng.normal(size=(n, n)))[0]
        dense[:] = 0
        dense[np.arange(6), np.arange(6)] = [0, 0, 1, 2, 3, 4]
        dense = (u @ dense @ v.T).astype(np.float32)
    return dense


@pytest.mark.parametrize("shape", [(6, 6), (9, 6), (6, 9)])
@pytest.mark.parametrize(
    "kind", ["zero", "rank_deficient", "rotated_rank_deficient", "repeated"]
)
@pytest.mark.parametrize("start", ["default", "coordinate", "constant"])
@pytest.mark.parametrize("which", ["LM", "SM"])
def test_svds_singular_triplets_include_nullspaces(
    mx, scipy_sparse, to_numpy, shape, kind, start, which
):
    dense = _svd_example(shape, kind)
    a = _matrix(mx, scipy_sparse, dense, "int64")
    u, s, vh = linalg.svds(
        a, k=3, ncv=min(shape), v0=_start(mx, shape[1], start), which=which
    )
    u, s, vh = to_numpy(u), to_numpy(s), to_numpy(vh)
    expected = np.linalg.svd(dense.astype(np.float64), compute_uv=False)
    expected = expected[:3] if which == "LM" else expected[-3:][::-1]
    np.testing.assert_allclose(s, expected, atol=3e-5)
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=3e-5)
    np.testing.assert_allclose(vh @ vh.T, np.eye(3), atol=3e-5)
    np.testing.assert_allclose(dense @ vh.T, u * s, atol=3e-5)
    np.testing.assert_allclose(dense.T @ u, vh.T * s, atol=3e-5)


@pytest.mark.parametrize("scale", [1e-25, 1.0, 1e25])
@pytest.mark.parametrize("start", ["coordinate", "constant", "tiny", "huge"])
def test_svds_small_singular_values_do_not_square_condition_number(
    mx, scipy_sparse, to_numpy, scale, start
):
    diagonal = np.array([1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], dtype=np.float32)
    dense = np.diag(diagonal) * scale
    a = _matrix(mx, scipy_sparse, dense)
    u, s, vh = linalg.svds(a, k=3, ncv=6, which="SM", v0=_start(mx, 6, start))
    u, s, vh = to_numpy(u), to_numpy(s) / scale, to_numpy(vh)
    np.testing.assert_allclose(s, diagonal[-3:][::-1], rtol=0.02, atol=2e-8)
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=3e-5)
    np.testing.assert_allclose(vh @ vh.T, np.eye(3), atol=3e-5)
    np.testing.assert_allclose(np.diag(diagonal) @ vh.T, u * s, atol=3e-6)
    np.testing.assert_allclose(np.diag(diagonal) @ u, vh.T * s, atol=3e-6)


@pytest.mark.parametrize("option", [False, "u", "vh"])
def test_svds_return_options_match_full_triplets(mx, scipy_sparse, to_numpy, option):
    dense = _svd_example((6, 9), "rank_deficient")
    a = _matrix(mx, scipy_sparse, dense)
    full = linalg.svds(a, k=2, ncv=6, which="SM")
    result = linalg.svds(a, k=2, ncv=6, which="SM", return_singular_vectors=option)
    if option is False:
        np.testing.assert_allclose(to_numpy(result), to_numpy(full[1]), atol=3e-6)
    else:
        missing = 2 if option == "u" else 0
        assert result[missing] is None
        for i in range(3):
            if i != missing:
                np.testing.assert_allclose(
                    to_numpy(result[i]), to_numpy(full[i]), atol=3e-6
                )


@pytest.mark.parametrize("shape", [(6, 6), (9, 6), (6, 9)])
@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("which", ["LM", "SM"])
def test_svds_general_rectangular_matrix(
    mx, scipy_sparse, to_numpy, shape, format, which
):
    dense = np.random.default_rng(239).normal(size=shape).astype(np.float32)
    a = _matrix(mx, scipy_sparse, dense, format=format)
    u, s, vh = linalg.svds(a, k=3, ncv=min(shape), which=which)
    u, s, vh = to_numpy(u), to_numpy(s), to_numpy(vh)
    expected = np.linalg.svd(dense.astype(np.float64), compute_uv=False)
    expected = expected[:3] if which == "LM" else expected[-3:][::-1]
    np.testing.assert_allclose(s, expected, atol=3e-5)
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=3e-5)
    np.testing.assert_allclose(vh @ vh.T, np.eye(3), atol=3e-5)
    np.testing.assert_allclose(dense @ vh.T, u * s, atol=3e-5)
    np.testing.assert_allclose(dense.T @ u, vh.T * s, atol=3e-5)


@pytest.mark.parametrize("which", ["LM", "SM"])
def test_svds_partial_projection_has_galerkin_residuals(
    mx, scipy_sparse, to_numpy, which
):
    dense = np.random.default_rng(173).normal(size=(8, 11)).astype(np.float32)
    a = _matrix(mx, scipy_sparse, dense)
    u, s, vh = linalg.svds(a, k=2, ncv=4, which=which)
    u, s, vh = to_numpy(u), to_numpy(s), to_numpy(vh)
    np.testing.assert_allclose(u.T @ u, np.eye(2), atol=3e-5)
    np.testing.assert_allclose(vh @ vh.T, np.eye(2), atol=3e-5)
    np.testing.assert_allclose(u.T @ (dense @ vh.T - u * s), 0, atol=3e-5)
    np.testing.assert_allclose(vh @ (dense.T @ u - vh.T * s), 0, atol=3e-5)


@pytest.mark.parametrize("routine", ["eigs", "svds"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_spectral_extraction_promotes_low_precision_inputs(
    mx, scipy_sparse, to_numpy, routine, dtype
):
    a = _matrix(mx, scipy_sparse, np.diag(np.arange(1, 7, dtype=np.float32)))
    a = ms.csr_array(
        (a.data.astype(getattr(mx, dtype)), a.indices, a.indptr),
        shape=a.shape,
        canonical=True,
        sorted_indices=True,
    )
    result = getattr(linalg, routine)(a, k=2, ncv=6)
    if routine == "eigs":
        values, vectors = result
        assert values.dtype == vectors.dtype == mx.complex64
        np.testing.assert_allclose(np.sort_complex(to_numpy(values)), [5, 6], atol=3e-5)
    else:
        u, s, vh = result
        assert u.dtype == s.dtype == vh.dtype == mx.float32
        np.testing.assert_allclose(to_numpy(s), [6, 5], atol=3e-5)


@pytest.mark.parametrize("shape", ["square", "tall", "wide"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("start", ["coordinate", "constant"])
@pytest.mark.parametrize("which", ["LM", "SM"])
def test_svds_large_sparse_repeated_and_null_subspaces(
    mx, scipy_sparse, to_numpy, shape, index_dtype, start, which
):
    n = 8192 if shape == "square" else 4096
    if which == "SM":
        diagonal = np.zeros(n, dtype=np.float32)
        diagonal[1] = 4
    else:
        diagonal = (1 + np.arange(n) % 4).astype(np.float32)
        diagonal[0] = 0
    sparse = scipy_sparse.diags(diagonal, format="csr")
    if shape != "square":
        sparse = scipy_sparse.vstack([sparse, scipy_sparse.csr_array((n, n))])
        if shape == "wide":
            sparse = sparse.T.tocsr()
    a = _matrix(mx, scipy_sparse, sparse, index_dtype)
    u, s, vh = linalg.svds(
        a, k=3, ncv=20, which=which, v0=_start(mx, a.shape[1], start)
    )
    u, s, vh = to_numpy(u), to_numpy(s), to_numpy(vh)
    np.testing.assert_allclose(s, [4, 4, 4] if which == "LM" else [0, 0, 0], atol=5e-5)
    np.testing.assert_allclose(u.T @ u, np.eye(3), atol=5e-5)
    np.testing.assert_allclose(vh @ vh.T, np.eye(3), atol=5e-5)
    np.testing.assert_allclose(sparse @ vh.T, u * s, atol=5e-5)
    np.testing.assert_allclose(sparse.T @ u, vh.T * s, atol=5e-5)


def test_svds_large_continuation_preserves_global_random_stream(
    mx, scipy_sparse, to_numpy
):
    diagonal = np.zeros(8192, dtype=np.float32)
    diagonal[1] = 4
    a = _matrix(mx, scipy_sparse, scipy_sparse.diags(diagonal, format="csr"))
    mx.random.seed(416)
    expected = to_numpy(mx.random.normal((16,)))
    mx.random.seed(416)
    values = linalg.svds(
        a,
        k=3,
        ncv=10,
        which="SM",
        v0=_start(mx, 8192, "coordinate"),
        return_singular_vectors=False,
    )
    mx.eval(values)
    np.testing.assert_array_equal(to_numpy(mx.random.normal((16,))), expected)
