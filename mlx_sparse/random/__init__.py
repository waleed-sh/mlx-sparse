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

"""Native sparse random constructors for mlx-sparse.

The :mod:`mlx_sparse.random` namespace is the SciPy-compatible public home for
random sparse matrix generation. The v0.0.6b0 skeleton fixes the argument
contract, RNG policy, density semantics, supported formats, and documentation
before the native C++/Metal generation kernels are wired in.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_sparse.random._validation import (
    normalize_dimension,
    normalize_random_array_args,
)


def random_array(
    shape,
    *,
    density=0.01,
    format="coo",
    dtype=None,
    rng=None,
    data_sampler=None,
    random_state=None,
    index_dtype=mx.int32,
    canonical=True,
):
    """Return a random sparse array with duplicate-free sampled coordinates.

    This function follows the SciPy ``random_array`` argument surface while
    returning mlx-sparse public array containers once the native generation
    kernels land. The output is rank-2 with ``shape=(m, n)`` and stores exactly
    ``round(m * n * density)`` structural nonzeros, clipped to the valid range
    ``[0, m * n]``. The rounding rule intentionally matches SciPy's
    ``int(round(...))`` behavior, including Python's ties-to-even rounding.

    Args:
        shape: Two non-negative integer dimensions ``(m, n)``. Rank-2 output is
            required; higher-rank sparse tensors are not part of this API.
        density: Fraction of matrix entries to sample, with
            ``0 <= density <= 1``. The resulting structural ``nnz`` is
            deterministic for a given ``shape`` and ``density``.
        format: Sparse output format. Supported values are ``"coo"``,
            ``"csr"``, ``"csc"``, and ``None`` (treated as ``"coo"``).
            SciPy-only formats such as ``"bsr"``, ``"dia"``, ``"dok"``, and
            ``"lil"`` are rejected clearly.
        dtype: Stored-value dtype. ``None`` defaults to ``mx.float32``. The
            current sparse containers accept ``mx.float32``, ``mx.float16``,
            ``mx.bfloat16``, and ``mx.complex64``. Native generation for
            ``mx.float64`` where available, integer dtypes, and ``mx.bool_`` is
            reserved for the value-kernel implementation and is not silently
            cast in this skeleton.
        rng: Random source. Pass an MLX PRNG key from ``mx.random.key(seed)``
            for reproducible CPU/Metal results, or pass an integer seed to
            create one. ``None`` is accepted for API compatibility and will use
            the native MLX random source when generation is implemented.
        data_sampler: Optional callable used for stored values. It will be
            called at most once with ``size=nnz`` by the native implementation.
            MLX array results keep lazy device execution; NumPy results are an
            explicit host-compatibility path and are not a benchmark path.
        random_state: SciPy compatibility alias for ``rng``. Passing both
            ``rng`` and ``random_state`` is an error.
        index_dtype: Integer dtype for structural buffers. Must be
            ``mx.int32`` or ``mx.int64``. Dimensions and ``nnz`` must fit the
            selected dtype.
        canonical: Whether the result should be duplicate-free and sorted in
            the requested format. Only ``True`` is accepted in the skeleton;
            noncanonical duplicate-preserving random output is not implemented.

    Returns:
        A ``COOArray``, ``CSRArray``, or ``CSCArray`` on the active MLX device
        once native generation is implemented. Reusing the same MLX key and
        inputs will be reproducible across CPU and Metal for the same package
        version; split keys should produce distinct structures and values.

    Raises:
        TypeError: If dimensions, dtypes, RNG objects, or samplers have invalid
            types. NumPy ``Generator`` and ``RandomState`` objects are rejected
            because they imply host-side structure generation.
        ValueError: If shape, density, format, seed, or alias combinations are
            invalid.
        OverflowError: If shape, ``m * n``, or ``nnz`` exceed native/index
            capacity.
        NotImplementedError: Until the native C++/Metal sparse random
            generation primitive is connected.

    Example::

        import mlx.core as mx
        import mlx_sparse as ms

        key = mx.random.key(0)
        A = ms.random.random_array((1024, 1024), density=0.01, format="csr", rng=key)
    """
    normalize_random_array_args(
        shape,
        density=density,
        format=format,
        dtype=dtype,
        rng=rng,
        random_state=random_state,
        index_dtype=index_dtype,
        canonical=canonical,
        sampler=data_sampler,
    )
    raise _native_random_not_implemented()


def random(
    m,
    n,
    density=0.01,
    format="coo",
    dtype=None,
    rng=None,
    data_rvs=None,
    *,
    random_state=None,
    index_dtype=mx.int32,
    canonical=True,
):
    """Return a random sparse matrix with SciPy-compatible ``random`` naming.

    This is the two-dimension convenience wrapper for
    :func:`random_array`. It accepts non-negative integer dimensions ``m`` and
    ``n`` instead of a shape tuple, samples exactly
    ``round(m * n * density)`` duplicate-free structural nonzeros in canonical
    mode, and returns the requested sparse ``format`` once native generation is
    implemented.

    Args:
        m: Number of rows. Must be a non-negative integer.
        n: Number of columns. Must be a non-negative integer.
        density: Fraction of entries to sample, with ``0 <= density <= 1``.
            The density-to-``nnz`` rule is deterministic and matches
            :func:`random_array`.
        format: ``"coo"``, ``"csr"``, ``"csc"``, or ``None`` for ``"coo"``.
            Unsupported SciPy formats are rejected.
        dtype: Stored-value dtype. ``None`` defaults to ``mx.float32``. The
            current sparse containers accept ``mx.float32``, ``mx.float16``,
            ``mx.bfloat16``, and ``mx.complex64``; ``float64``, integer, and
            bool value generation are reserved for native kernels.
        rng: ``None``, an integer seed, or an MLX PRNG key. Use
            ``mx.random.key(seed)`` for reproducible CPU/Metal output.
        data_rvs: SciPy-compatible name for the optional value sampler. The
            sampler is called at most once with ``size=nnz`` by the native
            implementation.
        random_state: Compatibility alias for ``rng``. Passing both aliases is
            an error.
        index_dtype: ``mx.int32`` or ``mx.int64`` structural dtype.
        canonical: Require duplicate-free canonical structure. Only ``True`` is
            supported in the skeleton.

    Returns:
        A ``COOArray``, ``CSRArray``, or ``CSCArray`` on the active MLX device
        once native C++/Metal generation lands. Same key plus same arguments is
        the reproducibility contract; no SciPy or NumPy bitstream equivalence is
        promised.

    Raises:
        TypeError: For invalid dimensions, dtype, index dtype, RNG, or sampler.
        ValueError: For invalid density, format, seed, or RNG alias use.
        OverflowError: If the shape or structural count cannot fit native or
            index buffers.
        NotImplementedError: Until native generation is connected.
    """
    m = normalize_dimension("m", m)
    n = normalize_dimension("n", n)
    return random_array(
        (m, n),
        density=density,
        format=format,
        dtype=dtype,
        rng=rng,
        data_sampler=data_rvs,
        random_state=random_state,
        index_dtype=index_dtype,
        canonical=canonical,
    )


def rand(
    m,
    n,
    density=0.01,
    format="coo",
    dtype=None,
    rng=None,
    *,
    random_state=None,
    index_dtype=mx.int32,
    canonical=True,
):
    """Return a uniformly valued random sparse matrix.

    ``rand`` is the SciPy-compatible convenience wrapper for
    :func:`random` without a custom value sampler. It uses shape ``(m, n)``,
    deterministic density rounding, the requested sparse ``format``, and the
    same RNG and reproducibility policy as :func:`random_array`.

    Args:
        m: Number of rows. Must be a non-negative integer.
        n: Number of columns. Must be a non-negative integer.
        density: Fraction of entries to sample, with ``0 <= density <= 1``.
        format: ``"coo"``, ``"csr"``, ``"csc"``, or ``None`` for ``"coo"``.
        dtype: Stored-value dtype. ``None`` defaults to ``mx.float32``. The
            current sparse containers accept ``mx.float32``, ``mx.float16``,
            ``mx.bfloat16``, and ``mx.complex64``; ``float64``, integer, and
            bool value generation are reserved for native kernels.
        rng: ``None``, an integer seed, or an MLX PRNG key. MLX keys are the
            first-class reproducible path across CPU and Metal devices.
        random_state: Compatibility alias for ``rng``. Passing both aliases is
            an error.
        index_dtype: ``mx.int32`` or ``mx.int64`` structural dtype.
        canonical: Require duplicate-free canonical structure. Only ``True`` is
            supported in the skeleton.

    Returns:
        A ``COOArray``, ``CSRArray``, or ``CSCArray`` on the active MLX device
        once native C++/Metal generation lands.

    Raises:
        TypeError: For invalid dimensions, dtype, index dtype, or RNG.
        ValueError: For invalid density, format, seed, or RNG alias use.
        OverflowError: If shape or structural counts exceed native/index
            capacity.
        NotImplementedError: Until native generation is connected.
    """
    return random(
        m,
        n,
        density=density,
        format=format,
        dtype=dtype,
        rng=rng,
        random_state=random_state,
        index_dtype=index_dtype,
        canonical=canonical,
    )


def _native_random_not_implemented() -> NotImplementedError:
    return NotImplementedError(
        "mlx_sparse.random validates the public v0.0.6b0 API, but native "
        "CPU/Metal sparse random generation is not implemented yet. The "
        "production path must be provided by C++/Metal kernels, not Python "
        "host-side structure generation."
    )


__all__ = ["random_array", "random", "rand"]
