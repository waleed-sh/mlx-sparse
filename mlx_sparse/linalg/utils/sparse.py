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

"""Sparse input normalization helpers for linalg routines."""

from __future__ import annotations

from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse._validation import normalize_shape


def square_shape(A_or_shape) -> tuple[int, int]:
    """Return a normalized square shape from a matrix, integer, or shape tuple."""

    if isinstance(A_or_shape, int):
        shape = (int(A_or_shape), int(A_or_shape))
    elif hasattr(A_or_shape, "shape"):
        shape = A_or_shape.shape
    else:
        shape = A_or_shape
    shape = normalize_shape(shape)
    if shape[0] != shape[1]:
        raise ValueError(f"preconditioners require a square shape, got {shape}.")
    return shape


def as_sparse(
    A,
    *,
    context: str,
    dense_guidance: str,
) -> CSRArray | CSCArray | COOArray:
    """Validate that ``A`` is one of the public sparse array containers."""

    if isinstance(A, (CSRArray, CSCArray, COOArray)):
        return A
    suffix = f" {dense_guidance}" if dense_guidance else ""
    raise TypeError(f"{context} expects CSRArray, COOArray, or CSCArray.{suffix}")


def as_csr(
    A,
    *,
    context: str,
    dense_guidance: str = "",
    canonicalize_csr: bool = False,
) -> CSRArray:
    """Return a public sparse matrix container as CSR.

    ``canonicalize_csr=False`` preserves the historical sparse inner-product
    behavior where CSR inputs are passed through unchanged.
    """

    if isinstance(A, CSRArray):
        return A.canonicalize() if canonicalize_csr else A
    if isinstance(A, COOArray):
        return A.tocsr(canonical=True)
    if isinstance(A, CSCArray):
        return A.tocsr(canonical=True)
    suffix = f" {dense_guidance}" if dense_guidance else ""
    raise TypeError(f"{context} expected CSRArray, COOArray, or CSCArray.{suffix}")


def inner_product_csr(A) -> CSRArray:
    """Return sparse inner-product input as CSR without canonicalizing CSR input."""

    return as_csr(A, context="sparse inner products")


def canonical_csr(
    A,
    *,
    context: str,
    dense_guidance: str,
    allow_sparse_linear_operator: bool = False,
) -> CSRArray:
    """Return ``A`` as canonical CSR without mutating the original object.

    Args:
        A: Sparse matrix input, or a sparse-backed ``LinearOperator`` when
            ``allow_sparse_linear_operator=True``.
        context: Human-readable caller name for error messages.
        dense_guidance: Sentence appended to unsupported-type errors.
        allow_sparse_linear_operator: Whether to unwrap sparse-backed linalg
            ``LinearOperator`` instances.

    Returns:
        Canonical CSR representation.
    """

    if isinstance(A, CSRArray):
        return A.canonicalize()
    if isinstance(A, COOArray):
        return A.tocsr(canonical=True)
    if isinstance(A, CSCArray):
        return A.tocsr(canonical=True)
    if allow_sparse_linear_operator:
        from mlx_sparse.linalg._interface import LinearOperator

        if isinstance(A, LinearOperator):
            if A._sparse_array is not None:
                return canonical_csr(
                    A._sparse_array,
                    context=context,
                    dense_guidance=dense_guidance,
                    allow_sparse_linear_operator=False,
                )
            raise TypeError(
                f"{context} accept LinearOperator only when it wraps a "
                "CSRArray, COOArray, or CSCArray (use aslinearoperator(sparse_array)). "
                "For fully matrix-free operators, implement a Python-level "
                "iterative solver loop."
            )
    accepted = "CSRArray, COOArray, CSCArray"
    if allow_sparse_linear_operator:
        accepted += ", or a sparse-backed LinearOperator"
    suffix = f" {dense_guidance}" if dense_guidance else ""
    raise TypeError(f"{context} expects {accepted}.{suffix}")
