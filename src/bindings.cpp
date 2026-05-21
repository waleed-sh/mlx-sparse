// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "sparse/coo_tocsr.h"
#include "sparse/csr_linalg.h"
#include "sparse/csr_matmul.h"
#include "sparse/csr_matvec.h"
#include "sparse/csr_sort_indices.h"
#include "sparse/csr_todense.h"
#include "sparse/csr_transpose.h"
#include "sparse/identity_like.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native sparse primitives for MLX";

  m.def(
      "identity_like",
      [](const mlx_sparse::mx::array &x) {
        return mlx_sparse::identity_like(x);
      },
      "x"_a, "Return a native MLX copy of x. Used as an extension smoke test.");

  m.def(
      "coo_tocsr",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, int n_rows, int n_cols) {
        return mlx_sparse::coo_tocsr(data, row, col, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "Convert COO buffers to row-sorted CSR buffers, preserving duplicates.");

  m.def(
      "csr_todense",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_todense(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Materialize CSR buffers as a dense MLX array.");

  m.def(
      "csr_sort_indices",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr) {
        return mlx_sparse::csr_sort_indices(data, indices, indptr);
      },
      "data"_a, "indices"_a, "indptr"_a,
      "Sort CSR column indices independently within each row.");

  m.def(
      "csr_transpose",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_transpose(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Transpose CSR buffers into a new row-sorted CSR representation.");

  m.def(
      "csr_matvec",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &x,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_matvec(data, indices, indptr, x, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "x"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSR buffers by a dense vector.");

  m.def(
      "csr_matvec_transpose",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &x,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_matvec_transpose(data, indices, indptr, x,
                                                n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "x"_a, "n_rows"_a, "n_cols"_a,
      "Multiply the transpose of CSR buffers by a dense vector.");

  m.def(
      "csr_matmul",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_matmul(data, indices, indptr, rhs, n_rows,
                                      n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSR buffers by a dense matrix.");

  m.def(
      "csr_matmul_transpose",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_matmul_transpose(data, indices, indptr, rhs,
                                                n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply the transpose of CSR buffers by a dense matrix.");

  m.def(
      "csr_cg",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &b,
         const mlx_sparse::mx::array &x0, int n_rows, int n_cols, float rtol,
         float atol, int maxiter) {
        return mlx_sparse::csr_cg(data, indices, indptr, b, x0, n_rows, n_cols,
                                  rtol, atol, maxiter);
      },
      "data"_a, "indices"_a, "indptr"_a, "b"_a, "x0"_a, "n_rows"_a,
      "n_cols"_a, "rtol"_a, "atol"_a, "maxiter"_a,
      "Solve a float32 SPD CSR system with conjugate gradients.");

  m.def(
      "csr_lanczos",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &v0,
         int n_rows, int n_cols, int k, bool reorthogonalize) {
        return mlx_sparse::csr_lanczos(data, indices, indptr, v0, n_rows,
                                       n_cols, k, reorthogonalize);
      },
      "data"_a, "indices"_a, "indptr"_a, "v0"_a, "n_rows"_a, "n_cols"_a,
      "k"_a, "reorthogonalize"_a,
      "Run a float32 CSR Lanczos basis construction.");

  m.def(
      "csr_gmres",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &b,
         const mlx_sparse::mx::array &x0, int n_rows, int n_cols, float rtol,
         float atol, int restart, int maxiter) {
        return mlx_sparse::csr_gmres(data, indices, indptr, b, x0, n_rows,
                                     n_cols, rtol, atol, restart, maxiter);
      },
      "data"_a, "indices"_a, "indptr"_a, "b"_a, "x0"_a, "n_rows"_a,
      "n_cols"_a, "rtol"_a, "atol"_a, "restart"_a, "maxiter"_a,
      "Solve a float32 CSR system with restarted GMRES.");

  m.def(
      "csr_minres",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &b,
         const mlx_sparse::mx::array &x0, int n_rows, int n_cols, float rtol,
         float atol, int maxiter) {
        return mlx_sparse::csr_minres(data, indices, indptr, b, x0, n_rows,
                                      n_cols, rtol, atol, maxiter);
      },
      "data"_a, "indices"_a, "indptr"_a, "b"_a, "x0"_a, "n_rows"_a,
      "n_cols"_a, "rtol"_a, "atol"_a, "maxiter"_a,
      "Solve a float32 Hermitian CSR system with MINRES.");

  m.def(
      "csr_arnoldi",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &v0,
         int n_rows, int n_cols, int k) {
        return mlx_sparse::csr_arnoldi(data, indices, indptr, v0, n_rows,
                                      n_cols, k);
      },
      "data"_a, "indices"_a, "indptr"_a, "v0"_a, "n_rows"_a, "n_cols"_a,
      "k"_a, "Run a float32 CSR Arnoldi basis construction.");

  m.def(
      "csr_eigsh",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols, int k,
         int ncv, const std::string &which) {
        return mlx_sparse::csr_eigsh(data, indices, indptr, n_rows, n_cols, k,
                                     ncv, which);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a, "k"_a,
      "ncv"_a, "which"_a,
      "Compute selected Hermitian Ritz pairs from a CSR matrix.");

  m.def(
      "csr_eigs",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols, int k,
         int ncv, const std::string &which) {
        return mlx_sparse::csr_eigs(data, indices, indptr, n_rows, n_cols, k,
                                    ncv, which);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a, "k"_a,
      "ncv"_a, "which"_a,
      "Compute selected Arnoldi Ritz pairs from a CSR matrix.");

  m.def(
      "csr_svds",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols, int k,
         int ncv, const std::string &which) {
        return mlx_sparse::csr_svds(data, indices, indptr, n_rows, n_cols, k,
                                    ncv, which);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a, "k"_a,
      "ncv"_a, "which"_a,
      "Compute selected singular triplets from a CSR matrix.");

  m.def(
      "csr_cholesky",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_cholesky(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute a sparse left-looking Cholesky factor in CSR format.");

  m.def(
      "csr_lu",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_lu(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute sparse LU factors with partial pivoting in CSR format.");

  m.def(
      "csr_triangular_solve",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &b,
         int n_rows, int n_cols, bool lower, bool unit_diagonal) {
        return mlx_sparse::csr_triangular_solve(
            data, indices, indptr, b, n_rows, n_cols, lower, unit_diagonal);
      },
      "data"_a, "indices"_a, "indptr"_a, "b"_a, "n_rows"_a, "n_cols"_a,
      "lower"_a, "unit_diagonal"_a,
      "Solve a sparse triangular CSR system.");

  m.def(
      "csr_vdot",
      [](const mlx_sparse::mx::array &lhs_data,
         const mlx_sparse::mx::array &lhs_indices,
         const mlx_sparse::mx::array &lhs_indptr,
         const mlx_sparse::mx::array &rhs_data,
         const mlx_sparse::mx::array &rhs_indices,
         const mlx_sparse::mx::array &rhs_indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_vdot(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                                    rhs_indices, rhs_indptr, n_rows, n_cols);
      },
      "lhs_data"_a, "lhs_indices"_a, "lhs_indptr"_a, "rhs_data"_a,
      "rhs_indices"_a, "rhs_indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute the sparse Frobenius inner product of two CSR arrays.");

  m.def(
      "csr_dot",
      [](const mlx_sparse::mx::array &lhs_data,
         const mlx_sparse::mx::array &lhs_indices,
         const mlx_sparse::mx::array &lhs_indptr,
         const mlx_sparse::mx::array &rhs_data,
         const mlx_sparse::mx::array &rhs_indices,
         const mlx_sparse::mx::array &rhs_indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_dot(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                                   rhs_indices, rhs_indptr, n_rows, n_cols);
      },
      "lhs_data"_a, "lhs_indices"_a, "lhs_indptr"_a, "rhs_data"_a,
      "rhs_indices"_a, "rhs_indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute the sparse Frobenius dot product of two CSR arrays.");

  m.def(
      "csr_permute_vector",
      [](const mlx_sparse::mx::array &x,
         const mlx_sparse::mx::array &perm) {
        return mlx_sparse::csr_permute_vector(x, perm);
      },
      "x"_a, "perm"_a, "Apply an int32 permutation to a float32 vector.");
}
