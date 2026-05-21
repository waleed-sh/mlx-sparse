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
#include <nanobind/stl/tuple.h>

#include "sparse/coo_tocsr.h"
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
}
