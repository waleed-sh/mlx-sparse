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

#pragma once

#include <tuple>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "sparse/common.h"

namespace mlx_sparse {

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_cg(const mx::array &data, const mx::array &indices, const mx::array &indptr,
       const mx::array &b, const mx::array &x0, int n_rows, int n_cols,
       float rtol, float atol, int maxiter, mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres(const mx::array &data, const mx::array &indices,
          const mx::array &indptr, const mx::array &b, const mx::array &x0,
          int n_rows, int n_cols, float rtol, float atol, int restart,
          int maxiter);

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_minres(const mx::array &data, const mx::array &indices,
           const mx::array &indptr, const mx::array &b, const mx::array &x0,
           int n_rows, int n_cols, float rtol, float atol, int maxiter);

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_lanczos(const mx::array &data, const mx::array &indices,
            const mx::array &indptr, const mx::array &v0, int n_rows,
            int n_cols, int k, bool reorthogonalize, mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array, mx::array>
csr_arnoldi(const mx::array &data, const mx::array &indices,
            const mx::array &indptr, const mx::array &v0, int n_rows,
            int n_cols, int k, mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array> csr_eigsh(const mx::array &data,
                                           const mx::array &indices,
                                           const mx::array &indptr, int n_rows,
                                           int n_cols, int k, int ncv,
                                           const std::string &which);

std::tuple<mx::array, mx::array> csr_eigs(const mx::array &data,
                                          const mx::array &indices,
                                          const mx::array &indptr, int n_rows,
                                          int n_cols, int k, int ncv,
                                          const std::string &which);

std::tuple<mx::array, mx::array, mx::array>
csr_svds(const mx::array &data, const mx::array &indices,
         const mx::array &indptr, int n_rows, int n_cols, int k, int ncv,
         const std::string &which);

std::tuple<mx::array, mx::array, mx::array>
csr_cholesky(const mx::array &data, const mx::array &indices,
             const mx::array &indptr, int n_rows, int n_cols);

std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array,
           mx::array>
csr_lu(const mx::array &data, const mx::array &indices, const mx::array &indptr,
       int n_rows, int n_cols);

mx::array csr_triangular_solve(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, const mx::array &b,
                               int n_rows, int n_cols, bool lower,
                               bool unit_diagonal, mx::StreamOrDevice s = {});

mx::array csr_vdot(const mx::array &lhs_data, const mx::array &lhs_indices,
                   const mx::array &lhs_indptr, const mx::array &rhs_data,
                   const mx::array &rhs_indices, const mx::array &rhs_indptr,
                   int n_rows, int n_cols, mx::StreamOrDevice s = {});

mx::array csr_dot(const mx::array &lhs_data, const mx::array &lhs_indices,
                  const mx::array &lhs_indptr, const mx::array &rhs_data,
                  const mx::array &rhs_indices, const mx::array &rhs_indptr,
                  int n_rows, int n_cols, mx::StreamOrDevice s = {});

mx::array csr_permute_vector(const mx::array &x, const mx::array &perm,
                             mx::StreamOrDevice s = {});

} // namespace mlx_sparse
