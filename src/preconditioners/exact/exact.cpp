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

#include "preconditioners/exact/exact.h"

#include <stdexcept>

#include "linalg/common/common.h"
#include "linalg/permute_vector/permute_vector.h"
#include "linalg/triangular_solve/triangular_solve.h"
#include "mlx/ops.h"

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

void validate_exact_factor_rhs(const mx::array &rhs, int n_rows,
                               const char *context) {
  if (rhs.ndim() != 1 && rhs.ndim() != 2) {
    throw std::invalid_argument(std::string(context) +
                                " rhs must be rank-1 or rank-2.");
  }
  require_linalg_float32(rhs, context);
  if (rhs.shape(0) != n_rows) {
    throw std::invalid_argument(std::string(context) +
                                " rhs has incompatible leading dimension.");
  }
  if (rhs.ndim() == 2 && rhs.shape(1) <= 0) {
    throw std::invalid_argument(
        std::string(context) + " rank-2 rhs must include at least one column.");
  }
}

void validate_exact_factor_csr(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, int n_rows, int n_cols,
                               const char *context) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(std::string(context) +
                                " requires a non-empty square factor.");
  }
  require_rank(data, 1, context);
  require_rank(indices, 1, context);
  require_rank(indptr, 1, context);
  require_linalg_float32(data, context);
  require_same_index_dtype(indices, indptr, context, context);
  require_size(indptr, n_rows + 1, context);
  if (indices.size() != data.size()) {
    throw std::invalid_argument(std::string(context) +
                                " data and indices must have equal length.");
  }
}

void validate_lu_exact_inputs(const mx::array &perm, const mx::array &l_data,
                              const mx::array &l_indices,
                              const mx::array &l_indptr,
                              const mx::array &u_data,
                              const mx::array &u_indices,
                              const mx::array &u_indptr, const mx::array &rhs,
                              int n_rows, int n_cols) {
  validate_exact_factor_csr(l_data, l_indices, l_indptr, n_rows, n_cols,
                            "csr_exact_lu_preconditioner_apply L");
  validate_exact_factor_csr(u_data, u_indices, u_indptr, n_rows, n_cols,
                            "csr_exact_lu_preconditioner_apply U");
  validate_exact_factor_rhs(rhs, n_rows, "csr_exact_lu_preconditioner_apply");
  require_rank(perm, 1, "csr_exact_lu_preconditioner_apply perm");
  if (perm.dtype() != mx::int32) {
    throw std::invalid_argument(
        "csr_exact_lu_preconditioner_apply perm must have dtype int32.");
  }
  require_size(perm, n_rows, "csr_exact_lu_preconditioner_apply perm");
}

void validate_cholesky_exact_inputs(
    const mx::array &l_data, const mx::array &l_indices,
    const mx::array &l_indptr, const mx::array &lt_data,
    const mx::array &lt_indices, const mx::array &lt_indptr,
    const mx::array &rhs, int n_rows, int n_cols) {
  validate_exact_factor_csr(l_data, l_indices, l_indptr, n_rows, n_cols,
                            "csr_exact_cholesky_preconditioner_apply L");
  validate_exact_factor_csr(lt_data, lt_indices, lt_indptr, n_rows, n_cols,
                            "csr_exact_cholesky_preconditioner_apply LT");
  validate_exact_factor_rhs(rhs, n_rows,
                            "csr_exact_cholesky_preconditioner_apply");
}

} // namespace

mx::array csr_exact_lu_preconditioner_apply(
    const mx::array &perm, const mx::array &l_data, const mx::array &l_indices,
    const mx::array &l_indptr, const mx::array &u_data,
    const mx::array &u_indices, const mx::array &u_indptr, const mx::array &rhs,
    int n_rows, int n_cols, mx::StreamOrDevice s) {
  validate_lu_exact_inputs(perm, l_data, l_indices, l_indptr, u_data, u_indices,
                           u_indptr, rhs, n_rows, n_cols);
  auto stream = mx::to_stream(s);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  auto permuted = csr_permute_vector(rhs_contig, perm, stream);
  auto y = csr_triangular_solve(l_data, l_indices, l_indptr, permuted, n_rows,
                                n_cols, true, true, stream);
  return csr_triangular_solve(u_data, u_indices, u_indptr, y, n_rows, n_cols,
                              false, false, stream);
}

mx::array csr_exact_cholesky_preconditioner_apply(
    const mx::array &l_data, const mx::array &l_indices,
    const mx::array &l_indptr, const mx::array &lt_data,
    const mx::array &lt_indices, const mx::array &lt_indptr,
    const mx::array &rhs, int n_rows, int n_cols, mx::StreamOrDevice s) {
  validate_cholesky_exact_inputs(l_data, l_indices, l_indptr, lt_data,
                                 lt_indices, lt_indptr, rhs, n_rows, n_cols);
  auto stream = mx::to_stream(s);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  auto y = csr_triangular_solve(l_data, l_indices, l_indptr, rhs_contig, n_rows,
                                n_cols, true, false, stream);
  return csr_triangular_solve(lt_data, lt_indices, lt_indptr, y, n_rows, n_cols,
                              false, false, stream);
}

} // namespace mlx_sparse
