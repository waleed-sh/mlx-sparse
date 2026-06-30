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

#include "common/common.h"

namespace mlx_sparse {

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab_jacobi(const mx::array &data, const mx::array &indices,
                    const mx::array &indptr, const mx::array &b,
                    const mx::array &x0, const mx::array &inv_diag, int n_rows,
                    int n_cols, float rtol, float atol, int maxiter,
                    mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array, mx::array, mx::array> csr_bicgstab_exact_lu(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, const mx::array &x0, const mx::array &perm,
    const mx::array &l_data, const mx::array &l_indices,
    const mx::array &l_indptr, const mx::array &u_data,
    const mx::array &u_indices, const mx::array &u_indptr, int n_rows,
    int n_cols, float rtol, float atol, int maxiter, mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab_ilu0(const mx::array &data, const mx::array &indices,
                  const mx::array &indptr, const mx::array &b,
                  const mx::array &x0, const mx::array &l_data,
                  const mx::array &l_indices, const mx::array &l_indptr,
                  const mx::array &u_data, const mx::array &u_indices,
                  const mx::array &u_indptr, int n_rows, int n_cols, float rtol,
                  float atol, int maxiter, mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab_exact_cholesky(const mx::array &data, const mx::array &indices,
                            const mx::array &indptr, const mx::array &b,
                            const mx::array &x0, const mx::array &l_data,
                            const mx::array &l_indices,
                            const mx::array &l_indptr, const mx::array &lt_data,
                            const mx::array &lt_indices,
                            const mx::array &lt_indptr, int n_rows, int n_cols,
                            float rtol, float atol, int maxiter,
                            mx::StreamOrDevice s = {});

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
class AccelerateFloatSolve;

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab_exact_accelerate(const mx::array &data, const mx::array &indices,
                              const mx::array &indptr, const mx::array &b,
                              const mx::array &x0,
                              const AccelerateFloatSolve &solver, int n_rows,
                              int n_cols, float rtol, float atol, int maxiter,
                              mx::StreamOrDevice s = {});
#endif

} // namespace mlx_sparse
