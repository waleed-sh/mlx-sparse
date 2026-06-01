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
csr_pcg_jacobi(const mx::array &data, const mx::array &indices,
               const mx::array &indptr, const mx::array &b, const mx::array &x0,
               const mx::array &inv_diag, int n_rows, int n_cols, float rtol,
               float atol, int maxiter, mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_pcg_ic0(const mx::array &data, const mx::array &indices,
            const mx::array &indptr, const mx::array &b, const mx::array &x0,
            const mx::array &l_data, const mx::array &l_indices,
            const mx::array &l_indptr, const mx::array &lt_data,
            const mx::array &lt_indices, const mx::array &lt_indptr, int n_rows,
            int n_cols, float rtol, float atol, int maxiter,
            mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array, mx::array, mx::array> csr_pcg_chebyshev(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, const mx::array &x0, const mx::array &m_data,
    const mx::array &m_indices, const mx::array &m_indptr, int n_rows,
    int n_cols, int degree, float lambda_min, float lambda_max, float rtol,
    float atol, int maxiter, mx::StreamOrDevice s = {});

} // namespace mlx_sparse
