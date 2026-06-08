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
#include "mlx/utils.h"

namespace mlx_sparse {

namespace mx = mlx::core;

mx::array coo_kron_data(const mx::array &lhs_data, const mx::array &rhs_data,
                        mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array>
coo_kron_indices(const mx::array &lhs_row, const mx::array &lhs_col,
                 const mx::array &rhs_row, const mx::array &rhs_col,
                 int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
                 mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array, mx::array>
coo_kron(const mx::array &lhs_data, const mx::array &lhs_row,
         const mx::array &lhs_col, const mx::array &rhs_data,
         const mx::array &rhs_row, const mx::array &rhs_col, int lhs_n_rows,
         int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
         mx::StreamOrDevice s = {});

} // namespace mlx_sparse
