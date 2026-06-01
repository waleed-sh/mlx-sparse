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

std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array>
csr_ilu0(const mx::array &data, const mx::array &indices,
         const mx::array &indptr, int n_rows, int n_cols, float shift,
         bool check);

mx::array csr_ilu0_preconditioner_apply(
    const mx::array &l_data, const mx::array &l_indices,
    const mx::array &l_indptr, const mx::array &u_data,
    const mx::array &u_indices, const mx::array &u_indptr, const mx::array &rhs,
    int n_rows, int n_cols, mx::StreamOrDevice s = {});

} // namespace mlx_sparse
