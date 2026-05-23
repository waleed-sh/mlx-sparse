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

#include "common/common.h"

namespace mlx_sparse {

mx::array csr_vdot(const mx::array &lhs_data, const mx::array &lhs_indices,
                   const mx::array &lhs_indptr, const mx::array &rhs_data,
                   const mx::array &rhs_indices, const mx::array &rhs_indptr,
                   int n_rows, int n_cols, mx::StreamOrDevice s = {});

mx::array csr_dot(const mx::array &lhs_data, const mx::array &lhs_indices,
                  const mx::array &lhs_indptr, const mx::array &rhs_data,
                  const mx::array &rhs_indices, const mx::array &rhs_indptr,
                  int n_rows, int n_cols, mx::StreamOrDevice s = {});

} // namespace mlx_sparse
