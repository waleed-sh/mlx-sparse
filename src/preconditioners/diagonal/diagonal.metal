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

#include "common/metal_common.h"

[[host_name("diagonal_preconditioner_apply_float32")]] [[kernel]] void
diagonal_preconditioner_apply_kernel(device const float *inv_diag [[buffer(0)]],
                                     device const float *rhs [[buffer(1)]],
                                     device float *out [[buffer(2)]],
                                     constant int &n_rows [[buffer(3)]],
                                     constant int &rhs_cols [[buffer(4)]],
                                     uint tid [[thread_position_in_grid]]) {
  const int total = n_rows * rhs_cols;
  if (static_cast<int>(tid) >= total) {
    return;
  }
  const int row = static_cast<int>(tid) / rhs_cols;
  out[tid] = inv_diag[row] * rhs[tid];
}
