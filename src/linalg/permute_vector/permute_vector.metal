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

#include "linalg/common/metal_common.h"

[[host_name("csr_permute_vector_float32")]] [[kernel]] void
csr_permute_vector_float32_kernel(
    device const float *x [[buffer(0)]], device const int *perm [[buffer(1)]],
    device float *out [[buffer(2)]], constant int &size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (static_cast<int>(tid) >= size) {
    return;
  }
  out[tid] = x[perm[tid]];
}
