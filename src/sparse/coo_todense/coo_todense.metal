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

template <typename T, typename I>
[[kernel]] void coo_todense_kernel(
    device const T *data [[buffer(0)]], device const I *row [[buffer(1)]],
    device const I *col [[buffer(2)]], device T *out [[buffer(3)]],
    constant int &n_rows [[buffer(4)]], constant int &n_cols [[buffer(5)]],
    constant int &nnz [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }

  for (int i = 0; i < n_rows * n_cols; ++i) {
    out[i] = T(0);
  }
  for (int p = 0; p < nnz; ++p) {
    const int offset =
        static_cast<int>(row[p]) * n_cols + static_cast<int>(col[p]);
    out[offset] = sparse_add_storage<T>(out[offset], data[p]);
  }
}

#define INSTANTIATE_COO_TODENSE(NAME, T, I)                                    \
  template [[host_name("coo_todense_" #NAME)]] [[kernel]] void                 \
  coo_todense_kernel<T, I>(device const T *, device const I *,                 \
                           device const I *, device T *, constant int &,       \
                           constant int &, constant int &, uint)

INSTANTIATE_COO_TODENSE(float32_int32, float, int);
INSTANTIATE_COO_TODENSE(float32_int64, float, long);
INSTANTIATE_COO_TODENSE(float16_int32, half, int);
INSTANTIATE_COO_TODENSE(float16_int64, half, long);
INSTANTIATE_COO_TODENSE(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_COO_TODENSE(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_COO_TODENSE(complex64_int32, complex64_t, int);
INSTANTIATE_COO_TODENSE(complex64_int64, complex64_t, long);

#undef INSTANTIATE_COO_TODENSE
