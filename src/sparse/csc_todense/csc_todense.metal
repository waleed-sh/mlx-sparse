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

template <typename T>
[[kernel]] void csc_todense_zero_kernel(device T *out [[buffer(0)]],
                                        constant int &size [[buffer(1)]],
                                        uint tid [[thread_position_in_grid]]) {
  if (static_cast<int>(tid) < size) {
    out[tid] = T(0);
  }
}

template <typename T, typename I>
[[kernel]] void csc_todense_fill_kernel(device const T *data [[buffer(0)]],
                                        device const I *indices [[buffer(1)]],
                                        device const I *indptr [[buffer(2)]],
                                        device T *out [[buffer(3)]],
                                        constant int &n_rows [[buffer(4)]],
                                        constant int &n_cols [[buffer(5)]],
                                        uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= n_cols) {
    return;
  }

  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    const int row = static_cast<int>(indices[p]);
    if (row >= 0 && row < n_rows) {
      const int offset = row * n_cols + static_cast<int>(col);
      out[offset] = sparse_add_storage<T>(out[offset], data[p]);
    }
  }
}

#define INSTANTIATE_CSC_TODENSE_ZERO(NAME, T)                                  \
  template [[host_name("csc_todense_zero_" #NAME)]] [[kernel]] void            \
  csc_todense_zero_kernel<T>(device T *, constant int &, uint)

INSTANTIATE_CSC_TODENSE_ZERO(float32, float);
INSTANTIATE_CSC_TODENSE_ZERO(float16, half);
INSTANTIATE_CSC_TODENSE_ZERO(bfloat16, bfloat16_t);
INSTANTIATE_CSC_TODENSE_ZERO(complex64, complex64_t);

#undef INSTANTIATE_CSC_TODENSE_ZERO

#define INSTANTIATE_CSC_TODENSE_FILL(NAME, T, I)                               \
  template [[host_name("csc_todense_fill_" #NAME)]] [[kernel]] void            \
  csc_todense_fill_kernel<T, I>(device const T *, device const I *,            \
                                device const I *, device T *, constant int &,  \
                                constant int &, uint)

INSTANTIATE_CSC_TODENSE_FILL(float32_int32, float, int);
INSTANTIATE_CSC_TODENSE_FILL(float32_int64, float, long);
INSTANTIATE_CSC_TODENSE_FILL(float16_int32, half, int);
INSTANTIATE_CSC_TODENSE_FILL(float16_int64, half, long);
INSTANTIATE_CSC_TODENSE_FILL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_TODENSE_FILL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_TODENSE_FILL(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_TODENSE_FILL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_TODENSE_FILL
