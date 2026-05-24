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
[[kernel]] void csc_sort_indices_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device T *out_data [[buffer(3)]],
    device I *out_indices [[buffer(4)]], device I *out_indptr [[buffer(5)]],
    constant int &nnz [[buffer(6)]], constant int &n_indptr [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid < static_cast<uint>(n_indptr)) {
    out_indptr[tid] = indptr[tid];
  }
  if (tid >= static_cast<uint>(nnz)) {
    return;
  }

  int col = 0;
  for (int c = 0; c + 1 < n_indptr; ++c) {
    if (static_cast<I>(tid) >= indptr[c] &&
        static_cast<I>(tid) < indptr[c + 1]) {
      col = c;
      break;
    }
  }

  const I start = indptr[col];
  const I end = indptr[col + 1];
  int rank = 0;
  for (I p = start; p < end; ++p) {
    if (indices[p] < indices[tid] ||
        (indices[p] == indices[tid] && p < static_cast<I>(tid))) {
      rank += 1;
    }
  }

  const I dst = start + static_cast<I>(rank);
  out_data[dst] = data[tid];
  out_indices[dst] = indices[tid];
}

#define INSTANTIATE_CSC_SORT(NAME, T, I)                                       \
  template [[host_name("csc_sort_indices_" #NAME)]] [[kernel]] void            \
  csc_sort_indices_kernel<T, I>(                                               \
      device const T *, device const I *, device const I *, device T *,        \
      device I *, device I *, constant int &, constant int &, uint)

INSTANTIATE_CSC_SORT(float32_int32, float, int);
INSTANTIATE_CSC_SORT(float32_int64, float, long);
INSTANTIATE_CSC_SORT(float16_int32, half, int);
INSTANTIATE_CSC_SORT(float16_int64, half, long);
INSTANTIATE_CSC_SORT(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_SORT(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_SORT(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_SORT(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_SORT
