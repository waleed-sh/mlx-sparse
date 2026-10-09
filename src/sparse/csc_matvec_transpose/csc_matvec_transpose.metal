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
[[kernel]] void csc_matvec_transpose_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *x [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_cols [[buffer(5)]],
    uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= n_cols) {
    return;
  }

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  const I start = indptr[col];
  const I end = indptr[col + 1];
  for (I p = start; p < end; ++p) {
    acc += sparse_multiply<T>(data[p], x[indices[p]]);
  }
  out[col] = sparse_accumulator<T>::cast(acc);
}

#define INSTANTIATE_CSC_MATVEC_T(NAME, T, I)                                   \
  template [[host_name("csc_matvec_transpose_" #NAME)]] [[kernel]] void        \
  csc_matvec_transpose_kernel<T, I>(device const T *, device const I *,        \
                                    device const I *, device const T *,        \
                                    device T *, constant int &, uint)

INSTANTIATE_CSC_MATVEC_T(float32_int32, float, int);
INSTANTIATE_CSC_MATVEC_T(float32_int64, float, long);
INSTANTIATE_CSC_MATVEC_T(float16_int32, half, int);
INSTANTIATE_CSC_MATVEC_T(float16_int64, half, long);
INSTANTIATE_CSC_MATVEC_T(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_MATVEC_T(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_MATVEC_T(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_MATVEC_T(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_MATVEC_T

template <typename T, typename I>
[[kernel]] void csc_matvec_transpose_vector_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *x [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_cols [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 group_count [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  const uint col = group_id.y * group_count.x + group_id.x;
  threadgroup typename sparse_accumulator<T>::type partial[32];

  if (col >= uint(n_cols)) {
    return;
  }

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  const I start = indptr[col];
  const I end = indptr[col + 1];
  for (I p = start + static_cast<I>(lane); p < end; p += 128) {
    acc += sparse_multiply<T>(data[p], x[indices[p]]);
  }
  const auto sum = sparse_cooperative_sum_128(acc, partial, simd_lane,
                                              simd_group, simd_width);

  if (lane == 0) {
    out[col] = sparse_accumulator<T>::cast(sum);
  }
}

#define INSTANTIATE_CSC_MATVEC_T_VECTOR(NAME, T, I)                            \
  template [[host_name("csc_matvec_transpose_vector_" #NAME)]] [[kernel]]      \
  void csc_matvec_transpose_vector_kernel<T, I>(                               \
      device const T *, device const I *, device const I *, device const T *,  \
      device T *, constant int &, uint2, uint2, uint, uint, uint, uint)

INSTANTIATE_CSC_MATVEC_T_VECTOR(float32_int32, float, int);
INSTANTIATE_CSC_MATVEC_T_VECTOR(float32_int64, float, long);
INSTANTIATE_CSC_MATVEC_T_VECTOR(float16_int32, half, int);
INSTANTIATE_CSC_MATVEC_T_VECTOR(float16_int64, half, long);
INSTANTIATE_CSC_MATVEC_T_VECTOR(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_MATVEC_T_VECTOR(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_MATVEC_T_VECTOR(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_MATVEC_T_VECTOR(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_MATVEC_T_VECTOR
