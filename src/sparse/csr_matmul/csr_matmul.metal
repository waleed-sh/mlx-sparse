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

template <typename T, typename I, typename O>
[[kernel]] void csr_matmul_kernel(device const T *data [[buffer(0)]],
                                  device const I *indices [[buffer(1)]],
                                  device const I *indptr [[buffer(2)]],
                                  device const T *rhs [[buffer(3)]],
                                  device T *out [[buffer(4)]],
                                  constant int &n_rows [[buffer(5)]],
                                  constant int &rhs_cols [[buffer(6)]],
                                  uint2 position [[thread_position_in_grid]],
                                  uint2 grid_size [[threads_per_grid]]) {
  const O tid = O(position.y) * grid_size.x + position.x;
  const O total = O(n_rows) * rhs_cols;
  if (tid >= total) {
    return;
  }

  const int row = int(tid / rhs_cols);
  const int rhs_col = int(tid % rhs_cols);

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    acc += sparse_multiply<T>(data[p], rhs[O(indices[p]) * rhs_cols + rhs_col]);
  }
  out[tid] = sparse_accumulator<T>::cast(acc);
}

template <typename T, typename I, typename O>
[[kernel]] void csr_matmul_vector_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *rhs [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_rows [[buffer(5)]],
    constant int &rhs_cols [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 group_count [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  const O out_id = O(group_id.y) * group_count.x + group_id.x;
  threadgroup typename sparse_accumulator<T>::type partial[32];

  if (out_id >= O(n_rows) * rhs_cols) {
    return;
  }

  const int row = int(out_id / rhs_cols);
  const int rhs_col = int(out_id % rhs_cols);
  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[row] + static_cast<I>(lane); p < indptr[row + 1];
       p += 128) {
    acc += sparse_multiply<T>(data[p], rhs[O(indices[p]) * rhs_cols + rhs_col]);
  }
  const auto sum = sparse_cooperative_sum_128(acc, partial, simd_lane,
                                              simd_group, simd_width);

  if (lane == 0) {
    out[out_id] = sparse_accumulator<T>::cast(sum);
  }
}

#define INSTANTIATE_CSR_MATMUL(NAME, T, I, O)                                  \
  template [[host_name("csr_matmul_" #NAME)]] [[kernel]] void                  \
  csr_matmul_kernel<T, I, O>(device const T *, device const I *,               \
                             device const I *, device const T *, device T *,   \
                             constant int &, constant int &, uint2, uint2);    \
  template [[host_name("csr_matmul_vector_" #NAME)]] [[kernel]] void           \
  csr_matmul_vector_kernel<T, I, O>(                                           \
      device const T *, device const I *, device const I *, device const T *,  \
      device T *, constant int &, constant int &, uint2, uint2, uint, uint,    \
      uint, uint)

INSTANTIATE_CSR_MATMUL(float32_int32, float, int, uint);
INSTANTIATE_CSR_MATMUL(float32_int32_large, float, int, size_t);
INSTANTIATE_CSR_MATMUL(float32_int64, float, long, uint);
INSTANTIATE_CSR_MATMUL(float32_int64_large, float, long, size_t);
INSTANTIATE_CSR_MATMUL(float16_int32, half, int, uint);
INSTANTIATE_CSR_MATMUL(float16_int32_large, half, int, size_t);
INSTANTIATE_CSR_MATMUL(float16_int64, half, long, uint);
INSTANTIATE_CSR_MATMUL(float16_int64_large, half, long, size_t);
INSTANTIATE_CSR_MATMUL(bfloat16_int32, bfloat16_t, int, uint);
INSTANTIATE_CSR_MATMUL(bfloat16_int32_large, bfloat16_t, int, size_t);
INSTANTIATE_CSR_MATMUL(bfloat16_int64, bfloat16_t, long, uint);
INSTANTIATE_CSR_MATMUL(bfloat16_int64_large, bfloat16_t, long, size_t);
INSTANTIATE_CSR_MATMUL(complex64_int32, complex64_t, int, uint);
INSTANTIATE_CSR_MATMUL(complex64_int32_large, complex64_t, int, size_t);
INSTANTIATE_CSR_MATMUL(complex64_int64, complex64_t, long, uint);
INSTANTIATE_CSR_MATMUL(complex64_int64_large, complex64_t, long, size_t);

#undef INSTANTIATE_CSR_MATMUL
