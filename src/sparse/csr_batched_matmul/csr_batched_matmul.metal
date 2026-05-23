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
[[kernel]] void csr_batched_matmul_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *rhs [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_rows [[buffer(5)]],
    constant int &n_cols [[buffer(6)]], constant int &batch_size [[buffer(7)]],
    constant int &rhs_cols [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  const int outputs_per_batch = n_rows * rhs_cols;
  const int total = batch_size * outputs_per_batch;
  if (static_cast<int>(tid) >= total) {
    return;
  }

  const int batch = static_cast<int>(tid) / outputs_per_batch;
  const int rem = static_cast<int>(tid) - batch * outputs_per_batch;
  const int row = rem / rhs_cols;
  const int rhs_col = rem - row * rhs_cols;
  const int rhs_batch = batch * n_cols * rhs_cols;
  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    acc += sparse_multiply<T>(
        data[p],
        rhs[rhs_batch + static_cast<int>(indices[p]) * rhs_cols + rhs_col]);
  }
  out[tid] = sparse_accumulator<T>::cast(acc);
}

template <typename T, typename I>
[[kernel]] void csr_batched_matmul_vector_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *rhs [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_rows [[buffer(5)]],
    constant int &n_cols [[buffer(6)]], constant int &batch_size [[buffer(7)]],
    constant int &rhs_cols [[buffer(8)]],
    uint out_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[128];

  const int outputs_per_batch = n_rows * rhs_cols;
  const int total = batch_size * outputs_per_batch;
  if (static_cast<int>(out_id) >= total) {
    return;
  }

  const int batch = static_cast<int>(out_id) / outputs_per_batch;
  const int rem = static_cast<int>(out_id) - batch * outputs_per_batch;
  const int row = rem / rhs_cols;
  const int rhs_col = rem - row * rhs_cols;
  const int rhs_batch = batch * n_cols * rhs_cols;
  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[row] + static_cast<I>(lane); p < indptr[row + 1];
       p += 128) {
    acc += sparse_multiply<T>(
        data[p],
        rhs[rhs_batch + static_cast<int>(indices[p]) * rhs_cols + rhs_col]);
  }
  partial[lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    out[out_id] = sparse_accumulator<T>::cast(partial[0]);
  }
}

#define INSTANTIATE_CSR_BATCHED_MATMUL(NAME, T, I)                             \
  template [[host_name("csr_batched_matmul_" #NAME)]] [[kernel]] void          \
  csr_batched_matmul_kernel<T, I>(device const T *, device const I *,          \
                                  device const I *, device const T *,          \
                                  device T *, constant int &, constant int &,  \
                                  constant int &, constant int &, uint);       \
  template [[host_name("csr_batched_matmul_vector_" #NAME)]] [[kernel]] void   \
  csr_batched_matmul_vector_kernel<T, I>(                                      \
      device const T *, device const I *, device const I *, device const T *,  \
      device T *, constant int &, constant int &, constant int &,              \
      constant int &, uint, uint)

INSTANTIATE_CSR_BATCHED_MATMUL(float32_int32, float, int);
INSTANTIATE_CSR_BATCHED_MATMUL(float32_int64, float, long);
INSTANTIATE_CSR_BATCHED_MATMUL(float16_int32, half, int);
INSTANTIATE_CSR_BATCHED_MATMUL(float16_int64, half, long);
INSTANTIATE_CSR_BATCHED_MATMUL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_BATCHED_MATMUL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_BATCHED_MATMUL(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_BATCHED_MATMUL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_BATCHED_MATMUL
