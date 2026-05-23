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
[[kernel]] void csr_row_sums_kernel(device const T *data [[buffer(0)]],
                                    device const I *indptr [[buffer(1)]],
                                    device T *out [[buffer(2)]],
                                    constant int &n_rows [[buffer(3)]],
                                    uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    acc += typename sparse_accumulator<T>::type(data[p]);
  }
  out[row] = sparse_accumulator<T>::cast(acc);
}

#define INSTANTIATE_CSR_ROW_SUMS(NAME, T, I)                                   \
  template [[host_name("csr_row_sums_" #NAME)]] [[kernel]] void                \
  csr_row_sums_kernel<T, I>(device const T *, device const I *, device T *,    \
                            constant int &, uint)

INSTANTIATE_CSR_ROW_SUMS(float32_int32, float, int);
INSTANTIATE_CSR_ROW_SUMS(float32_int64, float, long);
INSTANTIATE_CSR_ROW_SUMS(float16_int32, half, int);
INSTANTIATE_CSR_ROW_SUMS(float16_int64, half, long);
INSTANTIATE_CSR_ROW_SUMS(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_ROW_SUMS(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_ROW_SUMS(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_ROW_SUMS(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_ROW_SUMS

template <typename T, typename I>
[[kernel]] void csr_row_sums_vector_kernel(
    device const T *data [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device T *out [[buffer(2)]], constant int &n_rows [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[128];

  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  const I start = indptr[row];
  const I end = indptr[row + 1];
  for (I p = start + static_cast<I>(lane); p < end; p += 128) {
    acc += typename sparse_accumulator<T>::type(data[p]);
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
    out[row] = sparse_accumulator<T>::cast(partial[0]);
  }
}

#define INSTANTIATE_CSR_ROW_SUMS_VECTOR(NAME, T, I)                            \
  template [[host_name("csr_row_sums_vector_" #NAME)]] [[kernel]] void         \
  csr_row_sums_vector_kernel<T, I>(device const T *, device const I *,         \
                                   device T *, constant int &, uint, uint)

INSTANTIATE_CSR_ROW_SUMS_VECTOR(float32_int32, float, int);
INSTANTIATE_CSR_ROW_SUMS_VECTOR(float32_int64, float, long);
INSTANTIATE_CSR_ROW_SUMS_VECTOR(float16_int32, half, int);
INSTANTIATE_CSR_ROW_SUMS_VECTOR(float16_int64, half, long);
INSTANTIATE_CSR_ROW_SUMS_VECTOR(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_ROW_SUMS_VECTOR(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_ROW_SUMS_VECTOR(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_ROW_SUMS_VECTOR(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_ROW_SUMS_VECTOR
