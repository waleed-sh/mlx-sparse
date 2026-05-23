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

template <typename T> inline float sparse_norm_square(T value) {
  const float x = float(value);
  return x * x;
}

template <> inline float sparse_norm_square(complex64_t value) {
  return value.real * value.real + value.imag * value.imag;
}

template <typename T, typename I>
[[kernel]] void csr_row_norms_kernel(device const T *data [[buffer(0)]],
                                     device const I *indptr [[buffer(1)]],
                                     device float *out [[buffer(2)]],
                                     constant int &n_rows [[buffer(3)]],
                                     uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  float acc = 0.0f;
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    acc += sparse_norm_square<T>(data[p]);
  }
  out[row] = sqrt(acc);
}

#define INSTANTIATE_CSR_ROW_NORMS(NAME, T, I)                                  \
  template [[host_name("csr_row_norms_" #NAME)]] [[kernel]] void               \
  csr_row_norms_kernel<T, I>(device const T *, device const I *,               \
                             device float *, constant int &, uint)

INSTANTIATE_CSR_ROW_NORMS(float32_int32, float, int);
INSTANTIATE_CSR_ROW_NORMS(float32_int64, float, long);
INSTANTIATE_CSR_ROW_NORMS(float16_int32, half, int);
INSTANTIATE_CSR_ROW_NORMS(float16_int64, half, long);
INSTANTIATE_CSR_ROW_NORMS(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_ROW_NORMS(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_ROW_NORMS(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_ROW_NORMS(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_ROW_NORMS

template <typename T, typename I>
[[kernel]] void csr_row_norms_vector_kernel(
    device const T *data [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device float *out [[buffer(2)]], constant int &n_rows [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float partial[128];

  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  float acc = 0.0f;
  const I start = indptr[row];
  const I end = indptr[row + 1];
  for (I p = start + static_cast<I>(lane); p < end; p += 128) {
    acc += sparse_norm_square<T>(data[p]);
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
    out[row] = sqrt(partial[0]);
  }
}

#define INSTANTIATE_CSR_ROW_NORMS_VECTOR(NAME, T, I)                           \
  template [[host_name("csr_row_norms_vector_" #NAME)]] [[kernel]] void        \
  csr_row_norms_vector_kernel<T, I>(device const T *, device const I *,        \
                                    device float *, constant int &, uint,      \
                                    uint)

INSTANTIATE_CSR_ROW_NORMS_VECTOR(float32_int32, float, int);
INSTANTIATE_CSR_ROW_NORMS_VECTOR(float32_int64, float, long);
INSTANTIATE_CSR_ROW_NORMS_VECTOR(float16_int32, half, int);
INSTANTIATE_CSR_ROW_NORMS_VECTOR(float16_int64, half, long);
INSTANTIATE_CSR_ROW_NORMS_VECTOR(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_ROW_NORMS_VECTOR(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_ROW_NORMS_VECTOR(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_ROW_NORMS_VECTOR(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_ROW_NORMS_VECTOR
