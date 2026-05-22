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

#pragma once

#include "common/metal_common.h"

constant uint k_linalg_threads = 256;

inline float reduce_sum_256(float value, threadgroup float *scratch, uint lane) {
  scratch[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = k_linalg_threads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return scratch[0];
}

template <typename T>
inline T reduce_sum_256_any(T value, threadgroup T *scratch, uint lane) {
  scratch[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = k_linalg_threads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return scratch[0];
}

template <typename I>
inline void csr_spmv_f32(device const float *data, device const I *indices,
                         device const I *indptr, device const float *x,
                         device float *out, int n_rows, uint lane) {
  for (int row = static_cast<int>(lane); row < n_rows;
       row += static_cast<int>(k_linalg_threads)) {
    float acc = 0.0f;
    const I start = indptr[row];
    const I end = indptr[row + 1];
    for (I p = start; p < end; ++p) {
      acc += data[p] * x[indices[p]];
    }
    out[row] = acc;
  }
}

inline float vector_dot_f32(device const float *lhs, device const float *rhs,
                            int n, threadgroup float *scratch, uint lane) {
  float acc = 0.0f;
  for (int i = static_cast<int>(lane); i < n;
       i += static_cast<int>(k_linalg_threads)) {
    acc += lhs[i] * rhs[i];
  }
  return reduce_sum_256(acc, scratch, lane);
}

inline complex64_t sparse_conjugate(complex64_t value) {
  return complex64_t(value.real, -value.imag);
}

inline float sparse_conjugate(float value) { return value; }
