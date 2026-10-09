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

inline float reduce_max_256(float value, threadgroup float *scratch,
                            uint lane) {
  scratch[lane] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = k_linalg_threads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      scratch[lane] = max(scratch[lane], scratch[lane + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float result = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

inline float reduce_sum_256(float value, threadgroup float *scratch,
                            uint lane) {
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

inline bool initialize_krylov_basis(device const float *start,
                                    device float *basis, int n, int stride,
                                    threadgroup float *scratch, uint lane) {
  float local_scale = 0.0f;
  for (int row = static_cast<int>(lane); row < n;
       row += static_cast<int>(k_linalg_threads)) {
    local_scale =
        max(local_scale, isfinite(start[row]) ? abs(start[row]) : INFINITY);
  }
  const float scale = reduce_max_256(local_scale, scratch, lane);
  if (scale == 0.0f || !isfinite(scale)) {
    return false;
  }
  float squared = 0.0f;
  for (int row = static_cast<int>(lane); row < n;
       row += static_cast<int>(k_linalg_threads)) {
    const float value = start[row] / scale;
    squared += value * value;
  }
  const float norm = sqrt(reduce_sum_256(squared, scratch, lane));
  for (int row = static_cast<int>(lane); row < n;
       row += static_cast<int>(k_linalg_threads)) {
    basis[static_cast<size_t>(row) * stride] = (start[row] / scale) / norm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  return true;
}

inline float krylov_vector_norm(device const float *values, int n,
                                threadgroup float *scratch, uint lane) {
  float local_scale = 0.0f;
  for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
    local_scale = max(local_scale, abs(values[row]));
  }
  const float scale = reduce_max_256(local_scale, scratch, lane);
  if (scale == 0.0f) {
    return 0.0f;
  }
  float squared = 0.0f;
  for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
    const float value = values[row] / scale;
    squared += value * value;
  }
  const float norm = scale * sqrt(reduce_sum_256(squared, scratch, lane));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return norm;
}

inline float krylov_random_component(uint row, uint seed) {
  uint bits = row ^ seed;
  bits ^= bits >> 16;
  bits *= 0x7feb352d;
  bits ^= bits >> 15;
  bits *= 0x846ca68b;
  bits ^= bits >> 16;
  return float(bits >> 8) * (2.0f / 16777216.0f) - 1.0f;
}

inline bool restart_krylov_basis(device float *basis, device float *work, int n,
                                 int stride, int used,
                                 threadgroup float *scratch, uint lane) {
  for (uint attempt = 0; attempt < 3; ++attempt) {
    const uint seed = 0x9e3779b9u * uint(used) + 0x85ebca6bu * (attempt + 1);
    for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
      work[row] = krylov_random_component(uint(row), seed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const float initial_norm = krylov_vector_norm(work, n, scratch, lane);
    for (int pass = 0; pass < 2; ++pass) {
      for (int col = 0; col < used; ++col) {
        float local = 0.0f;
        for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
          local += basis[size_t(row) * stride + col] * work[row];
        }
        const float correction = reduce_sum_256(local, scratch, lane);
        for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
          work[row] -= correction * basis[size_t(row) * stride + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      }
    }
    const float norm = krylov_vector_norm(work, n, scratch, lane);
    if (norm > 16 * 1.1920928955078125e-7f * initial_norm) {
      for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
        basis[size_t(row) * stride + used] = work[row] / norm;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      return true;
    }
  }
  return false;
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
