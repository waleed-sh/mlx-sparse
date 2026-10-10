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

#include "linalg/common/metal_common.h"

[[kernel]] void csr_bidiagonal_zero(device float *out [[buffer(0)]],
                                    constant int &n [[buffer(1)]],
                                    uint row [[thread_position_in_grid]]) {
  if (row < uint(n)) {
    out[row] = 0.0f;
  }
}

template <typename I>
[[kernel]] void csr_bidiagonal_adjoint_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *u [[buffer(3)]], device atomic_float *out [[buffer(4)]],
    constant int &m [[buffer(5)]], uint row [[thread_position_in_grid]]) {
  if (row < uint(m)) {
    const float value = u[row];
    for (I t = indptr[row]; t < indptr[row + 1]; ++t) {
      atomic_fetch_add_explicit(&out[indices[t]], data[t] * value,
                                memory_order_relaxed);
    }
  }
}

template [[host_name(
    "csr_bidiagonal_adjoint_float32_"
    "int32")]] [[kernel]] decltype(csr_bidiagonal_adjoint_kernel<int32_t>)
    csr_bidiagonal_adjoint_kernel<int32_t>;
template [[host_name(
    "csr_bidiagonal_adjoint_float32_"
    "int64")]] [[kernel]] decltype(csr_bidiagonal_adjoint_kernel<int64_t>)
    csr_bidiagonal_adjoint_kernel<int64_t>;

// Every SIMD group computes the final scalar so all lanes can use it.
// The second barrier allows immediate reuse of the shared workspace.
inline float bidiagonal_sum(float value, threadgroup float *partial, uint lane,
                            uint simd_width) {
  const uint simd_lane = lane % simd_width;
  const float group_sum = simd_sum(value);
  if (simd_lane == 0) {
    partial[lane / simd_width] = group_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint group = simd_lane; group < k_linalg_threads / simd_width;
       group += simd_width) {
    total += partial[group];
  }
  total = simd_sum(total);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

inline float bidiagonal_norm(device const float *work, int n,
                             threadgroup float *partial, uint lane,
                             uint simd_width) {
  const uint simd_lane = lane % simd_width;
  float scale = 0.0f;
  for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
    scale = max(scale, abs(work[row]));
  }
  scale = simd_max(scale);
  if (simd_lane == 0) {
    partial[lane / simd_width] = scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  scale = 0.0f;
  for (uint group = simd_lane; group < k_linalg_threads / simd_width;
       group += simd_width) {
    scale = max(scale, partial[group]);
  }
  scale = simd_max(scale);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (scale == 0.0f) {
    return 0.0f;
  }
  float squared = 0.0f;
  for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
    const float value = work[row] / scale;
    squared += value * value;
  }
  return scale * sqrt(bidiagonal_sum(squared, partial, lane, simd_width));
}

inline bool append_bidiagonal_direction(device float *basis, device float *work,
                                        int n, int used,
                                        threadgroup float *scratch, uint lane,
                                        uint simd_width) {
  for (uint attempt = 0; attempt < 4; ++attempt) {
    if (attempt != 0) {
      const uint seed = 0x9e3779b9u * uint(used) + 0x85ebca6bu * attempt;
      for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
        work[row] = krylov_random_component(uint(row), seed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
    const float initial = bidiagonal_norm(work, n, scratch, lane, simd_width);
    for (int pass = 0; pass < 2; ++pass) {
      for (int col = 0; col < used; ++col) {
        float local = 0.0f;
        for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
          local += basis[size_t(col) * n + row] * work[row];
        }
        const float coefficient =
            bidiagonal_sum(local, scratch, lane, simd_width);
        for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
          work[row] -= coefficient * basis[size_t(col) * n + row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      }
    }
    const float residual = bidiagonal_norm(work, n, scratch, lane, simd_width);
    const float threshold =
        (attempt == 0 ? 8 : 16) * 1.1920928955078125e-7f * initial;
    if (residual > threshold) {
      for (int row = int(lane); row < n; row += int(k_linalg_threads)) {
        basis[size_t(used) * n + row] = work[row] / residual;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      return true;
    }
  }
  return false;
}

template <typename I>
[[kernel]] void csr_bidiagonal_basis_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *v0 [[buffer(3)]], device float *left [[buffer(4)]],
    device float *right [[buffer(5)]], device float *images [[buffer(6)]],
    device int *actual [[buffer(7)]], device float *left_work [[buffer(8)]],
    device float *right_work [[buffer(9)]], constant int &m [[buffer(10)]],
    constant int &n [[buffer(11)]], constant int &p [[buffer(12)]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  threadgroup float scratch[256];
  const int q = min(n, p + 1);
  for (size_t i = lane; i < size_t(m) * p; i += k_linalg_threads) {
    left[i] = 0.0f;
  }
  for (size_t i = lane; i < size_t(n) * q; i += k_linalg_threads) {
    right[i] = 0.0f;
  }
  if (lane == 0) {
    actual[0] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (!initialize_krylov_basis(v0, right, n, 1, scratch, lane)) {
    return;
  }
  for (int j = 0; j < p; ++j) {
    for (int row = int(lane); row < m; row += int(k_linalg_threads)) {
      float acc = 0.0f;
      for (I t = indptr[row]; t < indptr[row + 1]; ++t) {
        acc += data[t] * right[size_t(j) * n + indices[t]];
      }
      left_work[row] = acc;
      images[size_t(j) * m + row] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (!append_bidiagonal_direction(left, left_work, m, j, scratch, lane,
                                     simd_width)) {
      if (lane == 0) {
        actual[0] = -(j + 1);
      }
      return;
    }
    if (lane == 0) {
      actual[0] = j + 1;
    }
    if (j + 1 == q) {
      break;
    }
    for (int col = int(lane); col < n; col += int(k_linalg_threads)) {
      right_work[col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    device atomic_float *atomic_work =
        reinterpret_cast<device atomic_float *>(right_work);
    for (int row = int(lane); row < m; row += int(k_linalg_threads)) {
      const float value = left[size_t(j) * m + row];
      for (I t = indptr[row]; t < indptr[row + 1]; ++t) {
        atomic_fetch_add_explicit(&atomic_work[indices[t]], data[t] * value,
                                  memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (!append_bidiagonal_direction(right, right_work, n, j + 1, scratch, lane,
                                     simd_width)) {
      if (lane == 0) {
        actual[0] = -(j + 1);
      }
      return;
    }
  }
  if (q > p) {
    for (int row = int(lane); row < m; row += int(k_linalg_threads)) {
      float acc = 0.0f;
      for (I t = indptr[row]; t < indptr[row + 1]; ++t) {
        acc += data[t] * right[size_t(p) * n + indices[t]];
      }
      images[size_t(p) * m + row] = acc;
    }
  }
}

template [[host_name(
    "csr_bidiagonal_basis_float32_"
    "int32")]] [[kernel]] decltype(csr_bidiagonal_basis_kernel<int32_t>)
    csr_bidiagonal_basis_kernel<int32_t>;
template [[host_name(
    "csr_bidiagonal_basis_float32_"
    "int64")]] [[kernel]] decltype(csr_bidiagonal_basis_kernel<int64_t>)
    csr_bidiagonal_basis_kernel<int64_t>;

template <typename I>
[[kernel]] void csr_normal_lanczos_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *v0 [[buffer(3)]], device float *alphas [[buffer(4)]],
    device float *betas [[buffer(5)]], device float *basis [[buffer(6)]],
    device int *actual [[buffer(7)]], device float *work [[buffer(8)]],
    constant int &n_rows [[buffer(9)]], constant int &n_cols [[buffer(10)]],
    constant int &k [[buffer(11)]], constant int &complete_basis [[buffer(12)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[256];
  threadgroup float shared_alpha;
  threadgroup float beta_prev;
  threadgroup int shared_used;

  for (int i = static_cast<int>(lane); i < k;
       i += static_cast<int>(k_linalg_threads)) {
    alphas[i] = 0.0f;
    betas[i] = 0.0f;
  }
  for (int i = static_cast<int>(lane); i < n_cols * k;
       i += static_cast<int>(k_linalg_threads)) {
    basis[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  if (!initialize_krylov_basis(v0, basis, n_cols, k, scratch, lane)) {
    if (lane == 0) {
      actual[0] = 0;
    }
    return;
  }
  if (lane == 0) {
    actual[0] = 0;
    beta_prev = 0.0f;
    shared_used = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  device atomic_float *atomic_work =
      reinterpret_cast<device atomic_float *>(work);

  for (int j = 0; j < k; ++j) {
    for (int col = static_cast<int>(lane); col < n_cols;
         col += static_cast<int>(k_linalg_threads)) {
      work[col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (int row = static_cast<int>(lane); row < n_rows;
         row += static_cast<int>(k_linalg_threads)) {
      float ax = 0.0f;
      const I start = indptr[row];
      const I end = indptr[row + 1];
      for (I p = start; p < end; ++p) {
        const int col = static_cast<int>(indices[p]);
        if (col >= 0 && col < n_cols) {
          ax += data[p] * basis[col * k + j];
        }
      }
      if (ax != 0.0f) {
        for (I p = start; p < end; ++p) {
          const int col = static_cast<int>(indices[p]);
          if (col >= 0 && col < n_cols) {
            atomic_fetch_add_explicit(&atomic_work[col], data[p] * ax,
                                      memory_order_relaxed);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const float operator_norm = krylov_vector_norm(work, n_cols, scratch, lane);

    if (j > 0) {
      for (int col = static_cast<int>(lane); col < n_cols;
           col += static_cast<int>(k_linalg_threads)) {
        work[col] -= beta_prev * basis[col * k + j - 1];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    float alpha_local = 0.0f;
    for (int col = static_cast<int>(lane); col < n_cols;
         col += static_cast<int>(k_linalg_threads)) {
      alpha_local += basis[col * k + j] * work[col];
    }
    const float alpha = reduce_sum_256(alpha_local, scratch, lane);
    if (lane == 0) {
      alphas[j] = alpha;
      shared_alpha = alpha;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (int col = static_cast<int>(lane); col < n_cols;
         col += static_cast<int>(k_linalg_threads)) {
      work[col] -= shared_alpha * basis[col * k + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (int pass = 0; pass < 2; ++pass) {
      for (int orth_col = 0; orth_col <= j; ++orth_col) {
        float corr_local = 0.0f;
        for (int col = static_cast<int>(lane); col < n_cols;
             col += static_cast<int>(k_linalg_threads)) {
          corr_local += basis[col * k + orth_col] * work[col];
        }
        const float correction = reduce_sum_256(corr_local, scratch, lane);
        for (int col = static_cast<int>(lane); col < n_cols;
             col += static_cast<int>(k_linalg_threads)) {
          work[col] -= correction * basis[col * k + orth_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      }
    }

    const float beta = krylov_vector_norm(work, n_cols, scratch, lane);
    if (lane == 0) {
      betas[j] = beta;
      shared_used = j + 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (j + 1 == k) {
      break;
    }
    if (beta <= 8 * 1.1920928955078125e-7f * operator_norm) {
      if (lane == 0) {
        betas[j] = 0.0f;
        beta_prev = 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      if (complete_basis == 0) {
        break;
      }
      if (!restart_krylov_basis(basis, work, n_cols, k, j + 1, scratch, lane)) {
        if (lane == 0) {
          shared_used = -shared_used;
        }
        break;
      }
      continue;
    }
    if (lane == 0) {
      beta_prev = beta;
    }
    for (int row = int(lane); row < n_cols; row += int(k_linalg_threads)) {
      basis[size_t(row) * k + j + 1] = work[row] / beta;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  }

  if (lane == 0) {
    actual[0] = shared_used;
  }
}

template [[host_name("csr_normal_lanczos_float32_int32")]] [[kernel]] void
csr_normal_lanczos_kernel<int>(device const float *, device const int *,
                               device const int *, device const float *,
                               device float *, device float *, device float *,
                               device int *, device float *, constant int &,
                               constant int &, constant int &, constant int &,
                               uint);

template [[host_name("csr_normal_lanczos_float32_int64")]] [[kernel]] void
csr_normal_lanczos_kernel<long>(device const float *, device const long *,
                                device const long *, device const float *,
                                device float *, device float *, device float *,
                                device int *, device float *, constant int &,
                                constant int &, constant int &, constant int &,
                                uint);
