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

template <typename I>
[[kernel]] void csr_normal_lanczos_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device float *alphas [[buffer(3)]], device float *betas [[buffer(4)]],
    device float *basis [[buffer(5)]], device int *actual [[buffer(6)]],
    device float *work [[buffer(7)]], constant int &n_rows [[buffer(8)]],
    constant int &n_cols [[buffer(9)]], constant int &k [[buffer(10)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[256];
  threadgroup float shared_alpha;
  threadgroup float shared_beta;
  threadgroup float beta_prev;
  threadgroup int shared_done;
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
  const float inv_norm = rsqrt(static_cast<float>(n_cols));
  for (int col = static_cast<int>(lane); col < n_cols;
       col += static_cast<int>(k_linalg_threads)) {
    basis[col * k] = inv_norm;
    work[col] = 0.0f;
  }
  if (lane == 0) {
    actual[0] = 0;
    beta_prev = 0.0f;
    shared_done = 0;
    shared_used = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device atomic_float *atomic_work =
      reinterpret_cast<device atomic_float *>(work);

  for (int j = 0; j < k; ++j) {
    if (shared_done != 0) {
      break;
    }

    for (int col = static_cast<int>(lane); col < n_cols;
         col += static_cast<int>(k_linalg_threads)) {
      work[col] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);

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
    threadgroup_barrier(mem_flags::mem_device);

    if (j > 0) {
      for (int col = static_cast<int>(lane); col < n_cols;
           col += static_cast<int>(k_linalg_threads)) {
        work[col] -= beta_prev * basis[col * k + j - 1];
      }
      threadgroup_barrier(mem_flags::mem_device);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int col = static_cast<int>(lane); col < n_cols;
         col += static_cast<int>(k_linalg_threads)) {
      work[col] -= shared_alpha * basis[col * k + j];
    }
    threadgroup_barrier(mem_flags::mem_device);

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
        threadgroup_barrier(mem_flags::mem_device);
      }
    }

    float beta_local = 0.0f;
    for (int col = static_cast<int>(lane); col < n_cols;
         col += static_cast<int>(k_linalg_threads)) {
      beta_local += work[col] * work[col];
    }
    const float beta =
        sqrt(max(reduce_sum_256(beta_local, scratch, lane), 0.0f));
    if (lane == 0) {
      betas[j] = beta;
      shared_beta = beta;
      shared_used = j + 1;
      if (j + 1 == k || beta <= 1.1920928955078125e-7f) {
        shared_done = 1;
      } else {
        beta_prev = beta;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (shared_done == 0) {
      for (int col = static_cast<int>(lane); col < n_cols;
           col += static_cast<int>(k_linalg_threads)) {
        basis[col * k + j + 1] = work[col] / shared_beta;
      }
      threadgroup_barrier(mem_flags::mem_device);
    }
  }

  if (lane == 0) {
    actual[0] = shared_used;
  }
}

template [[host_name("csr_normal_lanczos_float32_int32")]] [[kernel]] void
csr_normal_lanczos_kernel<int>(device const float *, device const int *,
                               device const int *, device float *,
                               device float *, device float *, device int *,
                               device float *, constant int &, constant int &,
                               constant int &, uint);

template [[host_name("csr_normal_lanczos_float32_int64")]] [[kernel]] void
csr_normal_lanczos_kernel<long>(device const float *, device const long *,
                                device const long *, device float *,
                                device float *, device float *, device int *,
                                device float *, constant int &, constant int &,
                                constant int &, uint);
