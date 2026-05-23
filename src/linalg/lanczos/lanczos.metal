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
[[kernel]] void csr_lanczos_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *v0 [[buffer(3)]], device float *alphas [[buffer(4)]],
    device float *betas [[buffer(5)]], device float *basis [[buffer(6)]],
    device int *actual [[buffer(7)]], device float *work [[buffer(8)]],
    constant int &n_rows [[buffer(9)]], constant int &n_cols [[buffer(10)]],
    constant int &k [[buffer(11)]],
    constant int &reorthogonalize [[buffer(12)]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)n_cols;
  threadgroup float scratch[256];
  threadgroup float shared_scalar;
  threadgroup float shared_beta;
  threadgroup float beta_prev;
  threadgroup int shared_done;
  threadgroup int shared_used;

  for (int i = static_cast<int>(lane); i < k;
       i += static_cast<int>(k_linalg_threads)) {
    alphas[i] = 0.0f;
    betas[i] = 0.0f;
  }
  for (int i = static_cast<int>(lane); i < n_rows * k;
       i += static_cast<int>(k_linalg_threads)) {
    basis[i] = 0.0f;
  }

  float norm_local = 0.0f;
  for (int row = static_cast<int>(lane); row < n_rows;
       row += static_cast<int>(k_linalg_threads)) {
    norm_local += v0[row] * v0[row];
  }
  const float norm0 =
      sqrt(max(reduce_sum_256(norm_local, scratch, lane), 0.0f));
  for (int row = static_cast<int>(lane); row < n_rows;
       row += static_cast<int>(k_linalg_threads)) {
    basis[row * k] = norm0 <= 1.1920928955078125e-7f ? (row == 0 ? 1.0f : 0.0f)
                                                     : v0[row] / norm0;
  }
  if (lane == 0) {
    beta_prev = 0.0f;
    shared_done = 0;
    shared_used = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int j = 0; j < k; ++j) {
    if (shared_done != 0) {
      break;
    }

    for (int row = static_cast<int>(lane); row < n_rows;
         row += static_cast<int>(k_linalg_threads)) {
      float acc = 0.0f;
      for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
        acc += data[p] * basis[indices[p] * k + j];
      }
      work[row] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (j > 0) {
      for (int row = static_cast<int>(lane); row < n_rows;
           row += static_cast<int>(k_linalg_threads)) {
        work[row] -= beta_prev * basis[row * k + j - 1];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float alpha_local = 0.0f;
    for (int row = static_cast<int>(lane); row < n_rows;
         row += static_cast<int>(k_linalg_threads)) {
      alpha_local += basis[row * k + j] * work[row];
    }
    const float alpha = reduce_sum_256(alpha_local, scratch, lane);
    if (lane == 0) {
      alphas[j] = alpha;
    }
    for (int row = static_cast<int>(lane); row < n_rows;
         row += static_cast<int>(k_linalg_threads)) {
      work[row] -= alpha * basis[row * k + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (reorthogonalize != 0) {
      for (int pass = 0; pass < 2; ++pass) {
        for (int col = 0; col <= j; ++col) {
          float corr_local = 0.0f;
          for (int row = static_cast<int>(lane); row < n_rows;
               row += static_cast<int>(k_linalg_threads)) {
            corr_local += basis[row * k + col] * work[row];
          }
          const float corr = reduce_sum_256(corr_local, scratch, lane);
          for (int row = static_cast<int>(lane); row < n_rows;
               row += static_cast<int>(k_linalg_threads)) {
            work[row] -= corr * basis[row * k + col];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }

    float beta_local = 0.0f;
    for (int row = static_cast<int>(lane); row < n_rows;
         row += static_cast<int>(k_linalg_threads)) {
      beta_local += work[row] * work[row];
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
      for (int row = static_cast<int>(lane); row < n_rows;
           row += static_cast<int>(k_linalg_threads)) {
        basis[row * k + j + 1] = work[row] / shared_beta;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (lane == 0) {
    actual[0] = shared_used;
  }
}

template [[host_name("csr_lanczos_float32_int32")]] [[kernel]] void
csr_lanczos_kernel<int>(device const float *, device const int *,
                        device const int *, device const float *,
                        device float *, device float *, device float *,
                        device int *, device float *, constant int &,
                        constant int &, constant int &, constant int &, uint);

template [[host_name("csr_lanczos_float32_int64")]] [[kernel]] void
csr_lanczos_kernel<long>(device const float *, device const long *,
                         device const long *, device const float *,
                         device float *, device float *, device float *,
                         device int *, device float *, constant int &,
                         constant int &, constant int &, constant int &, uint);
