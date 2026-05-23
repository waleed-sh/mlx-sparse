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
[[kernel]] void csr_arnoldi_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *v0 [[buffer(3)]], device float *h [[buffer(4)]],
    device float *basis [[buffer(5)]], device int *actual [[buffer(6)]],
    device float *work [[buffer(7)]], constant int &n_rows [[buffer(8)]],
    constant int &n_cols [[buffer(9)]], constant int &k [[buffer(10)]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)n_cols;
  const int cols = k + 1;
  threadgroup float scratch[256];
  threadgroup float shared_scalar;
  threadgroup int shared_done;
  threadgroup int shared_used;

  for (int i = static_cast<int>(lane); i < cols * k;
       i += static_cast<int>(k_linalg_threads)) {
    h[i] = 0.0f;
  }
  for (int i = static_cast<int>(lane); i < n_rows * cols;
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
    basis[row * cols] = norm0 <= 1.1920928955078125e-7f
                            ? (row == 0 ? 1.0f : 0.0f)
                            : v0[row] / norm0;
  }
  if (lane == 0) {
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
        acc += data[p] * basis[indices[p] * cols + j];
      }
      work[row] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int pass = 0; pass < 2; ++pass) {
      for (int col = 0; col <= j; ++col) {
        float coeff_local = 0.0f;
        for (int row = static_cast<int>(lane); row < n_rows;
             row += static_cast<int>(k_linalg_threads)) {
          coeff_local += basis[row * cols + col] * work[row];
        }
        const float coeff = reduce_sum_256(coeff_local, scratch, lane);
        if (lane == 0) {
          h[col * k + j] += coeff;
        }
        for (int row = static_cast<int>(lane); row < n_rows;
             row += static_cast<int>(k_linalg_threads)) {
          work[row] -= coeff * basis[row * cols + col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    float h_next_local = 0.0f;
    for (int row = static_cast<int>(lane); row < n_rows;
         row += static_cast<int>(k_linalg_threads)) {
      h_next_local += work[row] * work[row];
    }
    const float h_next =
        sqrt(max(reduce_sum_256(h_next_local, scratch, lane), 0.0f));
    if (lane == 0) {
      h[(j + 1) * k + j] = h_next;
      shared_scalar = h_next;
      shared_used = j + 1;
      if (h_next <= 1.1920928955078125e-7f) {
        shared_done = 1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_done == 0) {
      for (int row = static_cast<int>(lane); row < n_rows;
           row += static_cast<int>(k_linalg_threads)) {
        basis[row * cols + j + 1] = work[row] / shared_scalar;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (lane == 0) {
    actual[0] = shared_used;
  }
}

template [[host_name("csr_arnoldi_float32_int32")]] [[kernel]] void
csr_arnoldi_kernel<int>(device const float *, device const int *,
                        device const int *, device const float *,
                        device float *, device float *, device int *,
                        device float *, constant int &, constant int &,
                        constant int &, uint);

template [[host_name("csr_arnoldi_float32_int64")]] [[kernel]] void
csr_arnoldi_kernel<long>(device const float *, device const long *,
                         device const long *, device const float *,
                         device float *, device float *, device int *,
                         device float *, constant int &, constant int &,
                         constant int &, uint);
