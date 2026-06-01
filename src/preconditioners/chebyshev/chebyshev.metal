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
inline void csr_spmm_f32(device const float *data, device const I *indices,
                         device const I *indptr, device const float *rhs,
                         device float *out, int n_rows, int rhs_cols,
                         uint lane) {
  const int total = n_rows * rhs_cols;
  for (int tid = static_cast<int>(lane); tid < total;
       tid += static_cast<int>(k_linalg_threads)) {
    const int row = tid / rhs_cols;
    const int rhs_col = tid - row * rhs_cols;
    float acc = 0.0f;
    for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
      acc += data[p] * rhs[indices[p] * rhs_cols + rhs_col];
    }
    out[tid] = acc;
  }
}

template <typename I>
[[kernel]] void csr_chebyshev_apply_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *rhs [[buffer(3)]], device float *out [[buffer(4)]],
    device float *work [[buffer(5)]], constant int &n_rows [[buffer(6)]],
    constant int &n_cols [[buffer(7)]], constant int &rhs_cols [[buffer(8)]],
    constant int &degree [[buffer(9)]],
    constant float &lambda_min [[buffer(10)]],
    constant float &lambda_max [[buffer(11)]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)n_cols;
  const int total = n_rows * rhs_cols;
  device float *x_prev = work;
  device float *x_next = work + total;
  device float *ax = work + 2 * total;

  const float scale = 2.0f / (lambda_max + lambda_min);
  const float alpha = 1.0f - scale * lambda_min;
  const float mu = 1.0f / alpha;
  const float omega_prod = 2.0f / alpha;
  float c_prev = 1.0f;
  float c_cur = mu;

  for (int i = static_cast<int>(lane); i < total;
       i += static_cast<int>(k_linalg_threads)) {
    x_prev[i] = 0.0f;
    x_next[i] = 0.0f;
    ax[i] = 0.0f;
    out[i] = scale * rhs[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int it = 1; it < degree; ++it) {
    csr_spmm_f32(data, indices, indptr, out, ax, n_rows, rhs_cols, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float c_next = 2.0f * mu * c_cur - c_prev;
    const float omega = omega_prod * c_cur / c_next;
    const float one_minus_omega = 1.0f - omega;
    const float omega_scale = omega * scale;
    for (int i = static_cast<int>(lane); i < total;
         i += static_cast<int>(k_linalg_threads)) {
      const float r = rhs[i] - ax[i];
      x_next[i] =
          one_minus_omega * x_prev[i] + omega * out[i] + omega_scale * r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = static_cast<int>(lane); i < total;
         i += static_cast<int>(k_linalg_threads)) {
      x_prev[i] = out[i];
      out[i] = x_next[i];
    }
    c_prev = c_cur;
    c_cur = c_next;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template [[host_name("csr_chebyshev_apply_float32_int32")]] [[kernel]] void
csr_chebyshev_apply_kernel<int>(device const float *, device const int *,
                                device const int *, device const float *,
                                device float *, device float *, constant int &,
                                constant int &, constant int &, constant int &,
                                constant float &, constant float &, uint);

template [[host_name("csr_chebyshev_apply_float32_int64")]] [[kernel]] void
csr_chebyshev_apply_kernel<long>(device const float *, device const long *,
                                 device const long *, device const float *,
                                 device float *, device float *, constant int &,
                                 constant int &, constant int &, constant int &,
                                 constant float &, constant float &, uint);
