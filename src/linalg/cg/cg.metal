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
[[kernel]] void csr_cg_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
    device const float *b [[buffer(3)]], device const float *x0 [[buffer(4)]],
    device float *x [[buffer(5)]], device int *info [[buffer(6)]],
    device float *residual [[buffer(7)]], device int *iterations [[buffer(8)]],
    device float *work [[buffer(9)]], constant int &n_rows [[buffer(10)]],
    constant int &n_cols [[buffer(11)]],
    constant int &maxiter [[buffer(12)]], constant float &rtol [[buffer(13)]],
    constant float &atol [[buffer(14)]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)n_cols;
  threadgroup float scratch[256];
  threadgroup float shared_rr;
  threadgroup float shared_tol;
  threadgroup float shared_denom;
  threadgroup float shared_rr_new;
  threadgroup int shared_status;
  threadgroup int shared_iters;

  device float *r = work;
  device float *p = work + n_rows;
  device float *ap = work + 2 * n_rows;

  for (int i = static_cast<int>(lane); i < n_rows;
       i += static_cast<int>(k_linalg_threads)) {
    x[i] = x0[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  csr_spmv_f32(data, indices, indptr, x, ap, n_rows, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float r_acc = 0.0f;
  float b_acc = 0.0f;
  for (int i = static_cast<int>(lane); i < n_rows;
       i += static_cast<int>(k_linalg_threads)) {
    const float ri = b[i] - ap[i];
    r[i] = ri;
    p[i] = ri;
    r_acc += ri * ri;
    b_acc += b[i] * b[i];
  }
  const float rr0 = reduce_sum_256(r_acc, scratch, lane);
  const float bb = reduce_sum_256(b_acc, scratch, lane);
  if (lane == 0) {
    shared_rr = rr0;
    shared_tol = max(atol, rtol * sqrt(max(bb, 0.0f)));
    shared_status = sqrt(max(rr0, 0.0f)) <= shared_tol ? 0 : maxiter;
    shared_iters = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int it = 1; it <= maxiter; ++it) {
    if (shared_status == 0 || shared_status == -1) {
      break;
    }

    csr_spmv_f32(data, indices, indptr, p, ap, n_rows, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float denom = vector_dot_f32(p, ap, n_rows, scratch, lane);
    if (lane == 0) {
      shared_denom = denom;
      if (fabs(denom) <= 1.1920928955078125e-7f) {
        shared_status = -1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status == -1) {
      break;
    }

    const float alpha = shared_rr / shared_denom;
    float rr_new_local = 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      x[i] += alpha * p[i];
      const float ri = r[i] - alpha * ap[i];
      r[i] = ri;
      rr_new_local += ri * ri;
    }
    const float rr_new = reduce_sum_256(rr_new_local, scratch, lane);
    if (lane == 0) {
      shared_rr_new = rr_new;
      shared_iters = it;
      if (sqrt(max(rr_new, 0.0f)) <= shared_tol) {
        shared_status = 0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status == 0) {
      shared_rr = shared_rr_new;
      break;
    }

    const float beta = shared_rr_new / shared_rr;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      p[i] = r[i] + beta * p[i];
    }
    if (lane == 0) {
      shared_rr = shared_rr_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    info[0] = shared_status;
    residual[0] = sqrt(max(shared_rr, 0.0f));
    iterations[0] = shared_iters;
  }
}

template [[host_name("csr_cg_float32_int32")]] [[kernel]] void
csr_cg_kernel<int>(device const float *, device const int *, device const int *,
                   device const float *, device const float *, device float *,
                   device int *, device float *, device int *, device float *,
                   constant int &, constant int &, constant int &,
                   constant float &, constant float &, uint);

template [[host_name("csr_cg_float32_int64")]] [[kernel]] void
csr_cg_kernel<long>(device const float *, device const long *,
                    device const long *, device const float *,
                    device const float *, device float *, device int *,
                    device float *, device int *, device float *,
                    constant int &, constant int &, constant int &,
                    constant float &, constant float &, uint);
