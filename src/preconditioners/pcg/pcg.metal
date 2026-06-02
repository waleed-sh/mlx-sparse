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
[[kernel]] void csr_pcg_jacobi_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *b [[buffer(3)]], device const float *x0 [[buffer(4)]],
    device const float *inv_diag [[buffer(5)]], device float *x [[buffer(6)]],
    device int *info [[buffer(7)]], device float *residual [[buffer(8)]],
    device int *iterations [[buffer(9)]], device float *work [[buffer(10)]],
    constant int &n_rows [[buffer(11)]], constant int &n_cols [[buffer(12)]],
    constant int &maxiter [[buffer(13)]], constant float &rtol [[buffer(14)]],
    constant float &atol [[buffer(15)]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)n_cols;
  threadgroup float scratch[256];
  threadgroup float shared_rho;
  threadgroup float shared_true_rr;
  threadgroup float shared_tol;
  threadgroup float shared_denom;
  threadgroup float shared_rho_new;
  threadgroup int shared_status;
  threadgroup int shared_iters;

  device float *r = work;
  device float *z = work + n_rows;
  device float *p = work + 2 * n_rows;
  device float *ap = work + 3 * n_rows;

  for (int i = static_cast<int>(lane); i < n_rows;
       i += static_cast<int>(k_linalg_threads)) {
    x[i] = x0[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  csr_spmv_f32(data, indices, indptr, x, ap, n_rows, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float rr_acc = 0.0f;
  float rho_acc = 0.0f;
  float b_acc = 0.0f;
  float invalid_acc = 0.0f;
  for (int i = static_cast<int>(lane); i < n_rows;
       i += static_cast<int>(k_linalg_threads)) {
    const float ri = b[i] - ap[i];
    const float zi = inv_diag[i] * ri;
    r[i] = ri;
    z[i] = zi;
    p[i] = zi;
    rr_acc += ri * ri;
    rho_acc += ri * zi;
    b_acc += b[i] * b[i];
    invalid_acc += (!isfinite(ri) || !isfinite(zi) || !isfinite(inv_diag[i]))
                       ? 1.0f
                       : 0.0f;
  }
  const float rr0 = reduce_sum_256(rr_acc, scratch, lane);
  const float rho0 = reduce_sum_256(rho_acc, scratch, lane);
  const float bb = reduce_sum_256(b_acc, scratch, lane);
  const float invalid0 = reduce_sum_256(invalid_acc, scratch, lane);
  if (lane == 0) {
    shared_true_rr = rr0;
    shared_rho = rho0;
    shared_tol = max(atol, rtol * sqrt(max(bb, 0.0f)));
    shared_status = maxiter > 0 ? maxiter : 1;
    shared_iters = 0;
    const float true_res = sqrt(max(rr0, 0.0f));
    if (invalid0 != 0.0f || !isfinite(rr0) || !isfinite(rho0)) {
      shared_status = -3;
    } else if (true_res <= shared_tol) {
      shared_status = 0;
    } else if (rho0 <= 0.0f) {
      shared_status = -2;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int it = 1; it <= maxiter; ++it) {
    if (shared_status <= 0) {
      break;
    }

    csr_spmv_f32(data, indices, indptr, p, ap, n_rows, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float denom = vector_dot_f32(p, ap, n_rows, scratch, lane);
    const float p_norm2 = vector_dot_f32(p, p, n_rows, scratch, lane);
    const float ap_norm2 = vector_dot_f32(ap, ap, n_rows, scratch, lane);
    if (lane == 0) {
      const float denom_scale = sqrt(max(p_norm2 * ap_norm2, 0.0f));
      const float denom_tol = 1.1920928955078125e-7f * max(1.0f, denom_scale);
      shared_denom = denom;
      if (!isfinite(denom) || !isfinite(denom_scale) ||
          fabs(denom) <= denom_tol) {
        shared_status = -1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    const float alpha = shared_rho / shared_denom;
    float rr_new_local = 0.0f;
    float rho_new_local = 0.0f;
    float invalid_local = !isfinite(alpha) ? 1.0f : 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      const float xi = x[i] + alpha * p[i];
      const float ri = r[i] - alpha * ap[i];
      const float zi = inv_diag[i] * ri;
      x[i] = xi;
      r[i] = ri;
      z[i] = zi;
      rr_new_local += ri * ri;
      rho_new_local += ri * zi;
      invalid_local +=
          (!isfinite(xi) || !isfinite(ri) || !isfinite(zi)) ? 1.0f : 0.0f;
    }
    const float rr_new = reduce_sum_256(rr_new_local, scratch, lane);
    const float rho_new = reduce_sum_256(rho_new_local, scratch, lane);
    const float invalid_new = reduce_sum_256(invalid_local, scratch, lane);
    if (lane == 0) {
      shared_true_rr = rr_new;
      shared_rho_new = rho_new;
      shared_iters = it;
      const float r_norm = sqrt(max(rr_new, 0.0f));
      if (invalid_new != 0.0f || !isfinite(rr_new) || !isfinite(r_norm)) {
        shared_status = -3;
      } else if (r_norm <= shared_tol) {
        shared_status = 0;
      } else if (!isfinite(rho_new) || rho_new <= 0.0f) {
        shared_status = -2;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status <= 0) {
      if (lane == 0 && shared_status == 0) {
        shared_rho = shared_rho_new;
      }
      break;
    }

    const float beta = shared_rho_new / shared_rho;
    float beta_invalid = !isfinite(beta) ? 1.0f : 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      const float pi = z[i] + beta * p[i];
      p[i] = pi;
      beta_invalid += !isfinite(pi) ? 1.0f : 0.0f;
    }
    const float beta_invalid_sum = reduce_sum_256(beta_invalid, scratch, lane);
    if (lane == 0) {
      if (beta_invalid_sum != 0.0f) {
        shared_status = -3;
      }
      shared_rho = shared_rho_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    info[0] = shared_status;
    residual[0] = sqrt(max(shared_true_rr, 0.0f));
    iterations[0] = shared_iters;
  }
}

template [[host_name("csr_pcg_jacobi_float32_int32")]] [[kernel]] void
csr_pcg_jacobi_kernel<int>(device const float *, device const int *,
                           device const int *, device const float *,
                           device const float *, device const float *,
                           device float *, device int *, device float *,
                           device int *, device float *, constant int &,
                           constant int &, constant int &, constant float &,
                           constant float &, uint);

template [[host_name("csr_pcg_jacobi_float32_int64")]] [[kernel]] void
csr_pcg_jacobi_kernel<long>(device const float *, device const long *,
                            device const long *, device const float *,
                            device const float *, device const float *,
                            device float *, device int *, device float *,
                            device int *, device float *, constant int &,
                            constant int &, constant int &, constant float &,
                            constant float &, uint);
