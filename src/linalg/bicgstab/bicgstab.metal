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
[[kernel]] void csr_bicgstab_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *b [[buffer(3)]], device const float *x0 [[buffer(4)]],
    device float *x [[buffer(5)]], device int *info [[buffer(6)]],
    device float *residual [[buffer(7)]], device int *iterations [[buffer(8)]],
    device float *work [[buffer(9)]], constant int &n_rows [[buffer(10)]],
    constant int &n_cols [[buffer(11)]], constant int &maxiter [[buffer(12)]],
    constant float &rtol [[buffer(13)]], constant float &atol [[buffer(14)]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)n_cols;
  threadgroup float scratch[256];
  threadgroup float shared_tol;
  threadgroup float shared_rr;
  threadgroup float shared_rho;
  threadgroup float shared_rho_prev;
  threadgroup float shared_alpha;
  threadgroup float shared_omega;
  threadgroup int shared_status;
  threadgroup int shared_iters;
  threadgroup int shared_need_true;

  device float *r = work;
  device float *r_hat = work + n_rows;
  device float *p = work + 2 * n_rows;
  device float *v = work + 3 * n_rows;
  device float *s = work + 4 * n_rows;
  device float *t = work + 5 * n_rows;
  device float *ax = work + 6 * n_rows;

  for (int i = static_cast<int>(lane); i < n_rows;
       i += static_cast<int>(k_linalg_threads)) {
    x[i] = x0[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  csr_spmv_f32(data, indices, indptr, x, ax, n_rows, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float rr_acc = 0.0f;
  float bb_acc = 0.0f;
  float invalid_acc = 0.0f;
  for (int i = static_cast<int>(lane); i < n_rows;
       i += static_cast<int>(k_linalg_threads)) {
    const float ri = b[i] - ax[i];
    r[i] = ri;
    r_hat[i] = ri;
    p[i] = 0.0f;
    v[i] = 0.0f;
    rr_acc += ri * ri;
    bb_acc += b[i] * b[i];
    invalid_acc +=
        (!isfinite(ri) || !isfinite(b[i]) || !isfinite(x[i])) ? 1.0f : 0.0f;
  }
  const float rr0 = reduce_sum_256(rr_acc, scratch, lane);
  const float bb = reduce_sum_256(bb_acc, scratch, lane);
  const float invalid0 = reduce_sum_256(invalid_acc, scratch, lane);
  if (lane == 0) {
    shared_rr = rr0;
    shared_tol = max(atol, rtol * sqrt(max(bb, 0.0f)));
    shared_rho_prev = 1.0f;
    shared_alpha = 1.0f;
    shared_omega = 1.0f;
    shared_status = maxiter > 0 ? maxiter : 1;
    shared_iters = 0;
    const float initial_residual = sqrt(max(rr0, 0.0f));
    if (invalid0 != 0.0f || !isfinite(rr0) || !isfinite(initial_residual) ||
        !isfinite(shared_tol)) {
      shared_status = -3;
    } else if (initial_residual <= shared_tol) {
      shared_status = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int it = 1; it <= maxiter; ++it) {
    if (shared_status <= 0) {
      break;
    }

    const float rho = vector_dot_f32(r_hat, r, n_rows, scratch, lane);
    const float r_hat_norm2 =
        vector_dot_f32(r_hat, r_hat, n_rows, scratch, lane);
    const float r_norm2 = vector_dot_f32(r, r, n_rows, scratch, lane);
    if (lane == 0) {
      shared_rho = rho;
      const float scale = sqrt(max(r_hat_norm2 * r_norm2, 0.0f));
      const float tol = 1.1920928955078125e-7f * max(1.0f, scale);
      if (!isfinite(rho) || !isfinite(scale) || fabs(rho) <= tol) {
        shared_status = !isfinite(rho) || !isfinite(scale) ? -3 : -1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    if (it == 1) {
      for (int i = static_cast<int>(lane); i < n_rows;
           i += static_cast<int>(k_linalg_threads)) {
        p[i] = r[i];
      }
    } else {
      const float beta =
          (shared_rho / shared_rho_prev) * (shared_alpha / shared_omega);
      float invalid_beta = (!isfinite(beta) || !isfinite(shared_omega) ||
                            fabs(shared_omega) <= 1.1920928955078125e-7f)
                               ? 1.0f
                               : 0.0f;
      for (int i = static_cast<int>(lane); i < n_rows;
           i += static_cast<int>(k_linalg_threads)) {
        const float pi = r[i] + beta * (p[i] - shared_omega * v[i]);
        p[i] = pi;
        invalid_beta += !isfinite(pi) ? 1.0f : 0.0f;
      }
      const float invalid_beta_sum =
          reduce_sum_256(invalid_beta, scratch, lane);
      if (lane == 0 && invalid_beta_sum != 0.0f) {
        shared_status = fabs(shared_omega) <= 1.1920928955078125e-7f ? -1 : -3;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    csr_spmv_f32(data, indices, indptr, p, v, n_rows, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float alpha_den = vector_dot_f32(r_hat, v, n_rows, scratch, lane);
    const float v_norm2 = vector_dot_f32(v, v, n_rows, scratch, lane);
    if (lane == 0) {
      const float scale = sqrt(max(r_hat_norm2 * v_norm2, 0.0f));
      const float tol = 1.1920928955078125e-7f * max(1.0f, scale);
      if (!isfinite(alpha_den) || !isfinite(scale) || fabs(alpha_den) <= tol) {
        shared_status = !isfinite(alpha_den) || !isfinite(scale) ? -3 : -1;
      } else {
        shared_alpha = shared_rho / alpha_den;
        if (!isfinite(shared_alpha)) {
          shared_status = -3;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    float s_acc = 0.0f;
    float s_invalid = 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      const float si = r[i] - shared_alpha * v[i];
      const float xi = x[i] + shared_alpha * p[i];
      s[i] = si;
      x[i] = xi;
      s_acc += si * si;
      s_invalid += (!isfinite(si) || !isfinite(xi)) ? 1.0f : 0.0f;
    }
    const float s_norm2 = reduce_sum_256(s_acc, scratch, lane);
    const float s_invalid_sum = reduce_sum_256(s_invalid, scratch, lane);
    if (lane == 0) {
      shared_iters = it;
      const float s_norm = sqrt(max(s_norm2, 0.0f));
      shared_need_true = 0;
      if (s_invalid_sum != 0.0f || !isfinite(s_norm2) || !isfinite(s_norm)) {
        shared_status = -3;
      } else if (s_norm <= shared_tol) {
        shared_need_true = 1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }
    if (shared_need_true != 0) {
      csr_spmv_f32(data, indices, indptr, x, ax, n_rows, lane);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float true_acc = 0.0f;
      float true_invalid = 0.0f;
      for (int i = static_cast<int>(lane); i < n_rows;
           i += static_cast<int>(k_linalg_threads)) {
        const float ri = b[i] - ax[i];
        true_acc += ri * ri;
        true_invalid += (!isfinite(ri) || !isfinite(ax[i]) || !isfinite(x[i]))
                            ? 1.0f
                            : 0.0f;
      }
      const float true_rr = reduce_sum_256(true_acc, scratch, lane);
      const float true_invalid_sum =
          reduce_sum_256(true_invalid, scratch, lane);
      if (lane == 0) {
        shared_rr = true_rr;
        const float true_norm = sqrt(max(true_rr, 0.0f));
        if (true_invalid_sum != 0.0f || !isfinite(true_rr) ||
            !isfinite(true_norm)) {
          shared_status = -3;
        } else if (true_norm <= shared_tol) {
          shared_status = 0;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (shared_status <= 0) {
        break;
      }
    }

    csr_spmv_f32(data, indices, indptr, s, t, n_rows, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float omega_num = vector_dot_f32(t, s, n_rows, scratch, lane);
    const float omega_den = vector_dot_f32(t, t, n_rows, scratch, lane);
    if (lane == 0) {
      const float omega_tol =
          1.1920928955078125e-7f * max(1.0f, sqrt(max(omega_den, 0.0f)));
      if (!isfinite(omega_num) || !isfinite(omega_den) ||
          omega_den <= omega_tol) {
        shared_status = !isfinite(omega_num) || !isfinite(omega_den) ? -3 : -1;
      } else {
        shared_omega = omega_num / omega_den;
        if (!isfinite(shared_omega)) {
          shared_status = -3;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    float rr_new_acc = 0.0f;
    float update_invalid = 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      const float xi = x[i] + shared_omega * s[i];
      const float ri = s[i] - shared_omega * t[i];
      x[i] = xi;
      r[i] = ri;
      rr_new_acc += ri * ri;
      update_invalid += (!isfinite(xi) || !isfinite(ri)) ? 1.0f : 0.0f;
    }
    const float rr_new = reduce_sum_256(rr_new_acc, scratch, lane);
    const float update_invalid_sum =
        reduce_sum_256(update_invalid, scratch, lane);
    if (lane == 0) {
      shared_rr = rr_new;
      const float r_norm = sqrt(max(rr_new, 0.0f));
      shared_need_true = 0;
      if (update_invalid_sum != 0.0f || !isfinite(rr_new) ||
          !isfinite(r_norm)) {
        shared_status = -3;
      } else if (r_norm <= shared_tol) {
        shared_need_true = 1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }
    if (shared_need_true != 0) {
      csr_spmv_f32(data, indices, indptr, x, ax, n_rows, lane);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float true_acc = 0.0f;
      float true_invalid = 0.0f;
      for (int i = static_cast<int>(lane); i < n_rows;
           i += static_cast<int>(k_linalg_threads)) {
        const float ri = b[i] - ax[i];
        r[i] = ri;
        true_acc += ri * ri;
        true_invalid += (!isfinite(ri) || !isfinite(ax[i]) || !isfinite(x[i]))
                            ? 1.0f
                            : 0.0f;
      }
      const float true_rr = reduce_sum_256(true_acc, scratch, lane);
      const float true_invalid_sum =
          reduce_sum_256(true_invalid, scratch, lane);
      if (lane == 0) {
        shared_rr = true_rr;
        const float true_norm = sqrt(max(true_rr, 0.0f));
        if (true_invalid_sum != 0.0f || !isfinite(true_rr) ||
            !isfinite(true_norm)) {
          shared_status = -3;
        } else if (true_norm <= shared_tol) {
          shared_status = 0;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (shared_status <= 0) {
        break;
      }
    }

    if (lane == 0) {
      if (fabs(shared_omega) <= 1.1920928955078125e-7f) {
        shared_status = -1;
      }
      shared_rho_prev = shared_rho;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (shared_status > 0 || shared_status == -1) {
    csr_spmv_f32(data, indices, indptr, x, ax, n_rows, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float true_acc = 0.0f;
    float true_invalid = 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      const float ri = b[i] - ax[i];
      true_acc += ri * ri;
      true_invalid +=
          (!isfinite(ri) || !isfinite(ax[i]) || !isfinite(x[i])) ? 1.0f : 0.0f;
    }
    const float true_rr = reduce_sum_256(true_acc, scratch, lane);
    const float true_invalid_sum = reduce_sum_256(true_invalid, scratch, lane);
    if (lane == 0) {
      shared_rr = true_rr;
      const float true_norm = sqrt(max(true_rr, 0.0f));
      if (true_invalid_sum != 0.0f || !isfinite(true_rr) ||
          !isfinite(true_norm)) {
        shared_status = -3;
      } else if (true_norm <= shared_tol) {
        shared_status = 0;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane == 0) {
    info[0] = shared_status;
    residual[0] = sqrt(max(shared_rr, 0.0f));
    iterations[0] = shared_iters;
  }
}

template [[host_name("csr_bicgstab_float32_int32")]] [[kernel]] void
csr_bicgstab_kernel<int>(device const float *, device const int *,
                         device const int *, device const float *,
                         device const float *, device float *, device int *,
                         device float *, device int *, device float *,
                         constant int &, constant int &, constant int &,
                         constant float &, constant float &, uint);

template [[host_name("csr_bicgstab_float32_int64")]] [[kernel]] void
csr_bicgstab_kernel<long>(device const float *, device const long *,
                          device const long *, device const float *,
                          device const float *, device float *, device int *,
                          device float *, device int *, device float *,
                          constant int &, constant int &, constant int &,
                          constant float &, constant float &, uint);
