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
[[kernel]] void csr_minres_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *b [[buffer(3)]], device const float *x0 [[buffer(4)]],
    device float *x [[buffer(5)]], device int *info [[buffer(6)]],
    device float *residual [[buffer(7)]], device int *iterations [[buffer(8)]],
    device float *work [[buffer(9)]], constant int &n_rows [[buffer(10)]],
    constant int &n_cols [[buffer(11)]], constant int &maxiter [[buffer(12)]],
    constant float &rtol [[buffer(13)]], constant float &atol [[buffer(14)]],
    constant float &shift [[buffer(15)]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)n_cols;
  constexpr float eps = 1.1920928955078125e-7f;
  threadgroup float scratch[256];
  threadgroup float shared_tol;
  threadgroup float shared_beta;
  threadgroup float shared_oldb;
  threadgroup float shared_dbar;
  threadgroup float shared_epsln;
  threadgroup float shared_phibar;
  threadgroup float shared_cs;
  threadgroup float shared_sn;
  threadgroup float shared_alfa;
  threadgroup float shared_delta;
  threadgroup float shared_gamma;
  threadgroup float shared_phi;
  threadgroup float shared_residual;
  threadgroup int shared_status;
  threadgroup int shared_iters;

  device float *r1 = work;
  device float *r2 = work + n_rows;
  device float *y = work + 2 * n_rows;
  device float *v = work + 3 * n_rows;
  device float *w = work + 4 * n_rows;
  device float *w1 = work + 5 * n_rows;
  device float *w2 = work + 6 * n_rows;
  device float *av = work + 7 * n_rows;

  for (int i = static_cast<int>(lane); i < n_rows;
       i += static_cast<int>(k_linalg_threads)) {
    x[i] = x0[i];
    r1[i] = 0.0f;
    r2[i] = 0.0f;
    y[i] = 0.0f;
    v[i] = 0.0f;
    w[i] = 0.0f;
    w1[i] = 0.0f;
    w2[i] = 0.0f;
    av[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  csr_spmv_f32(data, indices, indptr, x, av, n_rows, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float rr_local = 0.0f;
  float b_local = 0.0f;
  float invalid_local = !isfinite(shift) ? 1.0f : 0.0f;
  for (int i = static_cast<int>(lane); i < n_rows;
       i += static_cast<int>(k_linalg_threads)) {
    const float ai = av[i] - shift * x[i];
    const float ri = b[i] - ai;
    r2[i] = ri;
    y[i] = ri;
    rr_local += ri * ri;
    b_local += b[i] * b[i];
    invalid_local +=
        (!isfinite(ai) || !isfinite(ri) || !isfinite(b[i]) || !isfinite(x[i]))
            ? 1.0f
            : 0.0f;
  }
  const float rr0 = reduce_sum_256(rr_local, scratch, lane);
  const float bb = reduce_sum_256(b_local, scratch, lane);
  const float invalid0 = reduce_sum_256(invalid_local, scratch, lane);
  if (lane == 0) {
    shared_tol = max(atol, rtol * sqrt(max(bb, 0.0f)));
    shared_residual = sqrt(max(rr0, 0.0f));
    shared_status = maxiter > 0 ? maxiter : 1;
    shared_iters = 0;
    shared_beta = sqrt(max(rr0, 0.0f));
    shared_oldb = 0.0f;
    shared_dbar = 0.0f;
    shared_epsln = 0.0f;
    shared_phibar = shared_beta;
    shared_cs = -1.0f;
    shared_sn = 0.0f;
    if (invalid0 != 0.0f || !isfinite(rr0) || !isfinite(shared_residual)) {
      shared_status = -3;
    } else if (shared_residual <= shared_tol) {
      shared_status = 0;
    } else if (maxiter == 0) {
      shared_status = 1;
    } else if (shared_beta <= eps || !isfinite(shared_beta)) {
      shared_status = -1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int it = 1; it <= maxiter; ++it) {
    if (shared_status <= 0) {
      break;
    }

    const float inv_beta = 1.0f / shared_beta;
    float setup_invalid = !isfinite(inv_beta) ? 1.0f : 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      const float vi = inv_beta * y[i];
      v[i] = vi;
      setup_invalid += !isfinite(vi) ? 1.0f : 0.0f;
    }
    const float setup_invalid_sum =
        reduce_sum_256(setup_invalid, scratch, lane);
    if (lane == 0 && setup_invalid_sum != 0.0f) {
      shared_status = -3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    csr_spmv_f32(data, indices, indptr, v, av, n_rows, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float prev_scale = shared_beta / shared_oldb;
    float apply_invalid = 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      float wi = av[i] - shift * v[i];
      if (it >= 2) {
        wi -= prev_scale * r1[i];
      }
      av[i] = wi;
      apply_invalid += !isfinite(wi) ? 1.0f : 0.0f;
    }
    const float apply_invalid_sum =
        reduce_sum_256(apply_invalid, scratch, lane);
    if (lane == 0) {
      if ((it >= 2 && (!isfinite(prev_scale) || fabs(shared_oldb) <= eps)) ||
          apply_invalid_sum != 0.0f) {
        shared_status = -3;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    const float alfa = vector_dot_f32(v, av, n_rows, scratch, lane);
    if (lane == 0) {
      shared_alfa = alfa;
      if (!isfinite(alfa)) {
        shared_status = -3;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    const float diag_scale = shared_alfa / shared_beta;
    float r_invalid = !isfinite(diag_scale) ? 1.0f : 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      const float new_r = av[i] - diag_scale * r2[i];
      r1[i] = r2[i];
      r2[i] = new_r;
      y[i] = new_r;
      r_invalid += !isfinite(new_r) ? 1.0f : 0.0f;
    }
    const float r_invalid_sum = reduce_sum_256(r_invalid, scratch, lane);
    const float beta_inner = vector_dot_f32(r2, y, n_rows, scratch, lane);
    if (lane == 0) {
      if (r_invalid_sum != 0.0f || !isfinite(beta_inner) || beta_inner < 0.0f) {
        shared_status = -3;
      } else {
        shared_oldb = shared_beta;
        shared_beta = sqrt(max(beta_inner, 0.0f));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    if (lane == 0) {
      const float oldeps = shared_epsln;
      const float delta = shared_cs * shared_dbar + shared_sn * shared_alfa;
      const float gbar = shared_sn * shared_dbar - shared_cs * shared_alfa;
      shared_epsln = shared_sn * shared_beta;
      shared_dbar = -shared_cs * shared_beta;
      float gamma = sqrt(gbar * gbar + shared_beta * shared_beta);
      if (!isfinite(gamma)) {
        shared_status = -3;
      } else {
        gamma = max(gamma, eps);
        shared_cs = gbar / gamma;
        shared_sn = shared_beta / gamma;
        shared_phi = shared_cs * shared_phibar;
        shared_phibar = shared_sn * shared_phibar;
        shared_delta = delta;
        shared_gamma = gamma;
        shared_alfa = oldeps;
        if (!isfinite(shared_phi) || !isfinite(shared_phibar) ||
            !isfinite(shared_delta) || !isfinite(shared_gamma)) {
          shared_status = -3;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    float update_invalid = 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      const float prev_w1 = w2[i];
      const float prev_w2 = w[i];
      w1[i] = prev_w1;
      w2[i] = prev_w2;
      const float wi = (v[i] - shared_alfa * prev_w1 - shared_delta * prev_w2) /
                       shared_gamma;
      w[i] = wi;
      const float xi = x[i] + shared_phi * wi;
      x[i] = xi;
      update_invalid += (!isfinite(wi) || !isfinite(xi)) ? 1.0f : 0.0f;
    }
    const float update_invalid_sum =
        reduce_sum_256(update_invalid, scratch, lane);
    if (lane == 0) {
      shared_iters = it;
      if (update_invalid_sum != 0.0f) {
        shared_status = -3;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shared_status < 0) {
      break;
    }

    csr_spmv_f32(data, indices, indptr, x, av, n_rows, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float true_rr_local = 0.0f;
    float true_invalid = 0.0f;
    for (int i = static_cast<int>(lane); i < n_rows;
         i += static_cast<int>(k_linalg_threads)) {
      const float ai = av[i] - shift * x[i];
      const float ri = b[i] - ai;
      true_rr_local += ri * ri;
      true_invalid += (!isfinite(ai) || !isfinite(ri)) ? 1.0f : 0.0f;
    }
    const float true_rr = reduce_sum_256(true_rr_local, scratch, lane);
    const float true_invalid_sum = reduce_sum_256(true_invalid, scratch, lane);
    if (lane == 0) {
      shared_residual = sqrt(max(true_rr, 0.0f));
      if (true_invalid_sum != 0.0f || !isfinite(true_rr) ||
          !isfinite(shared_residual)) {
        shared_status = -3;
      } else if (shared_residual <= shared_tol) {
        shared_status = 0;
      } else if (shared_beta <= eps || !isfinite(shared_beta)) {
        shared_status = -1;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    info[0] = shared_status;
    residual[0] = shared_residual;
    iterations[0] = shared_iters;
  }
}

template [[host_name("csr_minres_float32_int32")]] [[kernel]] void
csr_minres_kernel<int>(device const float *, device const int *,
                       device const int *, device const float *,
                       device const float *, device float *, device int *,
                       device float *, device int *, device float *,
                       constant int &, constant int &, constant int &,
                       constant float &, constant float &, constant float &,
                       uint);

template [[host_name("csr_minres_float32_int64")]] [[kernel]] void
csr_minres_kernel<long>(device const float *, device const long *,
                        device const long *, device const float *,
                        device const float *, device float *, device int *,
                        device float *, device int *, device float *,
                        constant int &, constant int &, constant int &,
                        constant float &, constant float &, constant float &,
                        uint);
