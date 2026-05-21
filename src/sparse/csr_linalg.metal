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

#include "sparse/metal_common.h"

constant uint k_linalg_threads = 256;

inline float reduce_sum_256(float value, threadgroup float *scratch, uint lane) {
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

template <typename I>
[[kernel]] void csr_lanczos_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
    device const float *v0 [[buffer(3)]], device float *alphas [[buffer(4)]],
    device float *betas [[buffer(5)]], device float *basis [[buffer(6)]],
    device int *actual [[buffer(7)]], device float *work [[buffer(8)]],
    constant int &n_rows [[buffer(9)]], constant int &n_cols [[buffer(10)]],
    constant int &k [[buffer(11)]], constant int &reorthogonalize [[buffer(12)]],
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
  const float norm0 = sqrt(max(reduce_sum_256(norm_local, scratch, lane), 0.0f));
  for (int row = static_cast<int>(lane); row < n_rows;
       row += static_cast<int>(k_linalg_threads)) {
    basis[row * k] = norm0 <= 1.1920928955078125e-7f
                         ? (row == 0 ? 1.0f : 0.0f)
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
    const float beta = sqrt(max(reduce_sum_256(beta_local, scratch, lane), 0.0f));
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

template <typename I>
[[kernel]] void csr_arnoldi_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
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
  const float norm0 = sqrt(max(reduce_sum_256(norm_local, scratch, lane), 0.0f));
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

template <typename I>
[[kernel]] void csr_triangular_solve_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
    device const float *b [[buffer(3)]], device float *x [[buffer(4)]],
    constant int &n_rows [[buffer(5)]], constant int &n_cols [[buffer(6)]],
    constant int &lower [[buffer(7)]],
    constant int &unit_diagonal [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  (void)n_cols;
  if (tid != 0) {
    return;
  }
  if (lower != 0) {
    for (int row = 0; row < n_rows; ++row) {
      float sum = b[row];
      float diag = 1.0f;
      for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices[p]);
        if (col < row) {
          sum -= data[p] * x[col];
        } else if (col == row) {
          diag = data[p];
        }
      }
      x[row] = unit_diagonal != 0 ? sum : sum / diag;
    }
  } else {
    for (int row = n_rows - 1; row >= 0; --row) {
      float sum = b[row];
      float diag = 1.0f;
      for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices[p]);
        if (col > row) {
          sum -= data[p] * x[col];
        } else if (col == row) {
          diag = data[p];
        }
      }
      x[row] = unit_diagonal != 0 ? sum : sum / diag;
    }
  }
}

inline complex64_t sparse_conjugate(complex64_t value) {
  return complex64_t(value.real, -value.imag);
}

inline float sparse_conjugate(float value) { return value; }

template <typename T, typename I, bool ConjugateLhs>
[[kernel]] void csr_vdot_kernel(
    device const T *lhs_data [[buffer(0)]],
    device const I *lhs_indices [[buffer(1)]],
    device const I *lhs_indptr [[buffer(2)]],
    device const T *rhs_data [[buffer(3)]],
    device const I *rhs_indices [[buffer(4)]],
    device const I *rhs_indptr [[buffer(5)]], device T *out [[buffer(6)]],
    constant int &n_rows [[buffer(7)]], constant int &n_cols [[buffer(8)]],
    uint lane [[thread_index_in_threadgroup]]) {
  (void)n_cols;
  typedef typename sparse_accumulator<T>::type acc_t;
  threadgroup acc_t scratch[256];
  acc_t local = sparse_accumulator<T>::zero();
  for (int row = static_cast<int>(lane); row < n_rows;
       row += static_cast<int>(k_linalg_threads)) {
    I lp = lhs_indptr[row];
    I rp = rhs_indptr[row];
    const I lend = lhs_indptr[row + 1];
    const I rend = rhs_indptr[row + 1];
    while (lp < lend && rp < rend) {
      const I lc = lhs_indices[lp];
      const I rc = rhs_indices[rp];
      if (lc == rc) {
        const T lhs = ConjugateLhs ? sparse_conjugate(lhs_data[lp]) : lhs_data[lp];
        local += sparse_multiply<T>(lhs, rhs_data[rp]);
        ++lp;
        ++rp;
      } else if (lc < rc) {
        ++lp;
      } else {
        ++rp;
      }
    }
  }
  const acc_t reduced = reduce_sum_256_any<acc_t>(local, scratch, lane);
  if (lane == 0) {
    out[0] = sparse_accumulator<T>::cast(reduced);
  }
}

[[host_name("csr_permute_vector_float32")]] [[kernel]] void
csr_permute_vector_float32_kernel(
    device const float *x [[buffer(0)]], device const int *perm [[buffer(1)]],
    device float *out [[buffer(2)]], constant int &size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (static_cast<int>(tid) >= size) {
    return;
  }
  out[tid] = x[perm[tid]];
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

template [[host_name("csr_triangular_solve_float32_int32")]] [[kernel]] void
csr_triangular_solve_kernel<int>(
    device const float *, device const int *, device const int *,
    device const float *, device float *, constant int &, constant int &,
    constant int &, constant int &, uint);

template [[host_name("csr_triangular_solve_float32_int64")]] [[kernel]] void
csr_triangular_solve_kernel<long>(
    device const float *, device const long *, device const long *,
    device const float *, device float *, constant int &, constant int &,
    constant int &, constant int &, uint);

#define INSTANTIATE_CSR_INNER(OP, NAME, T, I, CONJ)                            \
  template [[host_name(#OP "_" #NAME)]] [[kernel]] void                        \
  csr_vdot_kernel<T, I, CONJ>(device const T *, device const I *,              \
                              device const I *, device const T *,              \
                              device const I *, device const I *, device T *,  \
                              constant int &, constant int &, uint)

INSTANTIATE_CSR_INNER(csr_vdot, float32_int32, float, int, true);
INSTANTIATE_CSR_INNER(csr_vdot, float32_int64, float, long, true);
INSTANTIATE_CSR_INNER(csr_vdot, complex64_int32, complex64_t, int, true);
INSTANTIATE_CSR_INNER(csr_vdot, complex64_int64, complex64_t, long, true);
INSTANTIATE_CSR_INNER(csr_dot, float32_int32, float, int, false);
INSTANTIATE_CSR_INNER(csr_dot, float32_int64, float, long, false);
INSTANTIATE_CSR_INNER(csr_dot, complex64_int32, complex64_t, int, false);
INSTANTIATE_CSR_INNER(csr_dot, complex64_int64, complex64_t, long, false);

#undef INSTANTIATE_CSR_INNER
