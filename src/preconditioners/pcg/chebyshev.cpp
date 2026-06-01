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

#include "preconditioners/pcg/pcg.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "linalg/common/common.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

class CSRPCGChebyshev : public mx::Primitive {
public:
  CSRPCGChebyshev(mx::Stream stream, int n_rows, int n_cols, int degree,
                  float lambda_min, float lambda_max, float rtol, float atol,
                  int maxiter)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), degree_(degree),
        lambda_min_(lambda_min), lambda_max_(lambda_max), rtol_(rtol),
        atol_(atol), maxiter_(maxiter) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRPCGChebyshev"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRPCGChebyshev &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           degree_ == rhs.degree_ && lambda_min_ == rhs.lambda_min_ &&
           lambda_max_ == rhs.lambda_max_ && rtol_ == rhs.rtol_ &&
           atol_ == rhs.atol_ && maxiter_ == rhs.maxiter_;
  }

private:
  int n_rows_;
  int n_cols_;
  int degree_;
  float lambda_min_;
  float lambda_max_;
  float rtol_;
  float atol_;
  int maxiter_;
};

inline bool finite_float(float value) { return std::isfinite(value); }

void validate_chebyshev_interval(int degree, float lambda_min,
                                 float lambda_max) {
  if (degree <= 0) {
    throw std::invalid_argument("csr_pcg_chebyshev degree must be positive.");
  }
  if (!finite_float(lambda_min) || !finite_float(lambda_max) ||
      lambda_min <= 0.0f || lambda_max <= lambda_min) {
    throw std::invalid_argument(
        "csr_pcg_chebyshev requires finite bounds satisfying "
        "0 < lambda_min < lambda_max.");
  }
}

template <typename I>
void chebyshev_apply_vector(const float *m_data, const I *m_indices,
                            const I *m_indptr, const float *rhs, float *out,
                            std::vector<float> &x_prev,
                            std::vector<float> &x_next, std::vector<float> &ax,
                            int n_rows, int degree, float lambda_min,
                            float lambda_max) {
  const float scale = 2.0f / (lambda_max + lambda_min);
  const float alpha = 1.0f - scale * lambda_min;
  const float mu = 1.0f / alpha;
  const float omega_prod = 2.0f / alpha;
  float c_prev = 1.0f;
  float c_cur = mu;

  std::fill(x_prev.begin(), x_prev.end(), 0.0f);
  std::fill(x_next.begin(), x_next.end(), 0.0f);
  for (int i = 0; i < n_rows; ++i) {
    out[i] = scale * rhs[i];
  }
  if (degree == 1) {
    return;
  }

  for (int it = 1; it < degree; ++it) {
    csr_spmv_float(m_data, m_indices, m_indptr, out, ax.data(), n_rows);
    const float c_next = 2.0f * mu * c_cur - c_prev;
    const float omega = omega_prod * c_cur / c_next;
    const float one_minus_omega = 1.0f - omega;
    const float omega_scale = omega * scale;
    for (int i = 0; i < n_rows; ++i) {
      const float r = rhs[i] - ax[static_cast<size_t>(i)];
      x_next[static_cast<size_t>(i)] =
          one_minus_omega * x_prev[static_cast<size_t>(i)] + omega * out[i] +
          omega_scale * r;
    }
    std::copy(out, out + n_rows, x_prev.begin());
    std::copy(x_next.begin(), x_next.end(), out);
    c_prev = c_cur;
    c_cur = c_next;
  }
}

template <typename I>
void csr_pcg_chebyshev_cpu_impl(const mx::array &data, const mx::array &indices,
                                const mx::array &indptr, const mx::array &b,
                                const mx::array &x0, const mx::array &m_data,
                                const mx::array &m_indices,
                                const mx::array &m_indptr, mx::array &x_out,
                                mx::array &info, mx::array &residual,
                                mx::array &iterations, int n_rows, int degree,
                                float lambda_min, float lambda_max, float rtol,
                                float atol, int maxiter, mx::Stream stream) {
  x_out.set_data(mx::allocator::malloc(x_out.nbytes()));
  info.set_data(mx::allocator::malloc(info.nbytes()));
  residual.set_data(mx::allocator::malloc(residual.nbytes()));
  iterations.set_data(mx::allocator::malloc(iterations.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(b);
  encoder.set_input_array(x0);
  encoder.set_input_array(m_data);
  encoder.set_input_array(m_indices);
  encoder.set_input_array(m_indptr);
  encoder.set_output_array(x_out);
  encoder.set_output_array(info);
  encoder.set_output_array(residual);
  encoder.set_output_array(iterations);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    b = mx::array::unsafe_weak_copy(b),
                    x0 = mx::array::unsafe_weak_copy(x0),
                    m_data = mx::array::unsafe_weak_copy(m_data),
                    m_indices = mx::array::unsafe_weak_copy(m_indices),
                    m_indptr = mx::array::unsafe_weak_copy(m_indptr),
                    x_out = mx::array::unsafe_weak_copy(x_out),
                    info = mx::array::unsafe_weak_copy(info),
                    residual = mx::array::unsafe_weak_copy(residual),
                    iterations = mx::array::unsafe_weak_copy(iterations),
                    n_rows, degree, lambda_min, lambda_max, rtol, atol,
                    maxiter]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    const auto *x0_ptr = x0.data<float>();
    const auto *m_data_ptr = m_data.data<float>();
    const auto *m_indices_ptr = m_indices.data<I>();
    const auto *m_indptr_ptr = m_indptr.data<I>();
    auto *x_ptr = x_out.data<float>();
    auto *info_ptr = info.data<int32_t>();
    auto *residual_ptr = residual.data<float>();
    auto *iterations_ptr = iterations.data<int32_t>();

    std::vector<float> r(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> z(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> p(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> ap(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> cheb_prev(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> cheb_next(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> cheb_ax(static_cast<size_t>(n_rows), 0.0f);
    std::copy(x0_ptr, x0_ptr + n_rows, x_ptr);

    csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, x_ptr, ap.data(), n_rows);
    double b_norm2 = 0.0;
    bool finite = true;
    for (int i = 0; i < n_rows; ++i) {
      const float ri = b_ptr[i] - ap[static_cast<size_t>(i)];
      r[static_cast<size_t>(i)] = ri;
      b_norm2 += static_cast<double>(b_ptr[i]) * static_cast<double>(b_ptr[i]);
      finite = finite && finite_float(ri) && finite_float(b_ptr[i]) &&
               finite_float(x_ptr[i]);
    }
    chebyshev_apply_vector(m_data_ptr, m_indices_ptr, m_indptr_ptr, r.data(),
                           z.data(), cheb_prev, cheb_next, cheb_ax, n_rows,
                           degree, lambda_min, lambda_max);
    std::copy(z.begin(), z.end(), p.begin());

    float true_rr = dot_float(r, r);
    float rho = dot_float(r, z);
    const float b_norm = std::sqrt(std::max(b_norm2, 0.0));
    const float tol = std::max(atol, rtol * b_norm);
    const float true_residual = std::sqrt(std::max(true_rr, 0.0f));

    if (!finite || !finite_float(true_rr) || !finite_float(rho)) {
      *info_ptr = -3;
      *residual_ptr = true_residual;
      *iterations_ptr = 0;
      return;
    }
    if (true_residual <= tol) {
      *info_ptr = 0;
      *residual_ptr = true_residual;
      *iterations_ptr = 0;
      return;
    }
    if (rho <= 0.0f) {
      *info_ptr = -2;
      *residual_ptr = true_residual;
      *iterations_ptr = 0;
      return;
    }

    const float eps = std::numeric_limits<float>::epsilon();
    int status = maxiter;
    int completed = 0;
    for (int it = 1; it <= maxiter; ++it) {
      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, p.data(), ap.data(),
                     n_rows);
      const float denom = dot_float(p, ap);
      const float p_norm2 = dot_float(p, p);
      const float ap_norm2 = dot_float(ap, ap);
      const float denom_scale = std::sqrt(std::max(p_norm2 * ap_norm2, 0.0f));
      const float denom_tol = eps * std::max(1.0f, denom_scale);
      if (!finite_float(denom) || !finite_float(denom_scale) ||
          std::abs(denom) <= denom_tol) {
        status = -1;
        completed = it - 1;
        break;
      }
      const float alpha = rho / denom;
      if (!finite_float(alpha)) {
        status = -3;
        completed = it - 1;
        break;
      }

      double true_rr_acc = 0.0;
      finite = true;
      for (int i = 0; i < n_rows; ++i) {
        x_ptr[i] += alpha * p[static_cast<size_t>(i)];
        const float ri =
            r[static_cast<size_t>(i)] - alpha * ap[static_cast<size_t>(i)];
        r[static_cast<size_t>(i)] = ri;
        true_rr_acc += static_cast<double>(ri) * static_cast<double>(ri);
        finite = finite && finite_float(x_ptr[i]) && finite_float(ri);
      }
      true_rr = static_cast<float>(true_rr_acc);
      const float r_norm = std::sqrt(std::max(true_rr, 0.0f));
      completed = it;
      if (!finite || !finite_float(true_rr) || !finite_float(r_norm)) {
        status = -3;
        break;
      }
      if (r_norm <= tol) {
        status = 0;
        break;
      }

      chebyshev_apply_vector(m_data_ptr, m_indices_ptr, m_indptr_ptr, r.data(),
                             z.data(), cheb_prev, cheb_next, cheb_ax, n_rows,
                             degree, lambda_min, lambda_max);
      const float rho_new = dot_float(r, z);
      if (!finite_float(rho_new) || rho_new <= 0.0f) {
        status = -2;
        break;
      }
      const float beta = rho_new / rho;
      if (!finite_float(beta)) {
        status = -3;
        break;
      }
      for (int i = 0; i < n_rows; ++i) {
        p[static_cast<size_t>(i)] =
            z[static_cast<size_t>(i)] + beta * p[static_cast<size_t>(i)];
      }
      rho = rho_new;
    }

    *info_ptr = status;
    *residual_ptr = std::sqrt(std::max(true_rr, 0.0f));
    *iterations_ptr = completed;
  });
}

} // namespace

void CSRPCGChebyshev::eval_cpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x0 = inputs[4];
  auto &m_data = inputs[5];
  auto &m_indices = inputs[6];
  auto &m_indptr = inputs[7];

  if (indices.dtype() == mx::int32) {
    csr_pcg_chebyshev_cpu_impl<int32_t>(
        data, indices, indptr, b, x0, m_data, m_indices, m_indptr, outputs[0],
        outputs[1], outputs[2], outputs[3], n_rows_, degree_, lambda_min_,
        lambda_max_, rtol_, atol_, maxiter_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_pcg_chebyshev_cpu_impl<int64_t>(
        data, indices, indptr, b, x0, m_data, m_indices, m_indptr, outputs[0],
        outputs[1], outputs[2], outputs[3], n_rows_, degree_, lambda_min_,
        lambda_max_, rtol_, atol_, maxiter_, stream());
    return;
  }
  throw std::runtime_error(
      "csr_pcg_chebyshev requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRPCGChebyshev::eval_gpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x0 = inputs[4];
  auto &m_data = inputs[5];
  auto &m_indices = inputs[6];
  auto &m_indptr = inputs[7];
  auto &x = outputs[0];
  auto &info = outputs[1];
  auto &residual = outputs[2];
  auto &iterations = outputs[3];

  x.set_data(mx::allocator::malloc(x.nbytes()));
  info.set_data(mx::allocator::malloc(info.nbytes()));
  residual.set_data(mx::allocator::malloc(residual.nbytes()));
  iterations.set_data(mx::allocator::malloc(iterations.nbytes()));
  mx::array work(
      mx::allocator::malloc(static_cast<size_t>(7 * n_rows_) * sizeof(float)),
      mx::Shape{7 * n_rows_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_pcg_chebyshev", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(b, 3);
  encoder.set_input_array(x0, 4);
  encoder.set_input_array(m_data, 5);
  encoder.set_input_array(m_indices, 6);
  encoder.set_input_array(m_indptr, 7);
  encoder.set_output_array(x, 8);
  encoder.set_output_array(info, 9);
  encoder.set_output_array(residual, 10);
  encoder.set_output_array(iterations, 11);
  encoder.set_output_array(work, 12);
  encoder.set_bytes(n_rows_, 13);
  encoder.set_bytes(n_cols_, 14);
  encoder.set_bytes(degree_, 15);
  encoder.set_bytes(lambda_min_, 16);
  encoder.set_bytes(lambda_max_, 17);
  encoder.set_bytes(maxiter_, 18);
  encoder.set_bytes(rtol_, 19);
  encoder.set_bytes(atol_, 20);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(work));
}
#else
void CSRPCGChebyshev::eval_gpu(const std::vector<mx::array> &,
                               std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_pcg_chebyshev has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array, mx::array> csr_pcg_chebyshev(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, const mx::array &x0, const mx::array &m_data,
    const mx::array &m_indices, const mx::array &m_indptr, int n_rows,
    int n_cols, int degree, float lambda_min, float lambda_max, float rtol,
    float atol, int maxiter, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_pcg_chebyshev requires a non-empty square matrix.");
  }
  if (maxiter < 0) {
    throw std::invalid_argument(
        "csr_pcg_chebyshev maxiter must be non-negative.");
  }
  validate_chebyshev_interval(degree, lambda_min, lambda_max);
  require_rank(data, 1, "csr_pcg_chebyshev data");
  require_rank(indices, 1, "csr_pcg_chebyshev indices");
  require_rank(indptr, 1, "csr_pcg_chebyshev indptr");
  require_rank(b, 1, "csr_pcg_chebyshev b");
  require_rank(x0, 1, "csr_pcg_chebyshev x0");
  require_rank(m_data, 1, "csr_pcg_chebyshev preconditioner data");
  require_rank(m_indices, 1, "csr_pcg_chebyshev preconditioner indices");
  require_rank(m_indptr, 1, "csr_pcg_chebyshev preconditioner indptr");
  require_linalg_float32(data, "csr_pcg_chebyshev data");
  require_linalg_float32(b, "csr_pcg_chebyshev b");
  require_linalg_float32(x0, "csr_pcg_chebyshev x0");
  require_linalg_float32(m_data, "csr_pcg_chebyshev preconditioner data");
  require_same_index_dtype(indices, indptr, "csr_pcg_chebyshev indices",
                           "csr_pcg_chebyshev indptr");
  require_same_index_dtype(m_indices, m_indptr,
                           "csr_pcg_chebyshev preconditioner indices",
                           "csr_pcg_chebyshev preconditioner indptr");
  if (indices.dtype() != m_indices.dtype()) {
    throw std::invalid_argument(
        "csr_pcg_chebyshev matrix and preconditioner indices must use the "
        "same dtype.");
  }
  require_size(indptr, n_rows + 1, "csr_pcg_chebyshev indptr");
  require_size(m_indptr, n_rows + 1, "csr_pcg_chebyshev preconditioner indptr");
  require_size(b, n_rows, "csr_pcg_chebyshev b");
  require_size(x0, n_cols, "csr_pcg_chebyshev x0");
  if (indices.size() != data.size() || m_indices.size() != m_data.size()) {
    throw std::invalid_argument(
        "csr_pcg_chebyshev data and indices lengths must match.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);
  auto m_data_contig = mx::contiguous(m_data, false, stream);
  auto m_indices_contig = mx::contiguous(m_indices, false, stream);
  auto m_indptr_contig = mx::contiguous(m_indptr, false, stream);

  auto primitive = std::make_shared<CSRPCGChebyshev>(
      stream, n_rows, n_cols, degree, lambda_min, lambda_max, rtol, atol,
      maxiter);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{n_rows}, mx::Shape{}, mx::Shape{}, mx::Shape{}},
      {mx::float32, mx::int32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, b_contig, x0_contig,
       m_data_contig, m_indices_contig, m_indptr_contig});
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

} // namespace mlx_sparse
