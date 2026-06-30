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

#include "linalg/bicgstab/bicgstab.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

#include "linalg/common/common.h"

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

inline bool finite_float(float value) { return std::isfinite(value); }

inline bool near_zero_dot(double value, double lhs_norm2, double rhs_norm2) {
  const double scale = std::sqrt(std::max(lhs_norm2 * rhs_norm2, 0.0));
  const double tol =
      static_cast<double>(std::numeric_limits<float>::epsilon()) *
      std::max(1.0, scale);
  return !std::isfinite(scale) || std::abs(value) <= tol;
}

class CSRBiCGSTAB : public mx::Primitive {
public:
  CSRBiCGSTAB(mx::Stream stream, int n_rows, int n_cols, float rtol, float atol,
              int maxiter)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), rtol_(rtol),
        atol_(atol), maxiter_(maxiter) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRBiCGSTAB"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRBiCGSTAB &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           rtol_ == rhs.rtol_ && atol_ == rhs.atol_ && maxiter_ == rhs.maxiter_;
  }

private:
  int n_rows_;
  int n_cols_;
  float rtol_;
  float atol_;
  int maxiter_;
};

template <typename I>
void csr_bicgstab_cpu_impl(const mx::array &data, const mx::array &indices,
                           const mx::array &indptr, const mx::array &b,
                           const mx::array &x0, mx::array &x_out,
                           mx::array &info, mx::array &residual,
                           mx::array &iterations, int n_rows, float rtol,
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
  encoder.set_output_array(x_out);
  encoder.set_output_array(info);
  encoder.set_output_array(residual);
  encoder.set_output_array(iterations);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    b = mx::array::unsafe_weak_copy(b),
                    x0 = mx::array::unsafe_weak_copy(x0),
                    x_out = mx::array::unsafe_weak_copy(x_out),
                    info = mx::array::unsafe_weak_copy(info),
                    residual = mx::array::unsafe_weak_copy(residual),
                    iterations = mx::array::unsafe_weak_copy(iterations),
                    n_rows, rtol, atol, maxiter]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    const auto *x0_ptr = x0.data<float>();
    auto *x_ptr = x_out.data<float>();
    auto *info_ptr = info.data<int32_t>();
    auto *residual_ptr = residual.data<float>();
    auto *iterations_ptr = iterations.data<int32_t>();

    std::vector<float> r(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> r_hat(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> p(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> v(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> s_vec(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> t(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> ax(static_cast<size_t>(n_rows), 0.0f);
    std::copy(x0_ptr, x0_ptr + n_rows, x_ptr);

    double b_norm2 = 0.0;
    bool setup_finite = true;
    for (int i = 0; i < n_rows; ++i) {
      b_norm2 += static_cast<double>(b_ptr[i]) * static_cast<double>(b_ptr[i]);
      setup_finite =
          setup_finite && finite_float(b_ptr[i]) && finite_float(x_ptr[i]);
    }
    const float b_norm = std::sqrt(std::max(b_norm2, 0.0));
    const float tolerance = std::max(atol, rtol * b_norm);

    auto true_residual_into = [&](std::vector<float> &dst) {
      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, x_ptr, ax.data(),
                     n_rows);
      double rr = 0.0;
      bool finite = true;
      for (int i = 0; i < n_rows; ++i) {
        const float ri = b_ptr[i] - ax[static_cast<size_t>(i)];
        dst[static_cast<size_t>(i)] = ri;
        rr += static_cast<double>(ri) * static_cast<double>(ri);
        finite = finite && finite_float(ri) &&
                 finite_float(ax[static_cast<size_t>(i)]) &&
                 finite_float(x_ptr[i]);
      }
      const float norm = std::sqrt(std::max(rr, 0.0));
      return std::pair<float, bool>{norm, finite && std::isfinite(rr) &&
                                              finite_float(norm)};
    };

    auto [residual_norm, residual_finite] = true_residual_into(r);
    std::copy(r.begin(), r.end(), r_hat.begin());

    if (!setup_finite || !finite_float(b_norm) || !finite_float(tolerance) ||
        !residual_finite) {
      *info_ptr = -3;
      *residual_ptr = residual_norm;
      *iterations_ptr = 0;
      return;
    }
    if (residual_norm <= tolerance) {
      *info_ptr = 0;
      *residual_ptr = residual_norm;
      *iterations_ptr = 0;
      return;
    }

    int status = maxiter > 0 ? maxiter : 1;
    int completed = 0;
    double rho_prev = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    for (int it = 1; it <= maxiter; ++it) {
      const double rho = dot_double(r_hat, r);
      const double r_hat_norm2 = dot_double(r_hat, r_hat);
      const double r_norm2 = dot_double(r, r);
      if (!std::isfinite(rho) || !std::isfinite(r_hat_norm2) ||
          !std::isfinite(r_norm2)) {
        status = -3;
        completed = it - 1;
        break;
      }
      if (near_zero_dot(rho, r_hat_norm2, r_norm2)) {
        status = -1;
        completed = it - 1;
        break;
      }

      if (it == 1) {
        std::copy(r.begin(), r.end(), p.begin());
      } else {
        if (std::abs(omega) <= std::numeric_limits<float>::epsilon()) {
          status = -1;
          completed = it - 1;
          break;
        }
        const double beta = (rho / rho_prev) * (alpha / omega);
        if (!std::isfinite(beta)) {
          status = -3;
          completed = it - 1;
          break;
        }
        bool finite = true;
        for (int i = 0; i < n_rows; ++i) {
          const double pi =
              static_cast<double>(r[static_cast<size_t>(i)]) +
              beta * (static_cast<double>(p[static_cast<size_t>(i)]) -
                      omega * static_cast<double>(v[static_cast<size_t>(i)]));
          p[static_cast<size_t>(i)] = static_cast<float>(pi);
          finite = finite && std::isfinite(pi) &&
                   finite_float(p[static_cast<size_t>(i)]);
        }
        if (!finite) {
          status = -3;
          completed = it - 1;
          break;
        }
      }

      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, p.data(), v.data(),
                     n_rows);
      const double alpha_den = dot_double(r_hat, v);
      const double v_norm2 = dot_double(v, v);
      if (!std::isfinite(alpha_den) || !std::isfinite(v_norm2)) {
        status = -3;
        completed = it - 1;
        break;
      }
      if (near_zero_dot(alpha_den, r_hat_norm2, v_norm2)) {
        status = -1;
        completed = it - 1;
        break;
      }
      alpha = rho / alpha_den;
      if (!std::isfinite(alpha)) {
        status = -3;
        completed = it - 1;
        break;
      }

      double s_norm2 = 0.0;
      bool finite = true;
      for (int i = 0; i < n_rows; ++i) {
        const double si =
            static_cast<double>(r[static_cast<size_t>(i)]) -
            alpha * static_cast<double>(v[static_cast<size_t>(i)]);
        const double xi =
            static_cast<double>(x_ptr[i]) +
            alpha * static_cast<double>(p[static_cast<size_t>(i)]);
        s_vec[static_cast<size_t>(i)] = static_cast<float>(si);
        x_ptr[i] = static_cast<float>(xi);
        s_norm2 += si * si;
        finite = finite && std::isfinite(si) && std::isfinite(xi) &&
                 finite_float(s_vec[static_cast<size_t>(i)]) &&
                 finite_float(x_ptr[i]);
      }
      completed = it;
      float s_norm = std::sqrt(std::max(s_norm2, 0.0));
      if (!finite || !std::isfinite(s_norm2) || !finite_float(s_norm)) {
        status = -3;
        break;
      }
      if (s_norm <= tolerance) {
        auto [true_norm, true_finite] = true_residual_into(ax);
        if (!true_finite) {
          status = -3;
          residual_norm = true_norm;
          break;
        }
        if (true_norm <= tolerance) {
          status = 0;
          residual_norm = true_norm;
          break;
        }
      }

      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, s_vec.data(), t.data(),
                     n_rows);
      const double omega_num = dot_double(t, s_vec);
      const double omega_den = dot_double(t, t);
      if (!std::isfinite(omega_num) || !std::isfinite(omega_den)) {
        status = -3;
        break;
      }
      if (near_zero_dot(omega_den, omega_den, 1.0)) {
        status = -1;
        break;
      }
      omega = omega_num / omega_den;
      if (!std::isfinite(omega)) {
        status = -3;
        break;
      }

      double r_norm2_new = 0.0;
      finite = true;
      for (int i = 0; i < n_rows; ++i) {
        const double xi =
            static_cast<double>(x_ptr[i]) +
            omega * static_cast<double>(s_vec[static_cast<size_t>(i)]);
        const double ri =
            static_cast<double>(s_vec[static_cast<size_t>(i)]) -
            omega * static_cast<double>(t[static_cast<size_t>(i)]);
        x_ptr[i] = static_cast<float>(xi);
        r[static_cast<size_t>(i)] = static_cast<float>(ri);
        r_norm2_new += ri * ri;
        finite = finite && std::isfinite(xi) && std::isfinite(ri) &&
                 finite_float(x_ptr[i]) &&
                 finite_float(r[static_cast<size_t>(i)]);
      }
      residual_norm = std::sqrt(std::max(r_norm2_new, 0.0));
      if (!finite || !std::isfinite(r_norm2_new) ||
          !finite_float(residual_norm)) {
        status = -3;
        break;
      }
      if (residual_norm <= tolerance) {
        auto [true_norm, true_finite] = true_residual_into(r);
        if (!true_finite) {
          status = -3;
          residual_norm = true_norm;
          break;
        }
        residual_norm = true_norm;
        if (true_norm <= tolerance) {
          status = 0;
          break;
        }
      }
      if (std::abs(omega) <= std::numeric_limits<float>::epsilon()) {
        status = -1;
        break;
      }
      rho_prev = rho;
    }

    if (status > 0 || status == -1) {
      auto [true_norm, true_finite] = true_residual_into(r);
      residual_norm = true_norm;
      if (!true_finite) {
        status = -3;
      } else if (true_norm <= tolerance) {
        status = 0;
      }
    }

    *info_ptr = status;
    *residual_ptr = residual_norm;
    *iterations_ptr = completed;
  });
}

} // namespace

void CSRBiCGSTAB::eval_cpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x0 = inputs[4];

  if (indices.dtype() == mx::int32) {
    csr_bicgstab_cpu_impl<int32_t>(data, indices, indptr, b, x0, outputs[0],
                                   outputs[1], outputs[2], outputs[3], n_rows_,
                                   rtol_, atol_, maxiter_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_bicgstab_cpu_impl<int64_t>(data, indices, indptr, b, x0, outputs[0],
                                   outputs[1], outputs[2], outputs[3], n_rows_,
                                   rtol_, atol_, maxiter_, stream());
    return;
  }
  throw std::runtime_error("csr_bicgstab requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRBiCGSTAB::eval_gpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x0 = inputs[4];
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
      sparse_kernel_name("csr_bicgstab", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(b, 3);
  encoder.set_input_array(x0, 4);
  encoder.set_output_array(x, 5);
  encoder.set_output_array(info, 6);
  encoder.set_output_array(residual, 7);
  encoder.set_output_array(iterations, 8);
  encoder.set_output_array(work, 9);
  encoder.set_bytes(n_rows_, 10);
  encoder.set_bytes(n_cols_, 11);
  encoder.set_bytes(maxiter_, 12);
  encoder.set_bytes(rtol_, 13);
  encoder.set_bytes(atol_, 14);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(work));
}
#else
void CSRBiCGSTAB::eval_gpu(const std::vector<mx::array> &,
                           std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_bicgstab has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab(const mx::array &data, const mx::array &indices,
             const mx::array &indptr, const mx::array &b, const mx::array &x0,
             int n_rows, int n_cols, float rtol, float atol, int maxiter,
             mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_bicgstab requires a non-empty square matrix.");
  }
  if (maxiter < 0) {
    throw std::invalid_argument("csr_bicgstab maxiter must be non-negative.");
  }
  require_rank(data, 1, "csr_bicgstab data");
  require_rank(indices, 1, "csr_bicgstab indices");
  require_rank(indptr, 1, "csr_bicgstab indptr");
  require_rank(b, 1, "csr_bicgstab b");
  require_rank(x0, 1, "csr_bicgstab x0");
  require_linalg_float32(data, "csr_bicgstab data");
  require_linalg_float32(b, "csr_bicgstab b");
  require_linalg_float32(x0, "csr_bicgstab x0");
  require_same_index_dtype(indices, indptr, "csr_bicgstab indices",
                           "csr_bicgstab indptr");
  require_size(indptr, n_rows + 1, "csr_bicgstab indptr");
  require_size(b, n_rows, "csr_bicgstab b");
  require_size(x0, n_cols, "csr_bicgstab x0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_bicgstab data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);

  auto primitive = std::make_shared<CSRBiCGSTAB>(stream, n_rows, n_cols, rtol,
                                                 atol, maxiter);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{n_rows}, mx::Shape{}, mx::Shape{}, mx::Shape{}},
      {mx::float32, mx::int32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, b_contig, x0_contig});
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

} // namespace mlx_sparse
