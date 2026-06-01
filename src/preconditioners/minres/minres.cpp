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

#include "preconditioners/minres/minres.h"

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

class CSRMINRESJacobi : public mx::Primitive {
public:
  CSRMINRESJacobi(mx::Stream stream, int n_rows, int n_cols, float rtol,
                  float atol, int maxiter, float shift)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), rtol_(rtol),
        atol_(atol), maxiter_(maxiter), shift_(shift) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRMINRESJacobi"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRMINRESJacobi &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           rtol_ == rhs.rtol_ && atol_ == rhs.atol_ &&
           maxiter_ == rhs.maxiter_ && shift_ == rhs.shift_;
  }

private:
  int n_rows_;
  int n_cols_;
  float rtol_;
  float atol_;
  int maxiter_;
  float shift_;
};

inline bool finite_float(float value) { return std::isfinite(value); }

template <typename I>
float shifted_true_residual(const float *data_ptr, const I *indices_ptr,
                            const I *indptr_ptr, const float *b_ptr,
                            const float *x_ptr, std::vector<float> &work,
                            int n_rows, float shift, bool &finite) {
  csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, x_ptr, work.data(), n_rows);
  double rr = 0.0;
  finite = true;
  for (int i = 0; i < n_rows; ++i) {
    const float ai = work[static_cast<size_t>(i)] - shift * x_ptr[i];
    const float ri = b_ptr[i] - ai;
    rr += static_cast<double>(ri) * static_cast<double>(ri);
    finite = finite && finite_float(ai) && finite_float(ri);
  }
  const float norm = std::sqrt(std::max(rr, 0.0));
  finite = finite && finite_float(norm);
  return norm;
}

template <typename I>
void csr_minres_jacobi_cpu_impl(const mx::array &data, const mx::array &indices,
                                const mx::array &indptr, const mx::array &b,
                                const mx::array &x0, const mx::array &inv_diag,
                                mx::array &x_out, mx::array &info,
                                mx::array &residual, mx::array &iterations,
                                int n_rows, float rtol, float atol, int maxiter,
                                float shift, mx::Stream stream) {
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
  encoder.set_input_array(inv_diag);
  encoder.set_output_array(x_out);
  encoder.set_output_array(info);
  encoder.set_output_array(residual);
  encoder.set_output_array(iterations);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    b = mx::array::unsafe_weak_copy(b),
                    x0 = mx::array::unsafe_weak_copy(x0),
                    inv_diag = mx::array::unsafe_weak_copy(inv_diag),
                    x_out = mx::array::unsafe_weak_copy(x_out),
                    info = mx::array::unsafe_weak_copy(info),
                    residual = mx::array::unsafe_weak_copy(residual),
                    iterations = mx::array::unsafe_weak_copy(iterations),
                    n_rows, rtol, atol, maxiter, shift]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    const auto *x0_ptr = x0.data<float>();
    const auto *inv_diag_ptr = inv_diag.data<float>();
    auto *x_ptr = x_out.data<float>();
    auto *info_ptr = info.data<int32_t>();
    auto *residual_ptr = residual.data<float>();
    auto *iterations_ptr = iterations.data<int32_t>();

    std::copy(x0_ptr, x0_ptr + n_rows, x_ptr);
    std::vector<float> r1(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> r2(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> y(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> v(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> w(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> w1(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> w2(static_cast<size_t>(n_rows), 0.0f);
    std::vector<float> av(static_cast<size_t>(n_rows), 0.0f);

    double b_norm2 = 0.0;
    bool finite = finite_float(shift);
    bool positive_preconditioner = true;
    for (int i = 0; i < n_rows; ++i) {
      b_norm2 += static_cast<double>(b_ptr[i]) * static_cast<double>(b_ptr[i]);
      finite = finite && finite_float(b_ptr[i]) && finite_float(x_ptr[i]) &&
               finite_float(inv_diag_ptr[i]);
      positive_preconditioner =
          positive_preconditioner && inv_diag_ptr[i] > 0.0f;
    }
    const float b_norm = std::sqrt(std::max(b_norm2, 0.0));
    const float tol = std::max(atol, rtol * b_norm);
    int status = maxiter > 0 ? maxiter : 1;
    int completed = 0;

    float true_residual =
        shifted_true_residual(data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr,
                              av, n_rows, shift, finite);
    if (!finite) {
      *info_ptr = -3;
      *residual_ptr = true_residual;
      *iterations_ptr = 0;
      return;
    }
    if (!positive_preconditioner) {
      *info_ptr = -2;
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
    if (maxiter == 0) {
      *info_ptr = status;
      *residual_ptr = true_residual;
      *iterations_ptr = 0;
      return;
    }

    for (int i = 0; i < n_rows; ++i) {
      const float ai = av[static_cast<size_t>(i)] - shift * x_ptr[i];
      const float ri = b_ptr[i] - ai;
      r2[static_cast<size_t>(i)] = ri;
      y[static_cast<size_t>(i)] = inv_diag_ptr[i] * ri;
    }

    double beta_inner = dot_float(r2, y);
    if (!std::isfinite(beta_inner) || beta_inner <= 0.0) {
      *info_ptr = -2;
      *residual_ptr = true_residual;
      *iterations_ptr = 0;
      return;
    }

    const double eps = std::numeric_limits<float>::epsilon();
    double beta = std::sqrt(beta_inner);
    double oldb = 0.0;
    double dbar = 0.0;
    double epsln = 0.0;
    double phibar = beta;
    double cs = -1.0;
    double sn = 0.0;

    for (int it = 1; it <= maxiter; ++it) {
      if (!std::isfinite(beta) || beta <= eps) {
        status = -2;
        break;
      }
      const double inv_beta = 1.0 / beta;
      for (int i = 0; i < n_rows; ++i) {
        v[static_cast<size_t>(i)] =
            static_cast<float>(inv_beta * y[static_cast<size_t>(i)]);
      }

      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, v.data(), av.data(),
                     n_rows);
      for (int i = 0; i < n_rows; ++i) {
        av[static_cast<size_t>(i)] -= shift * v[static_cast<size_t>(i)];
      }
      if (it >= 2) {
        if (!std::isfinite(oldb) || std::abs(oldb) <= eps) {
          status = -1;
          break;
        }
        const float prev_scale = static_cast<float>(beta / oldb);
        for (int i = 0; i < n_rows; ++i) {
          av[static_cast<size_t>(i)] -= prev_scale * r1[static_cast<size_t>(i)];
        }
      }

      const double alfa = dot_float(v, av);
      if (!std::isfinite(alfa)) {
        status = -3;
        break;
      }
      const float diag_scale = static_cast<float>(alfa / beta);
      for (int i = 0; i < n_rows; ++i) {
        av[static_cast<size_t>(i)] -= diag_scale * r2[static_cast<size_t>(i)];
      }
      r1.swap(r2);
      r2.swap(av);
      for (int i = 0; i < n_rows; ++i) {
        y[static_cast<size_t>(i)] =
            inv_diag_ptr[i] * r2[static_cast<size_t>(i)];
      }

      oldb = beta;
      beta_inner = dot_float(r2, y);
      if (!std::isfinite(beta_inner) || beta_inner < 0.0) {
        status = -2;
        break;
      }
      beta = std::sqrt(beta_inner);

      const double oldeps = epsln;
      const double delta = cs * dbar + sn * alfa;
      const double gbar = sn * dbar - cs * alfa;
      epsln = sn * beta;
      dbar = -cs * beta;
      double gamma = std::hypot(gbar, beta);
      if (!std::isfinite(gamma)) {
        status = -3;
        break;
      }
      gamma = std::max(gamma, eps);
      cs = gbar / gamma;
      sn = beta / gamma;
      const double phi = cs * phibar;
      phibar = sn * phibar;
      if (!std::isfinite(phi) || !std::isfinite(phibar)) {
        status = -3;
        break;
      }

      finite = true;
      for (int i = 0; i < n_rows; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const float prev_w1 = w2[idx];
        const float prev_w2 = w[idx];
        w1[idx] = prev_w1;
        w2[idx] = prev_w2;
        w[idx] = static_cast<float>(
            (static_cast<double>(v[idx]) - oldeps * prev_w1 - delta * prev_w2) /
            gamma);
        x_ptr[i] += static_cast<float>(phi * w[idx]);
        finite = finite && finite_float(w[idx]) && finite_float(x_ptr[i]);
      }
      completed = it;
      if (!finite) {
        status = -3;
        break;
      }

      true_residual =
          shifted_true_residual(data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr,
                                av, n_rows, shift, finite);
      if (!finite) {
        status = -3;
        break;
      }
      if (true_residual <= tol) {
        status = 0;
        break;
      }
      if (beta <= eps) {
        status = -1;
        break;
      }
    }

    *info_ptr = status;
    *residual_ptr = true_residual;
    *iterations_ptr = completed;
  });
}

} // namespace

void CSRMINRESJacobi::eval_cpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x0 = inputs[4];
  auto &inv_diag = inputs[5];

  if (indices.dtype() == mx::int32) {
    csr_minres_jacobi_cpu_impl<int32_t>(data, indices, indptr, b, x0, inv_diag,
                                        outputs[0], outputs[1], outputs[2],
                                        outputs[3], n_rows_, rtol_, atol_,
                                        maxiter_, shift_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_minres_jacobi_cpu_impl<int64_t>(data, indices, indptr, b, x0, inv_diag,
                                        outputs[0], outputs[1], outputs[2],
                                        outputs[3], n_rows_, rtol_, atol_,
                                        maxiter_, shift_, stream());
    return;
  }
  throw std::runtime_error(
      "csr_minres_jacobi requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRMINRESJacobi::eval_gpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x0 = inputs[4];
  auto &inv_diag = inputs[5];
  auto &x = outputs[0];
  auto &info = outputs[1];
  auto &residual = outputs[2];
  auto &iterations = outputs[3];

  x.set_data(mx::allocator::malloc(x.nbytes()));
  info.set_data(mx::allocator::malloc(info.nbytes()));
  residual.set_data(mx::allocator::malloc(residual.nbytes()));
  iterations.set_data(mx::allocator::malloc(iterations.nbytes()));
  mx::array work(
      mx::allocator::malloc(static_cast<size_t>(8 * n_rows_) * sizeof(float)),
      mx::Shape{8 * n_rows_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_minres_jacobi", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(b, 3);
  encoder.set_input_array(x0, 4);
  encoder.set_input_array(inv_diag, 5);
  encoder.set_output_array(x, 6);
  encoder.set_output_array(info, 7);
  encoder.set_output_array(residual, 8);
  encoder.set_output_array(iterations, 9);
  encoder.set_output_array(work, 10);
  encoder.set_bytes(n_rows_, 11);
  encoder.set_bytes(n_cols_, 12);
  encoder.set_bytes(maxiter_, 13);
  encoder.set_bytes(rtol_, 14);
  encoder.set_bytes(atol_, 15);
  encoder.set_bytes(shift_, 16);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(work));
}
#else
void CSRMINRESJacobi::eval_gpu(const std::vector<mx::array> &,
                               std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_minres_jacobi has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_minres_jacobi(const mx::array &data, const mx::array &indices,
                  const mx::array &indptr, const mx::array &b,
                  const mx::array &x0, const mx::array &inv_diag, int n_rows,
                  int n_cols, float rtol, float atol, int maxiter, float shift,
                  mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_minres_jacobi requires a non-empty square matrix.");
  }
  if (maxiter < 0) {
    throw std::invalid_argument(
        "csr_minres_jacobi maxiter must be non-negative.");
  }
  if (!std::isfinite(rtol) || rtol < 0.0f || !std::isfinite(atol) ||
      atol < 0.0f || !std::isfinite(shift)) {
    throw std::invalid_argument(
        "csr_minres_jacobi requires finite non-negative tolerances and finite "
        "shift.");
  }
  require_rank(data, 1, "csr_minres_jacobi data");
  require_rank(indices, 1, "csr_minres_jacobi indices");
  require_rank(indptr, 1, "csr_minres_jacobi indptr");
  require_rank(b, 1, "csr_minres_jacobi b");
  require_rank(x0, 1, "csr_minres_jacobi x0");
  require_rank(inv_diag, 1, "csr_minres_jacobi inv_diag");
  require_linalg_float32(data, "csr_minres_jacobi data");
  require_linalg_float32(b, "csr_minres_jacobi b");
  require_linalg_float32(x0, "csr_minres_jacobi x0");
  require_linalg_float32(inv_diag, "csr_minres_jacobi inv_diag");
  require_same_index_dtype(indices, indptr, "csr_minres_jacobi indices",
                           "csr_minres_jacobi indptr");
  require_size(indptr, n_rows + 1, "csr_minres_jacobi indptr");
  require_size(b, n_rows, "csr_minres_jacobi b");
  require_size(x0, n_cols, "csr_minres_jacobi x0");
  require_size(inv_diag, n_rows, "csr_minres_jacobi inv_diag");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_minres_jacobi data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);
  auto inv_diag_contig = mx::contiguous(inv_diag, false, stream);

  auto primitive = std::make_shared<CSRMINRESJacobi>(
      stream, n_rows, n_cols, rtol, atol, maxiter, shift);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{n_rows}, mx::Shape{}, mx::Shape{}, mx::Shape{}},
      {mx::float32, mx::int32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, b_contig, x0_contig,
       inv_diag_contig});
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

} // namespace mlx_sparse
