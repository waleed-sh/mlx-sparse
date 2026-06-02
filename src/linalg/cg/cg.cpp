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

#include "linalg/cg/cg.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
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

class CSRCG : public mx::Primitive {
public:
  CSRCG(mx::Stream stream, int n_rows, int n_cols, float rtol, float atol,
        int maxiter)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), rtol_(rtol),
        atol_(atol), maxiter_(maxiter) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRCG"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRCG &>(other);
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
void csr_cg_cpu_impl(const mx::array &data, const mx::array &indices,
                     const mx::array &indptr, const mx::array &b,
                     const mx::array &x0, mx::array &x_out, mx::array &info,
                     mx::array &residual, mx::array &iterations, int n_rows,
                     float rtol, float atol, int maxiter, mx::Stream stream) {
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

    std::vector<float> r(static_cast<size_t>(n_rows));
    std::vector<float> p(static_cast<size_t>(n_rows));
    std::vector<float> ap(static_cast<size_t>(n_rows));
    std::copy(x0_ptr, x0_ptr + n_rows, x_ptr);

    csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, x_ptr, ap.data(), n_rows);
    double b_norm2 = 0.0;
    double rr = 0.0;
    bool finite = true;
    for (int i = 0; i < n_rows; ++i) {
      r[i] = b_ptr[i] - ap[i];
      p[i] = r[i];
      b_norm2 += static_cast<double>(b_ptr[i]) * static_cast<double>(b_ptr[i]);
      rr += static_cast<double>(r[i]) * static_cast<double>(r[i]);
      finite = finite && finite_float(r[i]) && finite_float(p[i]) &&
               finite_float(ap[i]) && finite_float(b_ptr[i]) &&
               finite_float(x_ptr[i]);
    }

    const float residual_norm =
        static_cast<float>(std::sqrt(std::max(rr, 0.0)));
    const float b_norm = std::sqrt(std::max(b_norm2, 0.0));
    const float tol = std::max(atol, rtol * b_norm);
    const float eps = std::numeric_limits<float>::epsilon();

    if (!finite || !std::isfinite(rr) || !finite_float(residual_norm) ||
        !finite_float(b_norm)) {
      *info_ptr = -3;
      *residual_ptr = residual_norm;
      *iterations_ptr = 0;
      return;
    }
    if (residual_norm <= tol) {
      *info_ptr = 0;
      *residual_ptr = residual_norm;
      *iterations_ptr = 0;
      return;
    }

    int status = maxiter > 0 ? maxiter : 1;
    int completed = 0;
    for (int it = 1; it <= maxiter; ++it) {
      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, p.data(), ap.data(),
                     n_rows);
      double denom = 0.0;
      finite = true;
      for (int i = 0; i < n_rows; ++i) {
        denom += static_cast<double>(p[i]) * static_cast<double>(ap[i]);
        finite = finite && finite_float(p[i]) && finite_float(ap[i]);
      }
      if (!finite || !std::isfinite(denom)) {
        status = -3;
        completed = it - 1;
        break;
      }
      if (std::abs(denom) <= eps) {
        double p_norm2 = 0.0;
        double ap_norm2 = 0.0;
        for (int i = 0; i < n_rows; ++i) {
          p_norm2 += static_cast<double>(p[i]) * static_cast<double>(p[i]);
          ap_norm2 += static_cast<double>(ap[i]) * static_cast<double>(ap[i]);
        }
        const double denom_scale = std::sqrt(std::max(p_norm2 * ap_norm2, 0.0));
        const double denom_tol =
            std::min(static_cast<double>(eps), static_cast<double>(eps) *
                                                   static_cast<double>(eps) *
                                                   std::max(1.0, denom_scale));
        if (!std::isfinite(denom_scale)) {
          status = -3;
          completed = it - 1;
          break;
        }
        if (std::abs(denom) <= denom_tol) {
          status = -1;
          completed = it - 1;
          break;
        }
      }
      const double alpha = rr / denom;
      if (!std::isfinite(alpha)) {
        status = -3;
        completed = it - 1;
        break;
      }
      finite = true;
      double rr_new = 0.0;
      for (int i = 0; i < n_rows; ++i) {
        const double x_candidate =
            static_cast<double>(x_ptr[i]) + alpha * static_cast<double>(p[i]);
        const double r_candidate =
            static_cast<double>(r[i]) - alpha * static_cast<double>(ap[i]);
        const float x_candidate_f = static_cast<float>(x_candidate);
        const float r_candidate_f = static_cast<float>(r_candidate);
        if (!std::isfinite(x_candidate) || !std::isfinite(r_candidate) ||
            !finite_float(x_candidate_f) || !finite_float(r_candidate_f)) {
          finite = false;
          continue;
        }
        x_ptr[i] = x_candidate_f;
        r[i] = r_candidate_f;
        finite = finite && finite_float(x_ptr[i]) && finite_float(r[i]);
        rr_new += static_cast<double>(r[i]) * static_cast<double>(r[i]);
      }
      const float r_norm = static_cast<float>(std::sqrt(std::max(rr_new, 0.0)));
      completed = it;
      if (!finite || !std::isfinite(rr_new) || !finite_float(r_norm)) {
        status = -3;
        break;
      }
      if (r_norm <= tol) {
        status = 0;
        rr = rr_new;
        break;
      }
      const double beta = rr_new / rr;
      if (!std::isfinite(beta)) {
        status = -3;
        break;
      }
      for (int i = 0; i < n_rows; ++i) {
        const double p_candidate =
            static_cast<double>(r[i]) + beta * static_cast<double>(p[i]);
        const float p_candidate_f = static_cast<float>(p_candidate);
        if (!std::isfinite(p_candidate) || !finite_float(p_candidate_f)) {
          status = -3;
          break;
        }
        p[i] = p_candidate_f;
      }
      if (status < 0) {
        break;
      }
      rr = rr_new;
    }

    *info_ptr = status;
    *residual_ptr = static_cast<float>(std::sqrt(std::max(rr, 0.0)));
    *iterations_ptr = completed;
  });
}

} // namespace

void CSRCG::eval_cpu(const std::vector<mx::array> &inputs,
                     std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x0 = inputs[4];

  if (indices.dtype() == mx::int32) {
    csr_cg_cpu_impl<int32_t>(data, indices, indptr, b, x0, outputs[0],
                             outputs[1], outputs[2], outputs[3], n_rows_, rtol_,
                             atol_, maxiter_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_cg_cpu_impl<int64_t>(data, indices, indptr, b, x0, outputs[0],
                             outputs[1], outputs[2], outputs[3], n_rows_, rtol_,
                             atol_, maxiter_, stream());
    return;
  }
  throw std::runtime_error("csr_cg requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRCG::eval_gpu(const std::vector<mx::array> &inputs,
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
      mx::allocator::malloc(static_cast<size_t>(3 * n_rows_) * sizeof(float)),
      mx::Shape{3 * n_rows_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_cg", data.dtype(), indices.dtype());
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
void CSRCG::eval_gpu(const std::vector<mx::array> &, std::vector<mx::array> &) {
  throw std::runtime_error("csr_cg has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_cg(const mx::array &data, const mx::array &indices, const mx::array &indptr,
       const mx::array &b, const mx::array &x0, int n_rows, int n_cols,
       float rtol, float atol, int maxiter, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_cg requires a non-empty square matrix.");
  }
  if (maxiter < 0) {
    throw std::invalid_argument("csr_cg maxiter must be non-negative.");
  }
  require_rank(data, 1, "csr_cg data");
  require_rank(indices, 1, "csr_cg indices");
  require_rank(indptr, 1, "csr_cg indptr");
  require_rank(b, 1, "csr_cg b");
  require_rank(x0, 1, "csr_cg x0");
  require_linalg_float32(data, "csr_cg data");
  require_linalg_float32(b, "csr_cg b");
  require_linalg_float32(x0, "csr_cg x0");
  require_same_index_dtype(indices, indptr, "csr_cg indices", "csr_cg indptr");
  require_size(indptr, n_rows + 1, "csr_cg indptr");
  require_size(b, n_rows, "csr_cg b");
  require_size(x0, n_cols, "csr_cg x0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_cg data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);

  auto primitive =
      std::make_shared<CSRCG>(stream, n_rows, n_cols, rtol, atol, maxiter);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{n_rows}, mx::Shape{}, mx::Shape{}, mx::Shape{}},
      {mx::float32, mx::int32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, b_contig, x0_contig});
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

} // namespace mlx_sparse
