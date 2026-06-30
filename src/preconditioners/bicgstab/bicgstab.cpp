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

#include "preconditioners/bicgstab/bicgstab.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
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
#include "preconditioners/exact/exact.h"
#include "preconditioners/ilu0/ilu0.h"

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
#include "linalg/accelerate/solve/solve.h"
#endif

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

inline bool finite_float(float value) { return std::isfinite(value); }

bool finite_vector(const std::vector<float> &values) {
  for (float value : values) {
    if (!finite_float(value)) {
      return false;
    }
  }
  return true;
}

inline bool near_zero_dot(double value, double lhs_norm2, double rhs_norm2) {
  const double scale = std::sqrt(std::max(lhs_norm2 * rhs_norm2, 0.0));
  const double tol =
      static_cast<double>(std::numeric_limits<float>::epsilon()) *
      std::max(1.0, scale);
  return !std::isfinite(scale) || std::abs(value) <= tol;
}

mx::array vector_to_mx_array(const std::vector<float> &values) {
  return mx::array(values.begin(), mx::Shape{static_cast<int>(values.size())},
                   mx::float32);
}

std::vector<float> host_vector(mx::array values, int expected_size,
                               const char *context) {
  mx::eval(values);
  if (values.ndim() != 1 || values.shape(0) != expected_size) {
    throw std::runtime_error(std::string(context) +
                             " produced an incompatible vector shape.");
  }
  if (values.dtype() != mx::float32) {
    throw std::runtime_error(std::string(context) +
                             " produced a non-float32 vector.");
  }
  const auto *ptr = values.data<float>();
  return std::vector<float>(ptr, ptr + expected_size);
}

void require_bicgstab_base(const char *name, const mx::array &data,
                           const mx::array &indices, const mx::array &indptr,
                           const mx::array &b, const mx::array &x0, int n_rows,
                           int n_cols, int maxiter) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(std::string(name) +
                                " requires a non-empty square matrix.");
  }
  if (maxiter < 0) {
    throw std::invalid_argument(std::string(name) +
                                " maxiter must be non-negative.");
  }
  require_rank(data, 1, (std::string(name) + " data").c_str());
  require_rank(indices, 1, (std::string(name) + " indices").c_str());
  require_rank(indptr, 1, (std::string(name) + " indptr").c_str());
  require_rank(b, 1, (std::string(name) + " b").c_str());
  require_rank(x0, 1, (std::string(name) + " x0").c_str());
  require_linalg_float32(data, (std::string(name) + " data").c_str());
  require_linalg_float32(b, (std::string(name) + " b").c_str());
  require_linalg_float32(x0, (std::string(name) + " x0").c_str());
  require_same_index_dtype(indices, indptr,
                           (std::string(name) + " indices").c_str(),
                           (std::string(name) + " indptr").c_str());
  require_size(indptr, n_rows + 1, (std::string(name) + " indptr").c_str());
  require_size(b, n_rows, (std::string(name) + " b").c_str());
  require_size(x0, n_cols, (std::string(name) + " x0").c_str());
  if (indices.size() != data.size()) {
    throw std::invalid_argument(std::string(name) +
                                " data and indices must have equal length.");
  }
}

class CSRBiCGSTABJacobi : public mx::Primitive {
public:
  CSRBiCGSTABJacobi(mx::Stream stream, int n_rows, int n_cols, float rtol,
                    float atol, int maxiter)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), rtol_(rtol),
        atol_(atol), maxiter_(maxiter) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRBiCGSTABJacobi"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRBiCGSTABJacobi &>(other);
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

template <typename Matvec, typename ApplyPreconditioner>
std::tuple<std::vector<float>, int, float, int>
left_preconditioned_bicgstab_host(int n_rows, const std::vector<float> &rhs,
                                  const std::vector<float> &x0, float rtol,
                                  float atol, int maxiter, Matvec &&matvec,
                                  ApplyPreconditioner &&apply_preconditioner) {
  std::vector<float> x = x0;
  std::vector<float> r(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> r_hat(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> p(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> p_hat(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> v(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> s(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> s_hat(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> t(static_cast<size_t>(n_rows), 0.0f);

  const float b_norm = norm_float(rhs);
  const float tolerance = std::max(atol, rtol * b_norm);
  int status = maxiter > 0 ? maxiter : 1;
  int completed = 0;
  float residual_norm = std::numeric_limits<float>::infinity();

  if (!finite_vector(rhs) || !finite_vector(x) || !finite_float(b_norm) ||
      !finite_float(tolerance)) {
    return {x, -3, residual_norm, 0};
  }

  auto true_residual_into = [&](std::vector<float> &dst) {
    auto ax = matvec(x);
    if (static_cast<int>(ax.size()) != n_rows || !finite_vector(ax)) {
      throw std::runtime_error("bicgstab matvec produced invalid output.");
    }
    double rr = 0.0;
    bool finite = true;
    for (int i = 0; i < n_rows; ++i) {
      const float ri = rhs[static_cast<size_t>(i)] - ax[static_cast<size_t>(i)];
      dst[static_cast<size_t>(i)] = ri;
      rr += static_cast<double>(ri) * static_cast<double>(ri);
      finite = finite && finite_float(ri);
    }
    const float norm = std::sqrt(std::max(rr, 0.0));
    if (!finite || !std::isfinite(rr) || !finite_float(norm)) {
      throw std::runtime_error("bicgstab residual became non-finite.");
    }
    return norm;
  };

  try {
    residual_norm = true_residual_into(r);
  } catch (const std::exception &) {
    return {x, -3, residual_norm, 0};
  }
  std::copy(r.begin(), r.end(), r_hat.begin());
  if (residual_norm <= tolerance) {
    return {x, 0, residual_norm, 0};
  }

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
        const size_t idx = static_cast<size_t>(i);
        const double pi = static_cast<double>(r[idx]) +
                          beta * (static_cast<double>(p[idx]) -
                                  omega * static_cast<double>(v[idx]));
        p[idx] = static_cast<float>(pi);
        finite = finite && std::isfinite(pi) && finite_float(p[idx]);
      }
      if (!finite) {
        status = -3;
        completed = it - 1;
        break;
      }
    }

    try {
      p_hat = apply_preconditioner(p);
    } catch (const std::exception &) {
      status = -3;
      completed = it - 1;
      break;
    }
    if (static_cast<int>(p_hat.size()) != n_rows || !finite_vector(p_hat)) {
      status = -3;
      completed = it - 1;
      break;
    }

    try {
      v = matvec(p_hat);
    } catch (const std::exception &) {
      status = -1;
      completed = it - 1;
      break;
    }
    if (static_cast<int>(v.size()) != n_rows || !finite_vector(v)) {
      status = -3;
      completed = it - 1;
      break;
    }

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

    bool finite = true;
    double s_norm2 = 0.0;
    for (int i = 0; i < n_rows; ++i) {
      const size_t idx = static_cast<size_t>(i);
      const double si =
          static_cast<double>(r[idx]) - alpha * static_cast<double>(v[idx]);
      const double xi =
          static_cast<double>(x[idx]) + alpha * static_cast<double>(p_hat[idx]);
      s[idx] = static_cast<float>(si);
      x[idx] = static_cast<float>(xi);
      s_norm2 += si * si;
      finite = finite && std::isfinite(si) && std::isfinite(xi) &&
               finite_float(s[idx]) && finite_float(x[idx]);
    }
    completed = it;
    residual_norm = std::sqrt(std::max(s_norm2, 0.0));
    if (!finite || !std::isfinite(s_norm2) || !finite_float(residual_norm)) {
      status = -3;
      break;
    }
    if (residual_norm <= tolerance) {
      try {
        const float true_norm = true_residual_into(t);
        if (true_norm <= tolerance) {
          status = 0;
          residual_norm = true_norm;
          break;
        }
      } catch (const std::exception &) {
        status = -3;
        break;
      }
    }

    try {
      s_hat = apply_preconditioner(s);
    } catch (const std::exception &) {
      status = -3;
      break;
    }
    if (static_cast<int>(s_hat.size()) != n_rows || !finite_vector(s_hat)) {
      status = -3;
      break;
    }

    try {
      t = matvec(s_hat);
    } catch (const std::exception &) {
      status = -1;
      break;
    }
    if (static_cast<int>(t.size()) != n_rows || !finite_vector(t)) {
      status = -3;
      break;
    }

    const double omega_num = dot_double(t, s);
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
      const size_t idx = static_cast<size_t>(i);
      const double xi =
          static_cast<double>(x[idx]) + omega * static_cast<double>(s_hat[idx]);
      const double ri =
          static_cast<double>(s[idx]) - omega * static_cast<double>(t[idx]);
      x[idx] = static_cast<float>(xi);
      r[idx] = static_cast<float>(ri);
      r_norm2_new += ri * ri;
      finite = finite && std::isfinite(xi) && std::isfinite(ri) &&
               finite_float(x[idx]) && finite_float(r[idx]);
    }
    residual_norm = std::sqrt(std::max(r_norm2_new, 0.0));
    if (!finite || !std::isfinite(r_norm2_new) ||
        !finite_float(residual_norm)) {
      status = -3;
      break;
    }
    if (residual_norm <= tolerance) {
      try {
        const float true_norm = true_residual_into(r);
        residual_norm = true_norm;
        if (true_norm <= tolerance) {
          status = 0;
          break;
        }
      } catch (const std::exception &) {
        status = -3;
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
    try {
      const float true_norm = true_residual_into(r);
      residual_norm = true_norm;
      if (true_norm <= tolerance) {
        status = 0;
      }
    } catch (const std::exception &) {
      status = -3;
    }
  }
  return {x, status, residual_norm, completed};
}

template <typename I>
void csr_bicgstab_jacobi_cpu_impl(const mx::array &data,
                                  const mx::array &indices,
                                  const mx::array &indptr, const mx::array &b,
                                  const mx::array &x0,
                                  const mx::array &inv_diag, mx::array &x_out,
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
                    n_rows, rtol, atol, maxiter]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    const auto *x0_ptr = x0.data<float>();
    const auto *inv_diag_ptr = inv_diag.data<float>();

    std::vector<float> rhs(b_ptr, b_ptr + n_rows);
    std::vector<float> guess(x0_ptr, x0_ptr + n_rows);
    auto matvec = [&](const std::vector<float> &x) {
      return host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, x, n_rows);
    };
    auto apply = [&](const std::vector<float> &x) {
      std::vector<float> out(static_cast<size_t>(n_rows));
      for (int i = 0; i < n_rows; ++i) {
        out[static_cast<size_t>(i)] =
            inv_diag_ptr[i] * x[static_cast<size_t>(i)];
      }
      return out;
    };

    auto [x, status, residual_norm, completed] =
        left_preconditioned_bicgstab_host(n_rows, rhs, guess, rtol, atol,
                                          maxiter, matvec, apply);
    std::copy(x.begin(), x.end(), x_out.data<float>());
    *info.data<int32_t>() = status;
    *residual.data<float>() = residual_norm;
    *iterations.data<int32_t>() = completed;
  });
}

template <typename I, typename ApplyPreconditioner>
std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab_factor_impl(mx::array data, mx::array indices, mx::array indptr,
                         mx::array b, mx::array x0, int n_rows, float rtol,
                         float atol, int maxiter, ApplyPreconditioner &&apply) {
  mx::eval(data, indices, indptr, b, x0);
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  const auto *b_ptr = b.data<float>();
  const auto *x0_ptr = x0.data<float>();
  std::vector<float> rhs(b_ptr, b_ptr + n_rows);
  std::vector<float> guess(x0_ptr, x0_ptr + n_rows);
  auto matvec = [&](const std::vector<float> &x) {
    return host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, x, n_rows);
  };
  auto apply_vector = [&](const std::vector<float> &x) {
    return host_vector(apply(vector_to_mx_array(x)), n_rows,
                       "bicgstab preconditioner");
  };

  auto [solution, status, residual_norm, completed] =
      left_preconditioned_bicgstab_host(n_rows, rhs, guess, rtol, atol, maxiter,
                                        matvec, apply_vector);
  return {vector_to_mx_array(solution), mx::array(status, mx::int32),
          mx::array(residual_norm, mx::float32),
          mx::array(completed, mx::int32)};
}

} // namespace

void CSRBiCGSTABJacobi::eval_cpu(const std::vector<mx::array> &inputs,
                                 std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x0 = inputs[4];
  auto &inv_diag = inputs[5];

  if (indices.dtype() == mx::int32) {
    csr_bicgstab_jacobi_cpu_impl<int32_t>(
        data, indices, indptr, b, x0, inv_diag, outputs[0], outputs[1],
        outputs[2], outputs[3], n_rows_, rtol_, atol_, maxiter_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_bicgstab_jacobi_cpu_impl<int64_t>(
        data, indices, indptr, b, x0, inv_diag, outputs[0], outputs[1],
        outputs[2], outputs[3], n_rows_, rtol_, atol_, maxiter_, stream());
    return;
  }
  throw std::runtime_error(
      "csr_bicgstab_jacobi requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRBiCGSTABJacobi::eval_gpu(const std::vector<mx::array> &inputs,
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
      mx::allocator::malloc(static_cast<size_t>(9 * n_rows_) * sizeof(float)),
      mx::Shape{9 * n_rows_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_bicgstab_jacobi", data.dtype(), indices.dtype());
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
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(work));
}
#else
void CSRBiCGSTABJacobi::eval_gpu(const std::vector<mx::array> &,
                                 std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_bicgstab_jacobi has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab_jacobi(const mx::array &data, const mx::array &indices,
                    const mx::array &indptr, const mx::array &b,
                    const mx::array &x0, const mx::array &inv_diag, int n_rows,
                    int n_cols, float rtol, float atol, int maxiter,
                    mx::StreamOrDevice s) {
  require_bicgstab_base("csr_bicgstab_jacobi", data, indices, indptr, b, x0,
                        n_rows, n_cols, maxiter);
  require_rank(inv_diag, 1, "csr_bicgstab_jacobi inv_diag");
  require_linalg_float32(inv_diag, "csr_bicgstab_jacobi inv_diag");
  require_size(inv_diag, n_rows, "csr_bicgstab_jacobi inv_diag");

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);
  auto inv_diag_contig = mx::contiguous(inv_diag, false, stream);

  auto primitive = std::make_shared<CSRBiCGSTABJacobi>(stream, n_rows, n_cols,
                                                       rtol, atol, maxiter);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{n_rows}, mx::Shape{}, mx::Shape{}, mx::Shape{}},
      {mx::float32, mx::int32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, b_contig, x0_contig,
       inv_diag_contig});
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

std::tuple<mx::array, mx::array, mx::array, mx::array> csr_bicgstab_exact_lu(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, const mx::array &x0, const mx::array &perm,
    const mx::array &l_data, const mx::array &l_indices,
    const mx::array &l_indptr, const mx::array &u_data,
    const mx::array &u_indices, const mx::array &u_indptr, int n_rows,
    int n_cols, float rtol, float atol, int maxiter, mx::StreamOrDevice s) {
  require_bicgstab_base("csr_bicgstab_exact_lu", data, indices, indptr, b, x0,
                        n_rows, n_cols, maxiter);
  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);
  auto perm_contig = mx::contiguous(perm, false, stream);
  auto l_data_contig = mx::contiguous(l_data, false, stream);
  auto l_indices_contig = mx::contiguous(l_indices, false, stream);
  auto l_indptr_contig = mx::contiguous(l_indptr, false, stream);
  auto u_data_contig = mx::contiguous(u_data, false, stream);
  auto u_indices_contig = mx::contiguous(u_indices, false, stream);
  auto u_indptr_contig = mx::contiguous(u_indptr, false, stream);
  auto apply = [&](const mx::array &rhs) {
    return csr_exact_lu_preconditioner_apply(
        perm_contig, l_data_contig, l_indices_contig, l_indptr_contig,
        u_data_contig, u_indices_contig, u_indptr_contig, rhs, n_rows, n_cols,
        stream);
  };
  if (indices.dtype() == mx::int32) {
    return csr_bicgstab_factor_impl<int32_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
        rtol, atol, maxiter, apply);
  }
  if (indices.dtype() == mx::int64) {
    return csr_bicgstab_factor_impl<int64_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
        rtol, atol, maxiter, apply);
  }
  throw std::runtime_error(
      "csr_bicgstab_exact_lu requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab_ilu0(const mx::array &data, const mx::array &indices,
                  const mx::array &indptr, const mx::array &b,
                  const mx::array &x0, const mx::array &l_data,
                  const mx::array &l_indices, const mx::array &l_indptr,
                  const mx::array &u_data, const mx::array &u_indices,
                  const mx::array &u_indptr, int n_rows, int n_cols, float rtol,
                  float atol, int maxiter, mx::StreamOrDevice s) {
  require_bicgstab_base("csr_bicgstab_ilu0", data, indices, indptr, b, x0,
                        n_rows, n_cols, maxiter);
  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);
  auto l_data_contig = mx::contiguous(l_data, false, stream);
  auto l_indices_contig = mx::contiguous(l_indices, false, stream);
  auto l_indptr_contig = mx::contiguous(l_indptr, false, stream);
  auto u_data_contig = mx::contiguous(u_data, false, stream);
  auto u_indices_contig = mx::contiguous(u_indices, false, stream);
  auto u_indptr_contig = mx::contiguous(u_indptr, false, stream);
  auto apply = [&](const mx::array &rhs) {
    return csr_ilu0_preconditioner_apply(
        l_data_contig, l_indices_contig, l_indptr_contig, u_data_contig,
        u_indices_contig, u_indptr_contig, rhs, n_rows, n_cols, stream);
  };
  if (indices.dtype() == mx::int32) {
    return csr_bicgstab_factor_impl<int32_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
        rtol, atol, maxiter, apply);
  }
  if (indices.dtype() == mx::int64) {
    return csr_bicgstab_factor_impl<int64_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
        rtol, atol, maxiter, apply);
  }
  throw std::runtime_error(
      "csr_bicgstab_ilu0 requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab_exact_cholesky(const mx::array &data, const mx::array &indices,
                            const mx::array &indptr, const mx::array &b,
                            const mx::array &x0, const mx::array &l_data,
                            const mx::array &l_indices,
                            const mx::array &l_indptr, const mx::array &lt_data,
                            const mx::array &lt_indices,
                            const mx::array &lt_indptr, int n_rows, int n_cols,
                            float rtol, float atol, int maxiter,
                            mx::StreamOrDevice s) {
  require_bicgstab_base("csr_bicgstab_exact_cholesky", data, indices, indptr, b,
                        x0, n_rows, n_cols, maxiter);
  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);
  auto l_data_contig = mx::contiguous(l_data, false, stream);
  auto l_indices_contig = mx::contiguous(l_indices, false, stream);
  auto l_indptr_contig = mx::contiguous(l_indptr, false, stream);
  auto lt_data_contig = mx::contiguous(lt_data, false, stream);
  auto lt_indices_contig = mx::contiguous(lt_indices, false, stream);
  auto lt_indptr_contig = mx::contiguous(lt_indptr, false, stream);
  auto apply = [&](const mx::array &rhs) {
    return csr_exact_cholesky_preconditioner_apply(
        l_data_contig, l_indices_contig, l_indptr_contig, lt_data_contig,
        lt_indices_contig, lt_indptr_contig, rhs, n_rows, n_cols, stream);
  };
  if (indices.dtype() == mx::int32) {
    return csr_bicgstab_factor_impl<int32_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
        rtol, atol, maxiter, apply);
  }
  if (indices.dtype() == mx::int64) {
    return csr_bicgstab_factor_impl<int64_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
        rtol, atol, maxiter, apply);
  }
  throw std::runtime_error(
      "csr_bicgstab_exact_cholesky requires int32 or int64 indices.");
}

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bicgstab_exact_accelerate(const mx::array &data, const mx::array &indices,
                              const mx::array &indptr, const mx::array &b,
                              const mx::array &x0,
                              const AccelerateFloatSolve &solver, int n_rows,
                              int n_cols, float rtol, float atol, int maxiter,
                              mx::StreamOrDevice s) {
  require_bicgstab_base("csr_bicgstab_exact_accelerate", data, indices, indptr,
                        b, x0, n_rows, n_cols, maxiter);
  if (solver.rhs_size() != n_rows || solver.solution_size() != n_cols) {
    throw std::invalid_argument(
        "csr_bicgstab_exact_accelerate solver shape does not match A.");
  }
  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);
  auto apply = [&](const mx::array &rhs) { return solver.solve(rhs); };
  if (indices.dtype() == mx::int32) {
    return csr_bicgstab_factor_impl<int32_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
        rtol, atol, maxiter, apply);
  }
  if (indices.dtype() == mx::int64) {
    return csr_bicgstab_factor_impl<int64_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
        rtol, atol, maxiter, apply);
  }
  throw std::runtime_error(
      "csr_bicgstab_exact_accelerate requires int32 or int64 indices.");
}
#endif

} // namespace mlx_sparse
