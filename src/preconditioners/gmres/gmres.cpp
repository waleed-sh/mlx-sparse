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

#include "preconditioners/gmres/gmres.h"

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
#include "preconditioners/exact/exact.h"
#include "preconditioners/ilu0/ilu0.h"
#include "sparse/csr_matvec/csr_matvec.h"

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
#include "linalg/accelerate/solve/solve.h"
#endif

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

class CSRArnoldiJacobi : public mx::Primitive {
public:
  CSRArnoldiJacobi(mx::Stream stream, int n_rows, int n_cols, int k)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), k_(k) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRArnoldiJacobi"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRArnoldiJacobi &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ && k_ == rhs.k_;
  }

private:
  int n_rows_;
  int n_cols_;
  int k_;
};

inline bool finite_float(float value) { return std::isfinite(value); }

bool finite_vector(const std::vector<float> &values) {
  for (float value : values) {
    if (!finite_float(value)) {
      return false;
    }
  }
  return true;
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

template <typename ApplyPreconditioner>
std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres_native_left_preconditioned_impl(
    mx::array data, mx::array indices, mx::array indptr, mx::array b,
    mx::array x0, int n_rows, float rtol, float atol, int restart, int maxiter,
    mx::Stream stream, ApplyPreconditioner &&apply_preconditioner) {
  b.eval();
  x0.eval();
  const auto *b_ptr = b.data<float>();
  const auto *x0_ptr = x0.data<float>();

  std::vector<float> x(x0_ptr, x0_ptr + n_rows);
  std::vector<float> rhs(b_ptr, b_ptr + n_rows);
  const float b_norm = norm_float(rhs);
  const float tolerance = std::max(atol, rtol * b_norm);
  int iterations = 0;
  int status = maxiter > 0 ? maxiter : 1;
  float residual_norm = std::numeric_limits<float>::infinity();

  if (!finite_vector(x) || !finite_vector(rhs) || !finite_float(b_norm)) {
    return {vector_to_mx_array(x), mx::array(-3, mx::int32),
            mx::array(residual_norm, mx::float32), mx::array(0, mx::int32)};
  }

  while (iterations < maxiter) {
    std::vector<float> r(static_cast<size_t>(n_rows));
    try {
      auto x_mx = vector_to_mx_array(x);
      auto ax_mx =
          csr_matvec(data, indices, indptr, x_mx, n_rows, n_rows, stream);
      auto ax = host_vector(ax_mx, n_rows, "exact GMRES SpMV");
      for (int i = 0; i < n_rows; ++i) {
        r[static_cast<size_t>(i)] =
            rhs[static_cast<size_t>(i)] - ax[static_cast<size_t>(i)];
      }
    } catch (const std::exception &) {
      status = -1;
      break;
    }

    residual_norm = norm_float(r);
    if (!finite_vector(r) || !finite_float(residual_norm)) {
      status = -3;
      break;
    }
    if (residual_norm <= tolerance) {
      status = 0;
      break;
    }

    std::vector<float> z0;
    try {
      z0 = host_vector(apply_preconditioner(vector_to_mx_array(r)), n_rows,
                       "exact GMRES preconditioner");
    } catch (const std::exception &) {
      status = -1;
      break;
    }
    const float beta = norm_float(z0);
    if (!finite_vector(z0) || !finite_float(beta)) {
      status = -3;
      break;
    }
    if (beta <= std::numeric_limits<float>::epsilon()) {
      status = -1;
      break;
    }

    const int steps = std::min({restart, maxiter - iterations, n_rows});
    const int basis_cols = steps + 1;
    std::vector<float> basis(static_cast<size_t>(n_rows) * basis_cols, 0.0f);
    std::vector<float> h(static_cast<size_t>(basis_cols) * steps, 0.0f);
    for (int row = 0; row < n_rows; ++row) {
      basis[static_cast<size_t>(row) * basis_cols] =
          z0[static_cast<size_t>(row)] / beta;
    }

    int used = 0;
    bool failed = false;
    for (int j = 0; j < steps; ++j) {
      std::vector<float> q(static_cast<size_t>(n_rows));
      for (int row = 0; row < n_rows; ++row) {
        q[static_cast<size_t>(row)] =
            basis[static_cast<size_t>(row) * basis_cols + j];
      }

      std::vector<float> w;
      try {
        auto aq_mx = csr_matvec(data, indices, indptr, vector_to_mx_array(q),
                                n_rows, n_rows, stream);
        w = host_vector(apply_preconditioner(aq_mx), n_rows,
                        "exact GMRES Arnoldi preconditioner");
      } catch (const std::exception &) {
        status = -1;
        failed = true;
        break;
      }
      if (!finite_vector(w)) {
        status = -3;
        failed = true;
        break;
      }

      for (int pass = 0; pass < 2; ++pass) {
        for (int col = 0; col <= j; ++col) {
          double coeff = 0.0;
          for (int row = 0; row < n_rows; ++row) {
            coeff += basis[static_cast<size_t>(row) * basis_cols + col] *
                     w[static_cast<size_t>(row)];
          }
          h[static_cast<size_t>(col) * steps + j] += static_cast<float>(coeff);
          for (int row = 0; row < n_rows; ++row) {
            w[static_cast<size_t>(row)] -=
                static_cast<float>(coeff) *
                basis[static_cast<size_t>(row) * basis_cols + col];
          }
        }
      }

      const float h_next = norm_float(w);
      h[static_cast<size_t>(j + 1) * steps + j] = h_next;
      used = j + 1;
      if (!finite_float(h_next)) {
        status = -3;
        failed = true;
        break;
      }
      if (h_next <= std::numeric_limits<float>::epsilon()) {
        break;
      }
      for (int row = 0; row < n_rows; ++row) {
        basis[static_cast<size_t>(row) * basis_cols + j + 1] =
            w[static_cast<size_t>(row)] / h_next;
      }
    }
    if (failed) {
      break;
    }
    if (used == 0) {
      status = -1;
      break;
    }

    std::vector<double> h_used(static_cast<size_t>(used + 1) * used, 0.0);
    bool finite_h = true;
    for (int row = 0; row < used + 1; ++row) {
      for (int col = 0; col < used; ++col) {
        const float value = h[static_cast<size_t>(row) * steps + col];
        h_used[static_cast<size_t>(row) * used + col] = value;
        finite_h = finite_h && finite_float(value);
      }
    }
    if (!finite_h) {
      status = -3;
      break;
    }

    std::vector<double> e1(static_cast<size_t>(used + 1), 0.0);
    e1[0] = beta;
    std::vector<double> y;
    try {
      y = least_squares_upper_hessenberg_givens_qr(h_used, e1, used + 1, used);
    } catch (const std::exception &) {
      status = -1;
      break;
    }

    for (int row = 0; row < n_rows; ++row) {
      double update = 0.0;
      for (int col = 0; col < used; ++col) {
        update += basis[static_cast<size_t>(row) * basis_cols + col] *
                  y[static_cast<size_t>(col)];
      }
      if (!std::isfinite(update)) {
        status = -3;
        break;
      }
      x[static_cast<size_t>(row)] += static_cast<float>(update);
      if (!finite_float(x[static_cast<size_t>(row)])) {
        status = -3;
        break;
      }
    }
    if (status < 0) {
      break;
    }

    iterations += used;
    if (iterations >= maxiter) {
      try {
        auto ax =
            host_vector(csr_matvec(data, indices, indptr, vector_to_mx_array(x),
                                   n_rows, n_rows, stream),
                        n_rows, "exact GMRES final SpMV");
        std::vector<float> final_r(static_cast<size_t>(n_rows));
        for (int i = 0; i < n_rows; ++i) {
          final_r[static_cast<size_t>(i)] =
              rhs[static_cast<size_t>(i)] - ax[static_cast<size_t>(i)];
        }
        residual_norm = norm_float(final_r);
      } catch (const std::exception &) {
        status = -1;
        break;
      }
      if (!finite_float(residual_norm)) {
        status = -3;
      } else if (residual_norm <= tolerance) {
        status = 0;
      }
      break;
    }
  }

  return {vector_to_mx_array(x), mx::array(status, mx::int32),
          mx::array(residual_norm, mx::float32),
          mx::array(iterations, mx::int32)};
}

template <typename I>
void csr_arnoldi_jacobi_cpu_impl(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &inv_diag, const mx::array &v0, mx::array &h,
    mx::array &basis, mx::array &actual, int n_rows, int k, mx::Stream stream) {
  h.set_data(mx::allocator::malloc(h.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(inv_diag);
  encoder.set_input_array(v0);
  encoder.set_output_array(h);
  encoder.set_output_array(basis);
  encoder.set_output_array(actual);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    inv_diag = mx::array::unsafe_weak_copy(inv_diag),
                    v0 = mx::array::unsafe_weak_copy(v0),
                    h = mx::array::unsafe_weak_copy(h),
                    basis = mx::array::unsafe_weak_copy(basis),
                    actual = mx::array::unsafe_weak_copy(actual), n_rows,
                    k]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *inv_diag_ptr = inv_diag.data<float>();
    const auto *v0_ptr = v0.data<float>();
    auto *h_ptr = h.data<float>();
    auto *basis_ptr = basis.data<float>();
    auto *actual_ptr = actual.data<int32_t>();

    const int cols = k + 1;
    std::fill(h_ptr, h_ptr + static_cast<size_t>(cols) * k, 0.0f);
    std::fill(basis_ptr, basis_ptr + static_cast<size_t>(n_rows) * cols, 0.0f);

    double v_norm2 = 0.0;
    bool finite = true;
    for (int i = 0; i < n_rows; ++i) {
      const float vi = v0_ptr[i];
      v_norm2 += static_cast<double>(vi) * static_cast<double>(vi);
      finite = finite && finite_float(vi) && finite_float(inv_diag_ptr[i]);
    }
    const float v_norm = std::sqrt(std::max(v_norm2, 0.0));
    if (!finite || !finite_float(v_norm)) {
      *actual_ptr = 0;
      return;
    }
    if (v_norm <= std::numeric_limits<float>::epsilon()) {
      for (int i = 0; i < n_rows; ++i) {
        basis_ptr[static_cast<size_t>(i) * cols] = i == 0 ? 1.0f : 0.0f;
      }
    } else {
      for (int i = 0; i < n_rows; ++i) {
        basis_ptr[static_cast<size_t>(i) * cols] = v0_ptr[i] / v_norm;
      }
    }

    std::vector<float> w(static_cast<size_t>(n_rows));
    std::vector<float> q(static_cast<size_t>(n_rows));
    int used = 0;
    const float eps = std::numeric_limits<float>::epsilon();
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n_rows; ++i) {
        q[i] = basis_ptr[static_cast<size_t>(i) * cols + j];
      }
      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, q.data(), w.data(),
                     n_rows);
      for (int row = 0; row < n_rows; ++row) {
        w[static_cast<size_t>(row)] *= inv_diag_ptr[row];
      }
      for (int pass = 0; pass < 2; ++pass) {
        for (int col = 0; col <= j; ++col) {
          double coeff = 0.0;
          for (int row = 0; row < n_rows; ++row) {
            coeff += basis_ptr[static_cast<size_t>(row) * cols + col] * w[row];
          }
          h_ptr[static_cast<size_t>(col) * k + j] += static_cast<float>(coeff);
          for (int row = 0; row < n_rows; ++row) {
            w[row] -= static_cast<float>(coeff) *
                      basis_ptr[static_cast<size_t>(row) * cols + col];
          }
        }
      }
      const float h_next = norm_float(w);
      h_ptr[static_cast<size_t>(j + 1) * k + j] = h_next;
      used = j + 1;
      if (!finite_float(h_next)) {
        used = 0;
        break;
      }
      if (h_next <= eps) {
        break;
      }
      for (int row = 0; row < n_rows; ++row) {
        basis_ptr[static_cast<size_t>(row) * cols + j + 1] = w[row] / h_next;
      }
    }
    *actual_ptr = used;
  });
}

void CSRArnoldiJacobi::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &inv_diag = inputs[3];
  auto &v0 = inputs[4];

  if (indices.dtype() == mx::int32) {
    csr_arnoldi_jacobi_cpu_impl<int32_t>(data, indices, indptr, inv_diag, v0,
                                         outputs[0], outputs[1], outputs[2],
                                         n_rows_, k_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_arnoldi_jacobi_cpu_impl<int64_t>(data, indices, indptr, inv_diag, v0,
                                         outputs[0], outputs[1], outputs[2],
                                         n_rows_, k_, stream());
    return;
  }
  throw std::runtime_error(
      "csr_arnoldi_jacobi requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRArnoldiJacobi::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &inv_diag = inputs[3];
  auto &v0 = inputs[4];
  auto &h = outputs[0];
  auto &basis = outputs[1];
  auto &actual = outputs[2];

  h.set_data(mx::allocator::malloc(h.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));
  mx::array work(
      mx::allocator::malloc(static_cast<size_t>(n_rows_) * sizeof(float)),
      mx::Shape{n_rows_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_arnoldi_jacobi", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(inv_diag, 3);
  encoder.set_input_array(v0, 4);
  encoder.set_output_array(h, 5);
  encoder.set_output_array(basis, 6);
  encoder.set_output_array(actual, 7);
  encoder.set_output_array(work, 8);
  encoder.set_bytes(n_rows_, 9);
  encoder.set_bytes(n_cols_, 10);
  encoder.set_bytes(k_, 11);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(work));
}
#else
void CSRArnoldiJacobi::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_arnoldi_jacobi has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
csr_arnoldi_jacobi(const mx::array &data, const mx::array &indices,
                   const mx::array &indptr, const mx::array &inv_diag,
                   const mx::array &v0, int n_rows, int n_cols, int k,
                   mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_arnoldi_jacobi requires a non-empty square matrix.");
  }
  if (k <= 0 || k > n_rows) {
    throw std::invalid_argument(
        "csr_arnoldi_jacobi k must satisfy 0 < k <= n_rows.");
  }
  require_rank(data, 1, "csr_arnoldi_jacobi data");
  require_rank(indices, 1, "csr_arnoldi_jacobi indices");
  require_rank(indptr, 1, "csr_arnoldi_jacobi indptr");
  require_rank(inv_diag, 1, "csr_arnoldi_jacobi inv_diag");
  require_rank(v0, 1, "csr_arnoldi_jacobi v0");
  require_linalg_float32(data, "csr_arnoldi_jacobi data");
  require_linalg_float32(inv_diag, "csr_arnoldi_jacobi inv_diag");
  require_linalg_float32(v0, "csr_arnoldi_jacobi v0");
  require_same_index_dtype(indices, indptr, "csr_arnoldi_jacobi indices",
                           "csr_arnoldi_jacobi indptr");
  require_size(indptr, n_rows + 1, "csr_arnoldi_jacobi indptr");
  require_size(inv_diag, n_rows, "csr_arnoldi_jacobi inv_diag");
  require_size(v0, n_rows, "csr_arnoldi_jacobi v0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_arnoldi_jacobi data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto inv_diag_contig = mx::contiguous(inv_diag, false, stream);
  auto v0_contig = mx::contiguous(v0, false, stream);

  auto primitive =
      std::make_shared<CSRArnoldiJacobi>(stream, n_rows, n_cols, k);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{k + 1, k}, mx::Shape{n_rows, k + 1}, mx::Shape{}},
      {mx::float32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, inv_diag_contig, v0_contig});
  return {outputs[0], outputs[1], outputs[2]};
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres_jacobi_impl(mx::array data, mx::array indices, mx::array indptr,
                      mx::array b, mx::array x0, mx::array inv_diag, int n_rows,
                      float rtol, float atol, int restart, int maxiter,
                      mx::Stream stream) {
  data.eval();
  indices.eval();
  indptr.eval();
  b.eval();
  x0.eval();
  inv_diag.eval();
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  const auto *b_ptr = b.data<float>();
  const auto *x0_ptr = x0.data<float>();
  const auto *inv_diag_ptr = inv_diag.data<float>();

  std::vector<float> x(x0_ptr, x0_ptr + n_rows);
  std::vector<float> rhs(b_ptr, b_ptr + n_rows);
  std::vector<float> inv_diag_host(inv_diag_ptr, inv_diag_ptr + n_rows);
  const float b_norm = norm_float(rhs);
  const float tolerance = std::max(atol, rtol * b_norm);
  int iterations = 0;
  int status = maxiter > 0 ? maxiter : 1;
  float residual_norm = std::numeric_limits<float>::infinity();

  bool setup_finite = true;
  for (int i = 0; i < n_rows; ++i) {
    setup_finite = setup_finite && finite_float(rhs[static_cast<size_t>(i)]) &&
                   finite_float(x[static_cast<size_t>(i)]) &&
                   finite_float(inv_diag_host[static_cast<size_t>(i)]);
  }
  if (!setup_finite) {
    status = -3;
    mx::array x_out(x.begin(), mx::Shape{n_rows}, mx::float32);
    return {x_out, mx::array(status, mx::int32),
            mx::array(residual_norm, mx::float32), mx::array(0, mx::int32)};
  }

  while (iterations < maxiter) {
    auto ax = host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, x, n_rows);
    std::vector<float> r(static_cast<size_t>(n_rows));
    std::vector<float> z0(static_cast<size_t>(n_rows));
    bool finite = true;
    for (int i = 0; i < n_rows; ++i) {
      const size_t idx = static_cast<size_t>(i);
      r[idx] = rhs[idx] - ax[idx];
      z0[idx] = inv_diag_host[idx] * r[idx];
      finite = finite && finite_float(r[idx]) && finite_float(z0[idx]);
    }
    residual_norm = norm_float(r);
    if (!finite || !finite_float(residual_norm)) {
      status = -3;
      break;
    }
    if (residual_norm <= tolerance) {
      status = 0;
      break;
    }

    const float beta = norm_float(z0);
    if (!finite_float(beta) || beta <= std::numeric_limits<float>::epsilon()) {
      status = -1;
      break;
    }

    const int steps = std::min({restart, maxiter - iterations, n_rows});
    std::vector<float> v0_data(static_cast<size_t>(n_rows));
    for (int i = 0; i < n_rows; ++i) {
      v0_data[static_cast<size_t>(i)] = z0[static_cast<size_t>(i)] / beta;
    }
    auto v0 = mx::array(v0_data.begin(), mx::Shape{n_rows}, mx::float32);

    auto [h_mx, basis_mx, actual_k_mx] = csr_arnoldi_jacobi(
        data, indices, indptr, inv_diag, v0, n_rows, n_rows, steps, stream);
    mx::eval(h_mx, basis_mx, actual_k_mx);

    const int used = static_cast<int>(actual_k_mx.item<int32_t>());
    const float *h_ptr = h_mx.data<float>();
    const float *basis_ptr = basis_mx.data<float>();
    if (used == 0) {
      status = -1;
      break;
    }

    std::vector<double> h_used(static_cast<size_t>(used + 1) * used, 0.0);
    finite = true;
    for (int row = 0; row < used + 1; ++row) {
      for (int col = 0; col < used; ++col) {
        const float value = h_ptr[static_cast<size_t>(row) * steps + col];
        h_used[static_cast<size_t>(row) * used + col] = value;
        finite = finite && finite_float(value);
      }
    }
    if (!finite) {
      status = -3;
      break;
    }

    std::vector<double> e1(static_cast<size_t>(used + 1), 0.0);
    e1[0] = beta;
    std::vector<double> y;
    try {
      y = least_squares_upper_hessenberg_givens_qr(h_used, e1, used + 1, used);
    } catch (const std::exception &) {
      status = -1;
      break;
    }

    for (int row = 0; row < n_rows; ++row) {
      double update = 0.0;
      for (int col = 0; col < used; ++col) {
        const float basis_value =
            basis_ptr[static_cast<size_t>(row) * (steps + 1) + col];
        update +=
            static_cast<double>(basis_value) * y[static_cast<size_t>(col)];
      }
      if (!std::isfinite(update)) {
        status = -3;
        break;
      }
      x[static_cast<size_t>(row)] += static_cast<float>(update);
      if (!finite_float(x[static_cast<size_t>(row)])) {
        status = -3;
        break;
      }
    }
    if (status < 0) {
      break;
    }

    iterations += used;
    if (iterations >= maxiter) {
      auto final_ax =
          host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, x, n_rows);
      std::vector<float> final_r(static_cast<size_t>(n_rows));
      for (int i = 0; i < n_rows; ++i) {
        final_r[static_cast<size_t>(i)] =
            rhs[static_cast<size_t>(i)] - final_ax[static_cast<size_t>(i)];
      }
      residual_norm = norm_float(final_r);
      if (!finite_float(residual_norm)) {
        status = -3;
      } else if (residual_norm <= tolerance) {
        status = 0;
      }
      break;
    }
  }

  mx::array x_out(x.begin(), mx::Shape{n_rows}, mx::float32);
  mx::array info(status, mx::int32);
  mx::array residual(residual_norm, mx::float32);
  mx::array iters(iterations, mx::int32);
  return {x_out, info, residual, iters};
}

} // namespace

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres_jacobi(const mx::array &data, const mx::array &indices,
                 const mx::array &indptr, const mx::array &b,
                 const mx::array &x0, const mx::array &inv_diag, int n_rows,
                 int n_cols, float rtol, float atol, int restart, int maxiter,
                 mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_gmres_jacobi requires a non-empty square matrix.");
  }
  if (restart <= 0 || maxiter < 0) {
    throw std::invalid_argument(
        "csr_gmres_jacobi requires restart > 0 and maxiter >= 0.");
  }
  require_rank(data, 1, "csr_gmres_jacobi data");
  require_rank(indices, 1, "csr_gmres_jacobi indices");
  require_rank(indptr, 1, "csr_gmres_jacobi indptr");
  require_rank(b, 1, "csr_gmres_jacobi b");
  require_rank(x0, 1, "csr_gmres_jacobi x0");
  require_rank(inv_diag, 1, "csr_gmres_jacobi inv_diag");
  require_linalg_float32(data, "csr_gmres_jacobi data");
  require_linalg_float32(b, "csr_gmres_jacobi b");
  require_linalg_float32(x0, "csr_gmres_jacobi x0");
  require_linalg_float32(inv_diag, "csr_gmres_jacobi inv_diag");
  require_same_index_dtype(indices, indptr, "csr_gmres_jacobi indices",
                           "csr_gmres_jacobi indptr");
  require_size(indptr, n_rows + 1, "csr_gmres_jacobi indptr");
  require_size(b, n_rows, "csr_gmres_jacobi b");
  require_size(x0, n_cols, "csr_gmres_jacobi x0");
  require_size(inv_diag, n_rows, "csr_gmres_jacobi inv_diag");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_gmres_jacobi data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);
  auto inv_diag_contig = mx::contiguous(inv_diag, false, stream);

  if (indices.dtype() == mx::int32) {
    return csr_gmres_jacobi_impl<int32_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig,
        inv_diag_contig, n_rows, rtol, atol, restart, maxiter, stream);
  }
  if (indices.dtype() == mx::int64) {
    return csr_gmres_jacobi_impl<int64_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig,
        inv_diag_contig, n_rows, rtol, atol, restart, maxiter, stream);
  }
  throw std::runtime_error("csr_gmres_jacobi requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres_exact_lu(const mx::array &data, const mx::array &indices,
                   const mx::array &indptr, const mx::array &b,
                   const mx::array &x0, const mx::array &perm,
                   const mx::array &l_data, const mx::array &l_indices,
                   const mx::array &l_indptr, const mx::array &u_data,
                   const mx::array &u_indices, const mx::array &u_indptr,
                   int n_rows, int n_cols, float rtol, float atol, int restart,
                   int maxiter, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_gmres_exact_lu requires a non-empty square matrix.");
  }
  if (restart <= 0 || maxiter < 0) {
    throw std::invalid_argument(
        "csr_gmres_exact_lu requires restart > 0 and maxiter >= 0.");
  }
  require_rank(data, 1, "csr_gmres_exact_lu data");
  require_rank(indices, 1, "csr_gmres_exact_lu indices");
  require_rank(indptr, 1, "csr_gmres_exact_lu indptr");
  require_rank(b, 1, "csr_gmres_exact_lu b");
  require_rank(x0, 1, "csr_gmres_exact_lu x0");
  require_linalg_float32(data, "csr_gmres_exact_lu data");
  require_linalg_float32(b, "csr_gmres_exact_lu b");
  require_linalg_float32(x0, "csr_gmres_exact_lu x0");
  require_same_index_dtype(indices, indptr, "csr_gmres_exact_lu indices",
                           "csr_gmres_exact_lu indptr");
  require_size(indptr, n_rows + 1, "csr_gmres_exact_lu indptr");
  require_size(b, n_rows, "csr_gmres_exact_lu b");
  require_size(x0, n_cols, "csr_gmres_exact_lu x0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_gmres_exact_lu data and indices must have equal length.");
  }

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
  return csr_gmres_native_left_preconditioned_impl(
      data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
      rtol, atol, restart, maxiter, stream, apply);
}

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres_ilu0(const mx::array &data, const mx::array &indices,
               const mx::array &indptr, const mx::array &b, const mx::array &x0,
               const mx::array &l_data, const mx::array &l_indices,
               const mx::array &l_indptr, const mx::array &u_data,
               const mx::array &u_indices, const mx::array &u_indptr,
               int n_rows, int n_cols, float rtol, float atol, int restart,
               int maxiter, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_gmres_ilu0 requires a non-empty square matrix.");
  }
  if (restart <= 0 || maxiter < 0) {
    throw std::invalid_argument(
        "csr_gmres_ilu0 requires restart > 0 and maxiter >= 0.");
  }
  require_rank(data, 1, "csr_gmres_ilu0 data");
  require_rank(indices, 1, "csr_gmres_ilu0 indices");
  require_rank(indptr, 1, "csr_gmres_ilu0 indptr");
  require_rank(b, 1, "csr_gmres_ilu0 b");
  require_rank(x0, 1, "csr_gmres_ilu0 x0");
  require_linalg_float32(data, "csr_gmres_ilu0 data");
  require_linalg_float32(b, "csr_gmres_ilu0 b");
  require_linalg_float32(x0, "csr_gmres_ilu0 x0");
  require_same_index_dtype(indices, indptr, "csr_gmres_ilu0 indices",
                           "csr_gmres_ilu0 indptr");
  require_size(indptr, n_rows + 1, "csr_gmres_ilu0 indptr");
  require_size(b, n_rows, "csr_gmres_ilu0 b");
  require_size(x0, n_cols, "csr_gmres_ilu0 x0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_gmres_ilu0 data and indices must have equal length.");
  }

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
  return csr_gmres_native_left_preconditioned_impl(
      data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
      rtol, atol, restart, maxiter, stream, apply);
}

std::tuple<mx::array, mx::array, mx::array, mx::array> csr_gmres_exact_cholesky(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, const mx::array &x0, const mx::array &l_data,
    const mx::array &l_indices, const mx::array &l_indptr,
    const mx::array &lt_data, const mx::array &lt_indices,
    const mx::array &lt_indptr, int n_rows, int n_cols, float rtol, float atol,
    int restart, int maxiter, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_gmres_exact_cholesky requires a non-empty square matrix.");
  }
  if (restart <= 0 || maxiter < 0) {
    throw std::invalid_argument(
        "csr_gmres_exact_cholesky requires restart > 0 and maxiter >= 0.");
  }
  require_rank(data, 1, "csr_gmres_exact_cholesky data");
  require_rank(indices, 1, "csr_gmres_exact_cholesky indices");
  require_rank(indptr, 1, "csr_gmres_exact_cholesky indptr");
  require_rank(b, 1, "csr_gmres_exact_cholesky b");
  require_rank(x0, 1, "csr_gmres_exact_cholesky x0");
  require_linalg_float32(data, "csr_gmres_exact_cholesky data");
  require_linalg_float32(b, "csr_gmres_exact_cholesky b");
  require_linalg_float32(x0, "csr_gmres_exact_cholesky x0");
  require_same_index_dtype(indices, indptr, "csr_gmres_exact_cholesky indices",
                           "csr_gmres_exact_cholesky indptr");
  require_size(indptr, n_rows + 1, "csr_gmres_exact_cholesky indptr");
  require_size(b, n_rows, "csr_gmres_exact_cholesky b");
  require_size(x0, n_cols, "csr_gmres_exact_cholesky x0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_gmres_exact_cholesky data and indices must have equal length.");
  }

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
  return csr_gmres_native_left_preconditioned_impl(
      data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
      rtol, atol, restart, maxiter, stream, apply);
}

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres_exact_accelerate(const mx::array &data, const mx::array &indices,
                           const mx::array &indptr, const mx::array &b,
                           const mx::array &x0,
                           const AccelerateFloatSolve &solver, int n_rows,
                           int n_cols, float rtol, float atol, int restart,
                           int maxiter, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_gmres_exact_accelerate requires a non-empty square matrix.");
  }
  if (restart <= 0 || maxiter < 0) {
    throw std::invalid_argument(
        "csr_gmres_exact_accelerate requires restart > 0 and maxiter >= 0.");
  }
  if (solver.rhs_size() != n_rows || solver.solution_size() != n_cols) {
    throw std::invalid_argument(
        "csr_gmres_exact_accelerate solver shape does not match A.");
  }
  require_rank(data, 1, "csr_gmres_exact_accelerate data");
  require_rank(indices, 1, "csr_gmres_exact_accelerate indices");
  require_rank(indptr, 1, "csr_gmres_exact_accelerate indptr");
  require_rank(b, 1, "csr_gmres_exact_accelerate b");
  require_rank(x0, 1, "csr_gmres_exact_accelerate x0");
  require_linalg_float32(data, "csr_gmres_exact_accelerate data");
  require_linalg_float32(b, "csr_gmres_exact_accelerate b");
  require_linalg_float32(x0, "csr_gmres_exact_accelerate x0");
  require_same_index_dtype(indices, indptr,
                           "csr_gmres_exact_accelerate indices",
                           "csr_gmres_exact_accelerate indptr");
  require_size(indptr, n_rows + 1, "csr_gmres_exact_accelerate indptr");
  require_size(b, n_rows, "csr_gmres_exact_accelerate b");
  require_size(x0, n_cols, "csr_gmres_exact_accelerate x0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_gmres_exact_accelerate data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto x0_contig = mx::contiguous(x0, false, stream);

  auto apply = [&](const mx::array &rhs) { return solver.solve(rhs); };
  return csr_gmres_native_left_preconditioned_impl(
      data_contig, indices_contig, indptr_contig, b_contig, x0_contig, n_rows,
      rtol, atol, restart, maxiter, stream, apply);
}
#endif

} // namespace mlx_sparse
