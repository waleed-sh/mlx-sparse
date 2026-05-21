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

#include "sparse/csr_linalg.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
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

namespace mlx_sparse {

namespace {

constexpr int kSolverThreads = 256;

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
           rtol_ == rhs.rtol_ && atol_ == rhs.atol_ &&
           maxiter_ == rhs.maxiter_;
  }

private:
  int n_rows_;
  int n_cols_;
  float rtol_;
  float atol_;
  int maxiter_;
};

class CSRLanczos : public mx::Primitive {
public:
  CSRLanczos(mx::Stream stream, int n_rows, int n_cols, int k,
             bool reorthogonalize)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), k_(k),
        reorthogonalize_(reorthogonalize) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRLanczos"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRLanczos &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           k_ == rhs.k_ && reorthogonalize_ == rhs.reorthogonalize_;
  }

private:
  int n_rows_;
  int n_cols_;
  int k_;
  bool reorthogonalize_;
};

class CSRArnoldi : public mx::Primitive {
public:
  CSRArnoldi(mx::Stream stream, int n_rows, int n_cols, int k)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), k_(k) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRArnoldi"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRArnoldi &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ && k_ == rhs.k_;
  }

private:
  int n_rows_;
  int n_cols_;
  int k_;
};

class CSRTriangularSolve : public mx::Primitive {
public:
  CSRTriangularSolve(mx::Stream stream, int n_rows, int n_cols, bool lower,
                     bool unit_diagonal)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), lower_(lower),
        unit_diagonal_(unit_diagonal) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRTriangularSolve"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRTriangularSolve &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           lower_ == rhs.lower_ && unit_diagonal_ == rhs.unit_diagonal_;
  }

private:
  int n_rows_;
  int n_cols_;
  bool lower_;
  bool unit_diagonal_;
};

class CSRVdot : public mx::Primitive {
public:
  CSRVdot(mx::Stream stream, int n_rows, int n_cols, bool conjugate_lhs)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols),
        conjugate_lhs_(conjugate_lhs) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRVdot"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRVdot &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           conjugate_lhs_ == rhs.conjugate_lhs_;
  }

private:
  int n_rows_;
  int n_cols_;
  bool conjugate_lhs_;
};

class CSRPermuteVector : public mx::Primitive {
public:
  explicit CSRPermuteVector(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRPermuteVector"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

template <typename I>
void csr_spmv_float(const float *data, const I *indices, const I *indptr,
                    const float *x, float *out, int n_rows) {
  for (int row = 0; row < n_rows; ++row) {
    float acc = 0.0f;
    for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
      acc += data[p] * x[indices[p]];
    }
    out[row] = acc;
  }
}

float dot_float(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  double acc = 0.0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    acc += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
  }
  return static_cast<float>(acc);
}

float dot_column_float(const float *basis, const float *w, int n, int stride,
                       int col) {
  double acc = 0.0;
  for (int row = 0; row < n; ++row) {
    acc += static_cast<double>(basis[static_cast<size_t>(row) * stride + col]) *
           static_cast<double>(w[row]);
  }
  return static_cast<float>(acc);
}

float norm_float(const std::vector<float> &x) {
  return std::sqrt(std::max(dot_float(x, x), 0.0f));
}

std::vector<double> solve_dense_system(std::vector<double> a,
                                       std::vector<double> b, int n) {
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double pivot_abs = std::abs(a[static_cast<size_t>(col) * n + col]);
    for (int row = col + 1; row < n; ++row) {
      const double candidate =
          std::abs(a[static_cast<size_t>(row) * n + col]);
      if (candidate > pivot_abs) {
        pivot_abs = candidate;
        pivot = row;
      }
    }
    if (pivot_abs <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error("small dense solve encountered a singular matrix.");
    }
    if (pivot != col) {
      for (int j = col; j < n; ++j) {
        std::swap(a[static_cast<size_t>(col) * n + j],
                  a[static_cast<size_t>(pivot) * n + j]);
      }
      std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(pivot)]);
    }
    const double diag = a[static_cast<size_t>(col) * n + col];
    for (int row = col + 1; row < n; ++row) {
      const double factor = a[static_cast<size_t>(row) * n + col] / diag;
      if (factor == 0.0) {
        continue;
      }
      a[static_cast<size_t>(row) * n + col] = 0.0;
      for (int j = col + 1; j < n; ++j) {
        a[static_cast<size_t>(row) * n + j] -=
            factor * a[static_cast<size_t>(col) * n + j];
      }
      b[static_cast<size_t>(row)] -= factor * b[static_cast<size_t>(col)];
    }
  }
  std::vector<double> x(static_cast<size_t>(n), 0.0);
  for (int row = n - 1; row >= 0; --row) {
    double sum = b[static_cast<size_t>(row)];
    for (int col = row + 1; col < n; ++col) {
      sum -= a[static_cast<size_t>(row) * n + col] *
             x[static_cast<size_t>(col)];
    }
    x[static_cast<size_t>(row)] =
        sum / a[static_cast<size_t>(row) * n + row];
  }
  return x;
}

std::vector<double> least_squares_normal_equations(const std::vector<double> &a,
                                                   const std::vector<double> &b,
                                                   int rows, int cols) {
  std::vector<double> normal(static_cast<size_t>(cols) * cols, 0.0);
  std::vector<double> rhs(static_cast<size_t>(cols), 0.0);
  for (int i = 0; i < rows; ++i) {
    for (int col = 0; col < cols; ++col) {
      const double a_ic = a[static_cast<size_t>(i) * cols + col];
      rhs[static_cast<size_t>(col)] += a_ic * b[static_cast<size_t>(i)];
      for (int j = 0; j < cols; ++j) {
        normal[static_cast<size_t>(col) * cols + j] +=
            a_ic * a[static_cast<size_t>(i) * cols + j];
      }
    }
  }
  return solve_dense_system(std::move(normal), std::move(rhs), cols);
}

template <typename I>
std::vector<float> host_csr_spmv(const float *data, const I *indices,
                                 const I *indptr, const std::vector<float> &x,
                                 int n_rows) {
  std::vector<float> out(static_cast<size_t>(n_rows), 0.0f);
  csr_spmv_float(data, indices, indptr, x.data(), out.data(), n_rows);
  return out;
}

template <typename I>
std::vector<float>
host_csr_spmv_transpose(const float *data, const I *indices, const I *indptr,
                        const std::vector<float> &x, int n_rows, int n_cols) {
  std::vector<float> out(static_cast<size_t>(n_cols), 0.0f);
  for (int row = 0; row < n_rows; ++row) {
    const float x_row = x[static_cast<size_t>(row)];
    for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
      out[static_cast<size_t>(indices[p])] += data[p] * x_row;
    }
  }
  return out;
}

std::pair<std::vector<float>, std::vector<float>>
jacobi_symmetric(std::vector<float> a, int n) {
  std::vector<float> vectors(static_cast<size_t>(n) * n, 0.0f);
  for (int i = 0; i < n; ++i) {
    vectors[static_cast<size_t>(i) * n + i] = 1.0f;
  }
  const int max_sweeps = std::max(32, 12 * n * n);
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    int p = 0;
    int q = 1;
    float max_offdiag = 0.0f;
    for (int row = 0; row < n; ++row) {
      for (int col = row + 1; col < n; ++col) {
        const float value = std::abs(a[static_cast<size_t>(row) * n + col]);
        if (value > max_offdiag) {
          max_offdiag = value;
          p = row;
          q = col;
        }
      }
    }
    if (max_offdiag <= 1e-6f) {
      break;
    }
    const float app = a[static_cast<size_t>(p) * n + p];
    const float aqq = a[static_cast<size_t>(q) * n + q];
    const float apq = a[static_cast<size_t>(p) * n + q];
    const float tau = (aqq - app) / (2.0f * apq);
    const float t =
        (tau >= 0.0f ? 1.0f : -1.0f) /
        (std::abs(tau) + std::sqrt(1.0f + tau * tau));
    const float c = 1.0f / std::sqrt(1.0f + t * t);
    const float s = t * c;
    for (int k = 0; k < n; ++k) {
      const float akp = a[static_cast<size_t>(k) * n + p];
      const float akq = a[static_cast<size_t>(k) * n + q];
      a[static_cast<size_t>(k) * n + p] = c * akp - s * akq;
      a[static_cast<size_t>(k) * n + q] = s * akp + c * akq;
    }
    for (int k = 0; k < n; ++k) {
      const float apk = a[static_cast<size_t>(p) * n + k];
      const float aqk = a[static_cast<size_t>(q) * n + k];
      a[static_cast<size_t>(p) * n + k] = c * apk - s * aqk;
      a[static_cast<size_t>(q) * n + k] = s * apk + c * aqk;
    }
    for (int k = 0; k < n; ++k) {
      const float vkp = vectors[static_cast<size_t>(k) * n + p];
      const float vkq = vectors[static_cast<size_t>(k) * n + q];
      vectors[static_cast<size_t>(k) * n + p] = c * vkp - s * vkq;
      vectors[static_cast<size_t>(k) * n + q] = s * vkp + c * vkq;
    }
  }
  std::vector<float> values(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    values[static_cast<size_t>(i)] = a[static_cast<size_t>(i) * n + i];
  }
  return {values, vectors};
}

std::vector<int> select_ritz_indices(const std::vector<float> &values, int k,
                                     const std::string &which) {
  std::vector<int> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  if (which == "SM" || which == "SA" || which == "SR") {
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      if (which == "SM") {
        return std::abs(values[static_cast<size_t>(lhs)]) <
               std::abs(values[static_cast<size_t>(rhs)]);
      }
      return values[static_cast<size_t>(lhs)] < values[static_cast<size_t>(rhs)];
    });
  } else {
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      if (which == "LM") {
        return std::abs(values[static_cast<size_t>(lhs)]) >
               std::abs(values[static_cast<size_t>(rhs)]);
      }
      return values[static_cast<size_t>(lhs)] > values[static_cast<size_t>(rhs)];
    });
  }
  order.resize(static_cast<size_t>(k));
  return order;
}

template <typename Apply>
std::tuple<std::vector<float>, std::vector<float>, int>
host_lanczos_operator(int n, int steps, Apply &&apply) {
  std::vector<float> basis(static_cast<size_t>(n) * steps, 0.0f);
  std::vector<float> alphas(static_cast<size_t>(steps), 0.0f);
  std::vector<float> betas(static_cast<size_t>(steps), 0.0f);
  for (int row = 0; row < n; ++row) {
    basis[static_cast<size_t>(row) * steps] =
        1.0f / std::sqrt(static_cast<float>(n));
  }
  float beta_prev = 0.0f;
  int used = 0;
  for (int j = 0; j < steps; ++j) {
    std::vector<float> q(static_cast<size_t>(n));
    for (int row = 0; row < n; ++row) {
      q[static_cast<size_t>(row)] =
          basis[static_cast<size_t>(row) * steps + j];
    }
    auto w = apply(q);
    if (j > 0) {
      for (int row = 0; row < n; ++row) {
        w[static_cast<size_t>(row)] -=
            beta_prev * basis[static_cast<size_t>(row) * steps + j - 1];
      }
    }
    const float alpha = dot_float(q, w);
    alphas[static_cast<size_t>(j)] = alpha;
    for (int row = 0; row < n; ++row) {
      w[static_cast<size_t>(row)] -= alpha * q[static_cast<size_t>(row)];
    }
    for (int pass = 0; pass < 2; ++pass) {
      for (int col = 0; col <= j; ++col) {
        double coeff = 0.0;
        for (int row = 0; row < n; ++row) {
          coeff += basis[static_cast<size_t>(row) * steps + col] *
                   w[static_cast<size_t>(row)];
        }
        for (int row = 0; row < n; ++row) {
          w[static_cast<size_t>(row)] -=
              static_cast<float>(coeff) *
              basis[static_cast<size_t>(row) * steps + col];
        }
      }
    }
    const float beta = norm_float(w);
    betas[static_cast<size_t>(j)] = beta;
    used = j + 1;
    if (j + 1 == steps || beta <= std::numeric_limits<float>::epsilon()) {
      break;
    }
    for (int row = 0; row < n; ++row) {
      basis[static_cast<size_t>(row) * steps + j + 1] =
          w[static_cast<size_t>(row)] / beta;
    }
    beta_prev = beta;
  }
  std::vector<float> tridiagonal(static_cast<size_t>(used) * used, 0.0f);
  for (int i = 0; i < used; ++i) {
    tridiagonal[static_cast<size_t>(i) * used + i] =
        alphas[static_cast<size_t>(i)];
    if (i > 0) {
      tridiagonal[static_cast<size_t>(i) * used + i - 1] =
          betas[static_cast<size_t>(i - 1)];
      tridiagonal[static_cast<size_t>(i - 1) * used + i] =
          betas[static_cast<size_t>(i - 1)];
    }
  }
  return {tridiagonal, basis, used};
}

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

    csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, x_ptr, ap.data(),
                   n_rows);
    for (int i = 0; i < n_rows; ++i) {
      r[i] = b_ptr[i] - ap[i];
      p[i] = r[i];
    }

    const float norm_b = std::sqrt(std::max(dot_float(r, r), 0.0f));
    double b_norm2 = 0.0;
    for (int i = 0; i < n_rows; ++i) {
      b_norm2 += static_cast<double>(b_ptr[i]) * static_cast<double>(b_ptr[i]);
    }
    const float b_norm = std::sqrt(std::max(b_norm2, 0.0));
    const float tol = std::max(atol, rtol * b_norm);
    float rr = norm_b * norm_b;
    const float eps = std::numeric_limits<float>::epsilon();

    if (norm_b <= tol) {
      *info_ptr = 0;
      *residual_ptr = norm_b;
      *iterations_ptr = 0;
      return;
    }

    int status = maxiter;
    int completed = 0;
    for (int it = 1; it <= maxiter; ++it) {
      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, p.data(), ap.data(),
                     n_rows);
      const float denom = dot_float(p, ap);
      if (std::abs(denom) <= eps) {
        status = -1;
        completed = it - 1;
        break;
      }
      const float alpha = rr / denom;
      for (int i = 0; i < n_rows; ++i) {
        x_ptr[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
      }
      const float rr_new = dot_float(r, r);
      const float r_norm = std::sqrt(std::max(rr_new, 0.0f));
      completed = it;
      if (r_norm <= tol) {
        status = 0;
        rr = rr_new;
        break;
      }
      const float beta = rr_new / rr;
      for (int i = 0; i < n_rows; ++i) {
        p[i] = r[i] + beta * p[i];
      }
      rr = rr_new;
    }

    *info_ptr = status;
    *residual_ptr = std::sqrt(std::max(rr, 0.0f));
    *iterations_ptr = completed;
  });
}

template <typename I>
void csr_lanczos_cpu_impl(const mx::array &data, const mx::array &indices,
                          const mx::array &indptr, const mx::array &v0,
                          mx::array &alphas, mx::array &betas,
                          mx::array &basis, mx::array &actual, int n_rows,
                          int k, bool reorthogonalize, mx::Stream stream) {
  alphas.set_data(mx::allocator::malloc(alphas.nbytes()));
  betas.set_data(mx::allocator::malloc(betas.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(v0);
  encoder.set_output_array(alphas);
  encoder.set_output_array(betas);
  encoder.set_output_array(basis);
  encoder.set_output_array(actual);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    v0 = mx::array::unsafe_weak_copy(v0),
                    alphas = mx::array::unsafe_weak_copy(alphas),
                    betas = mx::array::unsafe_weak_copy(betas),
                    basis = mx::array::unsafe_weak_copy(basis),
                    actual = mx::array::unsafe_weak_copy(actual), n_rows, k,
                    reorthogonalize]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *v0_ptr = v0.data<float>();
    auto *alphas_ptr = alphas.data<float>();
    auto *betas_ptr = betas.data<float>();
    auto *basis_ptr = basis.data<float>();
    auto *actual_ptr = actual.data<int32_t>();

    std::fill(alphas_ptr, alphas_ptr + k, 0.0f);
    std::fill(betas_ptr, betas_ptr + k, 0.0f);
    std::fill(basis_ptr, basis_ptr + static_cast<size_t>(n_rows) * k, 0.0f);

    double v_norm2 = 0.0;
    for (int i = 0; i < n_rows; ++i) {
      v_norm2 += static_cast<double>(v0_ptr[i]) * static_cast<double>(v0_ptr[i]);
    }
    float v_norm = std::sqrt(std::max(v_norm2, 0.0));
    if (v_norm <= std::numeric_limits<float>::epsilon()) {
      for (int i = 0; i < n_rows; ++i) {
        basis_ptr[static_cast<size_t>(i) * k] = i == 0 ? 1.0f : 0.0f;
      }
    } else {
      for (int i = 0; i < n_rows; ++i) {
        basis_ptr[static_cast<size_t>(i) * k] = v0_ptr[i] / v_norm;
      }
    }

    std::vector<float> w(static_cast<size_t>(n_rows));
    std::vector<float> q(static_cast<size_t>(n_rows));
    int used = 0;
    float beta_prev = 0.0f;
    const float eps = std::numeric_limits<float>::epsilon();
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n_rows; ++i) {
        q[i] = basis_ptr[static_cast<size_t>(i) * k + j];
      }
      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, q.data(), w.data(),
                     n_rows);
      if (j > 0) {
        for (int i = 0; i < n_rows; ++i) {
          w[i] -= beta_prev * basis_ptr[static_cast<size_t>(i) * k + j - 1];
        }
      }
      float alpha = dot_column_float(basis_ptr, w.data(), n_rows, k, j);
      alphas_ptr[j] = alpha;
      for (int i = 0; i < n_rows; ++i) {
        w[i] -= alpha * q[i];
      }
      if (reorthogonalize) {
        for (int pass = 0; pass < 2; ++pass) {
          for (int col = 0; col <= j; ++col) {
            double corr = 0.0;
            for (int row = 0; row < n_rows; ++row) {
              corr += basis_ptr[static_cast<size_t>(row) * k + col] * w[row];
            }
            for (int row = 0; row < n_rows; ++row) {
              w[row] -= static_cast<float>(corr) *
                        basis_ptr[static_cast<size_t>(row) * k + col];
            }
          }
        }
      }
      float beta = norm_float(w);
      betas_ptr[j] = beta;
      used = j + 1;
      if (j + 1 == k || beta <= eps) {
        break;
      }
      for (int i = 0; i < n_rows; ++i) {
        basis_ptr[static_cast<size_t>(i) * k + j + 1] = w[i] / beta;
      }
      beta_prev = beta;
    }
    *actual_ptr = used;
  });
}

template <typename I>
void csr_arnoldi_cpu_impl(const mx::array &data, const mx::array &indices,
                          const mx::array &indptr, const mx::array &v0,
                          mx::array &h, mx::array &basis, mx::array &actual,
                          int n_rows, int k, mx::Stream stream) {
  h.set_data(mx::allocator::malloc(h.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(v0);
  encoder.set_output_array(h);
  encoder.set_output_array(basis);
  encoder.set_output_array(actual);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    v0 = mx::array::unsafe_weak_copy(v0),
                    h = mx::array::unsafe_weak_copy(h),
                    basis = mx::array::unsafe_weak_copy(basis),
                    actual = mx::array::unsafe_weak_copy(actual), n_rows,
                    k]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *v0_ptr = v0.data<float>();
    auto *h_ptr = h.data<float>();
    auto *basis_ptr = basis.data<float>();
    auto *actual_ptr = actual.data<int32_t>();

    const int cols = k + 1;
    std::fill(h_ptr, h_ptr + static_cast<size_t>(cols) * k, 0.0f);
    std::fill(basis_ptr, basis_ptr + static_cast<size_t>(n_rows) * cols, 0.0f);

    double v_norm2 = 0.0;
    for (int i = 0; i < n_rows; ++i) {
      v_norm2 += static_cast<double>(v0_ptr[i]) * static_cast<double>(v0_ptr[i]);
    }
    float v_norm = std::sqrt(std::max(v_norm2, 0.0));
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
      for (int pass = 0; pass < 2; ++pass) {
        for (int col = 0; col <= j; ++col) {
          double coeff = 0.0;
          for (int row = 0; row < n_rows; ++row) {
            coeff += basis_ptr[static_cast<size_t>(row) * cols + col] * w[row];
          }
          h_ptr[static_cast<size_t>(col) * k + j] +=
              static_cast<float>(coeff);
          for (int row = 0; row < n_rows; ++row) {
            w[row] -= static_cast<float>(coeff) *
                      basis_ptr[static_cast<size_t>(row) * cols + col];
          }
        }
      }
      float h_next = norm_float(w);
      h_ptr[static_cast<size_t>(j + 1) * k + j] = h_next;
      used = j + 1;
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

void require_linalg_float32(const mx::array &array, const char *name) {
  if (array.dtype() != mx::float32) {
    throw std::invalid_argument(std::string(name) +
                                " currently requires dtype float32.");
  }
}

void require_inner_product_dtype(const mx::array &array, const char *name) {
  if (array.dtype() != mx::float32 && array.dtype() != mx::complex64) {
    throw std::invalid_argument(
        std::string(name) + " requires dtype float32 or complex64.");
  }
}

template <typename T> T sparse_inner_product_value(T lhs, T rhs, bool) {
  return lhs * rhs;
}

template <>
mx::complex64_t sparse_inner_product_value(mx::complex64_t lhs,
                                           mx::complex64_t rhs,
                                           bool conjugate_lhs) {
  const std::complex<float> lhs_value(lhs);
  const std::complex<float> rhs_value(rhs);
  return mx::complex64_t((conjugate_lhs ? std::conj(lhs_value) : lhs_value) *
                         rhs_value);
}

template <typename I>
void csr_triangular_solve_cpu_impl(const mx::array &data,
                                   const mx::array &indices,
                                   const mx::array &indptr,
                                   const mx::array &b, mx::array &x,
                                   int n_rows, bool lower, bool unit_diagonal,
                                   mx::Stream stream) {
  x.set_data(mx::allocator::malloc(x.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(b);
  encoder.set_output_array(x);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    b = mx::array::unsafe_weak_copy(b),
                    x = mx::array::unsafe_weak_copy(x), n_rows, lower,
                    unit_diagonal]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    auto *x_ptr = x.data<float>();

    if (lower) {
      for (int row = 0; row < n_rows; ++row) {
        float sum = b_ptr[row];
        float diag = 1.0f;
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          const int col = static_cast<int>(indices_ptr[p]);
          if (col < row) {
            sum -= data_ptr[p] * x_ptr[col];
          } else if (col == row) {
            diag = data_ptr[p];
          }
        }
        if (!unit_diagonal &&
            std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
          throw std::runtime_error("csr_triangular_solve encountered a zero diagonal.");
        }
        x_ptr[row] = unit_diagonal ? sum : sum / diag;
      }
    } else {
      for (int row = n_rows - 1; row >= 0; --row) {
        float sum = b_ptr[row];
        float diag = 1.0f;
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          const int col = static_cast<int>(indices_ptr[p]);
          if (col > row) {
            sum -= data_ptr[p] * x_ptr[col];
          } else if (col == row) {
            diag = data_ptr[p];
          }
        }
        if (!unit_diagonal &&
            std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
          throw std::runtime_error("csr_triangular_solve encountered a zero diagonal.");
        }
        x_ptr[row] = unit_diagonal ? sum : sum / diag;
      }
    }
  });
}

template <typename T, typename I>
void csr_vdot_cpu_impl(const mx::array &lhs_data, const mx::array &lhs_indices,
                       const mx::array &lhs_indptr, const mx::array &rhs_data,
                       const mx::array &rhs_indices,
                       const mx::array &rhs_indptr, mx::array &out,
                       int n_rows, bool conjugate_lhs, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(lhs_data);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(lhs_indptr);
  encoder.set_input_array(rhs_data);
  encoder.set_input_array(rhs_indices);
  encoder.set_input_array(rhs_indptr);
  encoder.set_output_array(out);

  encoder.dispatch([lhs_data = mx::array::unsafe_weak_copy(lhs_data),
                    lhs_indices = mx::array::unsafe_weak_copy(lhs_indices),
                    lhs_indptr = mx::array::unsafe_weak_copy(lhs_indptr),
                    rhs_data = mx::array::unsafe_weak_copy(rhs_data),
                    rhs_indices = mx::array::unsafe_weak_copy(rhs_indices),
                    rhs_indptr = mx::array::unsafe_weak_copy(rhs_indptr),
                    out = mx::array::unsafe_weak_copy(out),
                    n_rows, conjugate_lhs]() mutable {
    const auto *lhs_data_ptr = lhs_data.data<T>();
    const auto *lhs_indices_ptr = lhs_indices.data<I>();
    const auto *lhs_indptr_ptr = lhs_indptr.data<I>();
    const auto *rhs_data_ptr = rhs_data.data<T>();
    const auto *rhs_indices_ptr = rhs_indices.data<I>();
    const auto *rhs_indptr_ptr = rhs_indptr.data<I>();
    using AccT = std::conditional_t<std::is_same_v<T, mx::complex64_t>,
                                    std::complex<double>, double>;
    AccT acc{};
    for (int row = 0; row < n_rows; ++row) {
      I lp = lhs_indptr_ptr[row];
      I rp = rhs_indptr_ptr[row];
      const I lend = lhs_indptr_ptr[row + 1];
      const I rend = rhs_indptr_ptr[row + 1];
      while (lp < lend && rp < rend) {
        const I lc = lhs_indices_ptr[lp];
        const I rc = rhs_indices_ptr[rp];
      if (lc == rc) {
          if constexpr (std::is_same_v<T, mx::complex64_t>) {
            const std::complex<float> lhs_value(lhs_data_ptr[lp]);
            const std::complex<float> rhs_value(rhs_data_ptr[rp]);
            acc += static_cast<std::complex<double>>(
                conjugate_lhs ? std::conj(lhs_value) * rhs_value
                              : lhs_value * rhs_value);
          } else {
            acc += static_cast<double>(sparse_inner_product_value<T>(
                lhs_data_ptr[lp], rhs_data_ptr[rp], conjugate_lhs));
          }
          ++lp;
          ++rp;
        } else if (lc < rc) {
          ++lp;
        } else {
          ++rp;
        }
      }
    }
    if constexpr (std::is_same_v<T, mx::complex64_t>) {
      *out.data<T>() =
          mx::complex64_t(static_cast<float>(acc.real()),
                          static_cast<float>(acc.imag()));
    } else {
      *out.data<T>() = static_cast<T>(acc);
    }
  });
}

void csr_permute_vector_cpu_impl(const mx::array &x, const mx::array &perm,
                                 mx::array &out, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(x);
  encoder.set_input_array(perm);
  encoder.set_output_array(out);

  encoder.dispatch([x = mx::array::unsafe_weak_copy(x),
                    perm = mx::array::unsafe_weak_copy(perm),
                    out = mx::array::unsafe_weak_copy(out)]() mutable {
    const auto *x_ptr = x.data<float>();
    const auto *perm_ptr = perm.data<int32_t>();
    auto *out_ptr = out.data<float>();
    for (size_t i = 0; i < out.size(); ++i) {
      out_ptr[i] = x_ptr[perm_ptr[i]];
    }
  });
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array>
make_csr_arrays_float32(const std::vector<float> &data,
                        const std::vector<I> &indices,
                        const std::vector<I> &indptr, mx::Dtype index_dtype) {
  return {mx::array(data.begin(),
                    mx::Shape{static_cast<int>(data.size())}, mx::float32),
          mx::array(indices.begin(),
                    mx::Shape{static_cast<int>(indices.size())}, index_dtype),
          mx::array(indptr.begin(),
                    mx::Shape{static_cast<int>(indptr.size())}, index_dtype)};
}

template <typename I>
std::vector<std::map<int, float>>
read_csr_rows_float32(mx::array data, mx::array indices, mx::array indptr,
                      int n_rows) {
  data.eval();
  indices.eval();
  indptr.eval();
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  std::vector<std::map<int, float>> rows(static_cast<size_t>(n_rows));
  for (int row = 0; row < n_rows; ++row) {
    for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
      rows[static_cast<size_t>(row)][static_cast<int>(indices_ptr[p])] +=
          data_ptr[p];
    }
  }
  return rows;
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array>
csr_cholesky_impl(mx::array data, mx::array indices, mx::array indptr,
                  int n_rows, int n_cols, mx::Dtype index_dtype) {
  auto input_rows =
      read_csr_rows_float32<I>(std::move(data), std::move(indices),
                               std::move(indptr), n_rows);
  std::vector<std::map<int, float>> lower(static_cast<size_t>(n_rows));
  for (int row = 0; row < n_rows; ++row) {
    for (const auto &[col, value] : input_rows[static_cast<size_t>(row)]) {
      if (col < 0 || col >= n_cols) {
        throw std::invalid_argument("csr_cholesky input contains an out-of-bounds column.");
      }
      if (row >= col) {
        lower[static_cast<size_t>(row)][col] += value;
      }
    }
  }
  for (int row = 0; row < n_rows; ++row) {
    for (const auto &[col, value] : input_rows[static_cast<size_t>(row)]) {
      if (row < col && lower[static_cast<size_t>(col)].count(row) == 0) {
        lower[static_cast<size_t>(col)][row] = value;
      }
    }
  }

  std::vector<std::vector<std::pair<int, float>>> columns(
      static_cast<size_t>(n_rows));
  std::vector<float> diag(static_cast<size_t>(n_rows), 0.0f);
  const float eps = std::numeric_limits<float>::epsilon();

  for (int row = 0; row < n_rows; ++row) {
    auto &current = lower[static_cast<size_t>(row)];
    current.try_emplace(row, 0.0f);
    for (auto it = current.begin(); it != current.lower_bound(row); ++it) {
      const int pivot_col = it->first;
      if (std::abs(diag[static_cast<size_t>(pivot_col)]) <= eps) {
        throw std::runtime_error("csr_cholesky encountered a zero pivot.");
      }
      const float factor =
          it->second / diag[static_cast<size_t>(pivot_col)];
      it->second = factor;
      for (const auto &[update_col, update_value] :
           columns[static_cast<size_t>(pivot_col)]) {
        if (update_col < row) {
          current[update_col] -= factor * update_value;
        }
      }
      current[row] -= factor * factor;
      columns[static_cast<size_t>(pivot_col)].push_back({row, factor});
    }
    const float diag_value = current[row];
    if (diag_value <= eps) {
      throw std::runtime_error("csr_cholesky requires a positive-definite matrix.");
    }
    diag[static_cast<size_t>(row)] = std::sqrt(diag_value);
    current[row] = diag[static_cast<size_t>(row)];
  }

  std::vector<float> out_data;
  std::vector<I> out_indices;
  std::vector<I> out_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  for (int row = 0; row < n_rows; ++row) {
    for (const auto &[col, value] : lower[static_cast<size_t>(row)]) {
      if (col <= row && std::abs(value) > eps) {
        out_data.push_back(value);
        out_indices.push_back(static_cast<I>(col));
      }
    }
    out_indptr[static_cast<size_t>(row) + 1] =
        static_cast<I>(out_data.size());
  }
  return make_csr_arrays_float32(out_data, out_indices, out_indptr,
                                 index_dtype);
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array,
           mx::array>
csr_lu_impl(mx::array data, mx::array indices, mx::array indptr, int n_rows,
            int n_cols, mx::Dtype index_dtype) {
  if (n_rows != n_cols) {
    throw std::invalid_argument("csr_lu requires a square matrix.");
  }
  auto rows = read_csr_rows_float32<I>(std::move(data), std::move(indices),
                                       std::move(indptr), n_rows);
  std::vector<std::map<int, float>> L(static_cast<size_t>(n_rows));
  std::vector<std::map<int, float>> U(static_cast<size_t>(n_rows));
  std::vector<int32_t> perm(static_cast<size_t>(n_rows));
  std::iota(perm.begin(), perm.end(), 0);
  const float eps = std::numeric_limits<float>::epsilon();

  for (int k = 0; k < n_rows; ++k) {
    int pivot_row = k;
    float pivot_abs = 0.0f;
    for (int row = k; row < n_rows; ++row) {
      auto found = rows[static_cast<size_t>(row)].find(k);
      const float value = found == rows[static_cast<size_t>(row)].end()
                              ? 0.0f
                              : found->second;
      if (std::abs(value) > pivot_abs) {
        pivot_abs = std::abs(value);
        pivot_row = row;
      }
    }
    if (pivot_abs <= eps) {
      throw std::runtime_error("csr_lu encountered a structurally singular pivot.");
    }
    if (pivot_row != k) {
      std::swap(rows[static_cast<size_t>(pivot_row)],
                rows[static_cast<size_t>(k)]);
      std::swap(perm[static_cast<size_t>(pivot_row)],
                perm[static_cast<size_t>(k)]);
      for (int col = 0; col < k; ++col) {
        std::swap(L[static_cast<size_t>(pivot_row)][col],
                  L[static_cast<size_t>(k)][col]);
      }
    }

    L[static_cast<size_t>(k)][k] = 1.0f;
    for (const auto &[col, value] : rows[static_cast<size_t>(k)]) {
      if (col >= k && std::abs(value) > eps) {
        U[static_cast<size_t>(k)][col] = value;
      }
    }
    const float pivot = U[static_cast<size_t>(k)][k];
    for (int row = k + 1; row < n_rows; ++row) {
      auto entry = rows[static_cast<size_t>(row)].find(k);
      if (entry == rows[static_cast<size_t>(row)].end() ||
          std::abs(entry->second) <= eps) {
        continue;
      }
      const float factor = entry->second / pivot;
      L[static_cast<size_t>(row)][k] = factor;
      rows[static_cast<size_t>(row)].erase(entry);
      for (const auto &[col, upper_value] : U[static_cast<size_t>(k)]) {
        if (col > k) {
          auto &slot = rows[static_cast<size_t>(row)][col];
          slot -= factor * upper_value;
          if (std::abs(slot) <= eps) {
            rows[static_cast<size_t>(row)].erase(col);
          }
        }
      }
    }
  }

  std::vector<float> l_data;
  std::vector<I> l_indices;
  std::vector<I> l_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  std::vector<float> u_data;
  std::vector<I> u_indices;
  std::vector<I> u_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  for (int row = 0; row < n_rows; ++row) {
    for (const auto &[col, value] : L[static_cast<size_t>(row)]) {
      if (col <= row && std::abs(value) > eps) {
        l_data.push_back(value);
        l_indices.push_back(static_cast<I>(col));
      }
    }
    l_indptr[static_cast<size_t>(row) + 1] =
        static_cast<I>(l_data.size());
    for (const auto &[col, value] : U[static_cast<size_t>(row)]) {
      if (col >= row && std::abs(value) > eps) {
        u_data.push_back(value);
        u_indices.push_back(static_cast<I>(col));
      }
    }
    u_indptr[static_cast<size_t>(row) + 1] =
        static_cast<I>(u_data.size());
  }

  auto permutation =
      mx::array(perm.begin(), mx::Shape{static_cast<int>(perm.size())},
                mx::int32);
  auto [l_data_array, l_indices_array, l_indptr_array] =
      make_csr_arrays_float32(l_data, l_indices, l_indptr, index_dtype);
  auto [u_data_array, u_indices_array, u_indptr_array] =
      make_csr_arrays_float32(u_data, u_indices, u_indptr, index_dtype);
  return {permutation,   l_data_array,   l_indices_array, l_indptr_array,
          u_data_array, u_indices_array, u_indptr_array};
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres_impl(mx::array data, mx::array indices, mx::array indptr, mx::array b,
               mx::array x0, int n_rows, float rtol, float atol, int restart,
               int maxiter) {
  data.eval();
  indices.eval();
  indptr.eval();
  b.eval();
  x0.eval();
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  const auto *b_ptr = b.data<float>();
  const auto *x0_ptr = x0.data<float>();

  std::vector<float> x(x0_ptr, x0_ptr + n_rows);
  std::vector<float> rhs(b_ptr, b_ptr + n_rows);
  const float b_norm = norm_float(rhs);
  const float tolerance = std::max(atol, rtol * b_norm);
  int iterations = 0;
  int status = maxiter;
  float residual_norm = std::numeric_limits<float>::infinity();

  auto stream = mx::default_stream(mx::default_device());

  while (iterations < maxiter) {
    auto ax = host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, x, n_rows);
    std::vector<float> r(static_cast<size_t>(n_rows));
    for (int i = 0; i < n_rows; ++i) {
      r[static_cast<size_t>(i)] =
          rhs[static_cast<size_t>(i)] - ax[static_cast<size_t>(i)];
    }
    const float beta = norm_float(r);
    residual_norm = beta;
    if (beta <= tolerance) {
      status = 0;
      break;
    }

    const int steps = std::min({restart, maxiter - iterations, n_rows});
    std::vector<float> v0_data(static_cast<size_t>(n_rows));
    for (int i = 0; i < n_rows; ++i)
      v0_data[static_cast<size_t>(i)] = r[static_cast<size_t>(i)] / beta;
    auto v0 = mx::array(v0_data.begin(), mx::Shape{n_rows}, mx::float32);

    // Arnoldi factorisation via GPU kernel (falls back to CPU if no GPU device)
    auto [h_mx, basis_mx, actual_k_mx] =
        csr_arnoldi(data, indices, indptr, v0, n_rows, n_rows, steps, stream);
    mx::eval(h_mx, basis_mx, actual_k_mx);

    const int used = static_cast<int>(actual_k_mx.item<int32_t>());
    const float *h_ptr = h_mx.data<float>();
    const float *basis_ptr = basis_mx.data<float>();

    // Build (used+1)×used Hessenberg from the (steps+1)×steps output
    std::vector<double> h_used(static_cast<size_t>(used + 1) * used, 0.0);
    for (int row = 0; row < used + 1; ++row) {
      for (int col = 0; col < used; ++col) {
        h_used[static_cast<size_t>(row) * used + col] =
            h_ptr[static_cast<size_t>(row) * steps + col];
      }
    }
    std::vector<double> e1(static_cast<size_t>(used + 1), 0.0);
    e1[0] = beta;
    auto y = least_squares_normal_equations(h_used, e1, used + 1, used);

    // x += V[:,0:used] * y  (basis has shape (n_rows, steps+1))
    for (int row = 0; row < n_rows; ++row) {
      double update = 0.0;
      for (int col = 0; col < used; ++col) {
        update +=
            basis_ptr[static_cast<size_t>(row) * (steps + 1) + col] *
            y[static_cast<size_t>(col)];
      }
      x[static_cast<size_t>(row)] += static_cast<float>(update);
    }
    iterations += used;
    if (used == 0) {
      status = -1;
      break;
    }
  }

  mx::array x_out(x.begin(), mx::Shape{n_rows}, mx::float32);
  mx::array info(status, mx::int32);
  mx::array residual(residual_norm, mx::float32);
  mx::array iters(iterations, mx::int32);
  return {x_out, info, residual, iters};
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_minres_impl(mx::array data, mx::array indices, mx::array indptr, mx::array b,
                mx::array x0, int n_rows, float rtol, float atol,
                int maxiter) {
  data.eval();
  indices.eval();
  indptr.eval();
  b.eval();
  x0.eval();
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  const auto *b_ptr = b.data<float>();
  const auto *x0_ptr = x0.data<float>();

  std::vector<float> x_base(x0_ptr, x0_ptr + n_rows);
  std::vector<float> rhs(b_ptr, b_ptr + n_rows);
  auto ax = host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, x_base, n_rows);
  std::vector<float> r0(static_cast<size_t>(n_rows));
  for (int i = 0; i < n_rows; ++i) {
    r0[static_cast<size_t>(i)] =
        rhs[static_cast<size_t>(i)] - ax[static_cast<size_t>(i)];
  }
  const float beta0 = norm_float(r0);
  const float tolerance = std::max(atol, rtol * norm_float(rhs));
  if (beta0 <= tolerance) {
    return {mx::array(x_base.begin(), mx::Shape{n_rows}, mx::float32),
            mx::array(0, mx::int32), mx::array(beta0, mx::float32),
            mx::array(0, mx::int32)};
  }

  const int steps = std::min(maxiter, n_rows);
  std::vector<float> v0_data(static_cast<size_t>(n_rows));
  for (int i = 0; i < n_rows; ++i)
    v0_data[static_cast<size_t>(i)] = r0[static_cast<size_t>(i)] / beta0;
  auto v0 = mx::array(v0_data.begin(), mx::Shape{n_rows}, mx::float32);
  auto stream = mx::default_stream(mx::default_device());

  // Lanczos tridiagonalisation via GPU kernel
  auto [alphas_mx, betas_mx, basis_mx, actual_k_mx] =
      csr_lanczos(data, indices, indptr, v0, n_rows, n_rows, steps, true,
                  stream);
  mx::eval(alphas_mx, betas_mx, basis_mx, actual_k_mx);

  const int used = static_cast<int>(actual_k_mx.item<int32_t>());
  const float *alphas_ptr = alphas_mx.data<float>();
  const float *betas_ptr = betas_mx.data<float>();
  const float *basis_ptr = basis_mx.data<float>();

  // Build extended (used+1)×used tridiagonal for the least-squares problem
  std::vector<double> tbar(static_cast<size_t>(used + 1) * used, 0.0);
  for (int j = 0; j < used; ++j) {
    tbar[static_cast<size_t>(j) * used + j] = alphas_ptr[j];
    if (j > 0) {
      tbar[static_cast<size_t>(j) * used + j - 1] = betas_ptr[j - 1];
      tbar[static_cast<size_t>(j - 1) * used + j] = betas_ptr[j - 1];
    }
    tbar[static_cast<size_t>(j + 1) * used + j] = betas_ptr[j];
  }
  std::vector<double> rhs_small(static_cast<size_t>(used + 1), 0.0);
  rhs_small[0] = beta0;
  auto y = least_squares_normal_equations(tbar, rhs_small, used + 1, used);

  // x += V * y  (basis has shape (n_rows, steps))
  std::vector<float> x = x_base;
  for (int row = 0; row < n_rows; ++row) {
    double update = 0.0;
    for (int col = 0; col < used; ++col) {
      update += basis_ptr[static_cast<size_t>(row) * steps + col] *
                y[static_cast<size_t>(col)];
    }
    x[static_cast<size_t>(row)] += static_cast<float>(update);
  }
  auto final_ax = host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, x, n_rows);
  std::vector<float> final_r(static_cast<size_t>(n_rows));
  for (int i = 0; i < n_rows; ++i) {
    final_r[static_cast<size_t>(i)] =
        rhs[static_cast<size_t>(i)] - final_ax[static_cast<size_t>(i)];
  }
  const float residual_norm = norm_float(final_r);
  const int status = residual_norm <= tolerance ? 0 : maxiter;
  return {mx::array(x.begin(), mx::Shape{n_rows}, mx::float32),
          mx::array(status, mx::int32), mx::array(residual_norm, mx::float32),
          mx::array(used, mx::int32)};
}

template <typename I>
std::tuple<mx::array, mx::array>
csr_eigsh_impl(mx::array data, mx::array indices, mx::array indptr, int n_rows,
               int k, int ncv, const std::string &which) {
  const int steps = std::min(n_rows, std::max(ncv, k + 1));
  std::vector<float> v0_data(static_cast<size_t>(n_rows),
                             1.0f / std::sqrt(static_cast<float>(n_rows)));
  auto v0 = mx::array(v0_data.begin(), mx::Shape{n_rows}, mx::float32);
  auto stream = mx::default_stream(mx::default_device());

  // Lanczos tridiagonalisation via GPU kernel (falls back to CPU if no GPU device)
  auto [alphas_mx, betas_mx, basis_mx, actual_k_mx] =
      csr_lanczos(data, indices, indptr, v0, n_rows, n_rows, steps, true,
                  stream);
  mx::eval(alphas_mx, betas_mx, basis_mx, actual_k_mx);

  const int used = static_cast<int>(actual_k_mx.item<int32_t>());
  const float *alphas_ptr = alphas_mx.data<float>();
  const float *betas_ptr = betas_mx.data<float>();
  const float *basis_ptr = basis_mx.data<float>();

  // Build used×used symmetric tridiagonal matrix
  std::vector<float> tridiagonal(static_cast<size_t>(used) * used, 0.0f);
  for (int i = 0; i < used; ++i) {
    tridiagonal[static_cast<size_t>(i) * used + i] = alphas_ptr[i];
    if (i > 0) {
      tridiagonal[static_cast<size_t>(i) * used + i - 1] = betas_ptr[i - 1];
      tridiagonal[static_cast<size_t>(i - 1) * used + i] = betas_ptr[i - 1];
    }
  }
  auto [values_all, vectors_small] = jacobi_symmetric(tridiagonal, used);
  auto selected = select_ritz_indices(values_all, k, which);

  // Back-transform Ritz vectors: eigvec = basis * vectors_small[:,eig_col]
  std::vector<float> values(static_cast<size_t>(k), 0.0f);
  std::vector<float> vectors(static_cast<size_t>(n_rows) * k, 0.0f);
  for (int out_col = 0; out_col < k; ++out_col) {
    const int eig_col = selected[static_cast<size_t>(out_col)];
    values[static_cast<size_t>(out_col)] =
        values_all[static_cast<size_t>(eig_col)];
    for (int row = 0; row < n_rows; ++row) {
      double acc = 0.0;
      for (int j = 0; j < used; ++j) {
        acc += basis_ptr[static_cast<size_t>(row) * steps + j] *
               vectors_small[static_cast<size_t>(j) * used + eig_col];
      }
      vectors[static_cast<size_t>(row) * k + out_col] =
          static_cast<float>(acc);
    }
  }
  return {mx::array(values.begin(), mx::Shape{k}, mx::float32),
          mx::array(vectors.begin(), mx::Shape{n_rows, k}, mx::float32)};
}

std::vector<float> qr_eigenvalues_real(std::vector<float> h, int n) {
  std::vector<float> q(static_cast<size_t>(n) * n, 0.0f);
  std::vector<float> r(static_cast<size_t>(n) * n, 0.0f);
  for (int sweep = 0; sweep < std::max(64, 64 * n); ++sweep) {
    std::fill(q.begin(), q.end(), 0.0f);
    std::fill(r.begin(), r.end(), 0.0f);
    for (int col = 0; col < n; ++col) {
      std::vector<float> v(static_cast<size_t>(n));
      for (int row = 0; row < n; ++row) {
        v[static_cast<size_t>(row)] = h[static_cast<size_t>(row) * n + col];
      }
      for (int prev = 0; prev < col; ++prev) {
        double coeff = 0.0;
        for (int row = 0; row < n; ++row) {
          coeff += q[static_cast<size_t>(row) * n + prev] * v[static_cast<size_t>(row)];
        }
        r[static_cast<size_t>(prev) * n + col] = static_cast<float>(coeff);
        for (int row = 0; row < n; ++row) {
          v[static_cast<size_t>(row)] -=
              static_cast<float>(coeff) * q[static_cast<size_t>(row) * n + prev];
        }
      }
      const float v_norm = norm_float(v);
      if (v_norm <= std::numeric_limits<float>::epsilon()) {
        q[static_cast<size_t>(col) * n + col] = 1.0f;
      } else {
        r[static_cast<size_t>(col) * n + col] = v_norm;
        for (int row = 0; row < n; ++row) {
          q[static_cast<size_t>(row) * n + col] =
              v[static_cast<size_t>(row)] / v_norm;
        }
      }
    }
    std::vector<float> next(static_cast<size_t>(n) * n, 0.0f);
    for (int row = 0; row < n; ++row) {
      for (int col = 0; col < n; ++col) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
          acc += r[static_cast<size_t>(row) * n + j] *
                 q[static_cast<size_t>(j) * n + col];
        }
        next[static_cast<size_t>(row) * n + col] = static_cast<float>(acc);
      }
    }
    h.swap(next);
  }
  std::vector<float> values(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    values[static_cast<size_t>(i)] = h[static_cast<size_t>(i) * n + i];
  }
  return values;
}

template <typename I>
std::tuple<mx::array, mx::array>
csr_eigs_impl(mx::array data, mx::array indices, mx::array indptr, int n_rows,
              int k, int ncv, const std::string &which) {
  const int steps = std::min(n_rows, std::max(ncv, k + 1));
  std::vector<float> v0_data(static_cast<size_t>(n_rows),
                             1.0f / std::sqrt(static_cast<float>(n_rows)));
  auto v0 = mx::array(v0_data.begin(), mx::Shape{n_rows}, mx::float32);
  auto stream = mx::default_stream(mx::default_device());

  // Arnoldi factorisation via GPU kernel (falls back to CPU if no GPU device)
  auto [h_mx, basis_mx, actual_k_mx] =
      csr_arnoldi(data, indices, indptr, v0, n_rows, n_rows, steps, stream);
  mx::eval(h_mx, basis_mx, actual_k_mx);

  const int used = static_cast<int>(actual_k_mx.item<int32_t>());
  const float *h_ptr = h_mx.data<float>();
  const float *basis_ptr = basis_mx.data<float>();

  // Extract used×used sub-Hessenberg (H has shape (steps+1, steps))
  std::vector<float> h_square(static_cast<size_t>(used) * used, 0.0f);
  for (int row = 0; row < used; ++row) {
    for (int col = 0; col < used; ++col) {
      h_square[static_cast<size_t>(row) * used + col] =
          h_ptr[static_cast<size_t>(row) * steps + col];
    }
  }
  auto values_all = qr_eigenvalues_real(h_square, used);
  auto selected = select_ritz_indices(values_all, k, which);

  // Ritz vectors are the corresponding Krylov basis vectors
  std::vector<float> values(static_cast<size_t>(k), 0.0f);
  std::vector<float> vectors(static_cast<size_t>(n_rows) * k, 0.0f);
  for (int out_col = 0; out_col < k; ++out_col) {
    const int eig_col = selected[static_cast<size_t>(out_col)];
    values[static_cast<size_t>(out_col)] =
        values_all[static_cast<size_t>(eig_col)];
    for (int row = 0; row < n_rows; ++row) {
      vectors[static_cast<size_t>(row) * k + out_col] =
          basis_ptr[static_cast<size_t>(row) * (steps + 1) + (eig_col % used)];
    }
  }
  return {mx::array(values.begin(), mx::Shape{k}, mx::float32),
          mx::array(vectors.begin(), mx::Shape{n_rows, k}, mx::float32)};
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array>
csr_svds_impl(mx::array data, mx::array indices, mx::array indptr, int n_rows,
              int n_cols, int k, int ncv, const std::string &which) {
  data.eval();
  indices.eval();
  indptr.eval();
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  const int steps = std::min(n_cols, std::max(ncv, k + 1));
  auto [tridiagonal, basis, used] = host_lanczos_operator(
      n_cols, steps, [&](const std::vector<float> &x) {
        auto ax = host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, x, n_rows);
        return host_csr_spmv_transpose(data_ptr, indices_ptr, indptr_ptr, ax,
                                       n_rows, n_cols);
      });
  auto [evals_all, vecs_small] = jacobi_symmetric(tridiagonal, used);
  auto selected = select_ritz_indices(evals_all, k, which);
  std::vector<float> singular(static_cast<size_t>(k), 0.0f);
  std::vector<float> right(static_cast<size_t>(n_cols) * k, 0.0f);
  std::vector<float> left(static_cast<size_t>(n_rows) * k, 0.0f);
  for (int out_col = 0; out_col < k; ++out_col) {
    const int eig_col = selected[static_cast<size_t>(out_col)];
    const float sigma =
        std::sqrt(std::max(evals_all[static_cast<size_t>(eig_col)], 0.0f));
    singular[static_cast<size_t>(out_col)] = sigma;
    std::vector<float> v(static_cast<size_t>(n_cols), 0.0f);
    for (int row = 0; row < n_cols; ++row) {
      double acc = 0.0;
      for (int j = 0; j < used; ++j) {
        acc += basis[static_cast<size_t>(row) * steps + j] *
               vecs_small[static_cast<size_t>(j) * used + eig_col];
      }
      v[static_cast<size_t>(row)] = static_cast<float>(acc);
      right[static_cast<size_t>(out_col) * n_cols + row] =
          static_cast<float>(acc);
    }
    auto av = host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, v, n_rows);
    for (int row = 0; row < n_rows; ++row) {
      left[static_cast<size_t>(row) * k + out_col] =
          sigma <= std::numeric_limits<float>::epsilon()
              ? 0.0f
              : av[static_cast<size_t>(row)] / sigma;
    }
  }
  return {mx::array(left.begin(), mx::Shape{n_rows, k}, mx::float32),
          mx::array(singular.begin(), mx::Shape{k}, mx::float32),
          mx::array(right.begin(), mx::Shape{k, n_cols}, mx::float32)};
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
                             outputs[1], outputs[2], outputs[3], n_rows_,
                             rtol_, atol_, maxiter_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_cg_cpu_impl<int64_t>(data, indices, indptr, b, x0, outputs[0],
                             outputs[1], outputs[2], outputs[3], n_rows_,
                             rtol_, atol_, maxiter_, stream());
    return;
  }
  throw std::runtime_error("csr_cg requires int32 or int64 indices.");
}

void CSRLanczos::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &v0 = inputs[3];

  if (indices.dtype() == mx::int32) {
    csr_lanczos_cpu_impl<int32_t>(data, indices, indptr, v0, outputs[0],
                                  outputs[1], outputs[2], outputs[3], n_rows_,
                                  k_, reorthogonalize_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_lanczos_cpu_impl<int64_t>(data, indices, indptr, v0, outputs[0],
                                  outputs[1], outputs[2], outputs[3], n_rows_,
                                  k_, reorthogonalize_, stream());
    return;
  }
  throw std::runtime_error("csr_lanczos requires int32 or int64 indices.");
}

void CSRArnoldi::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &v0 = inputs[3];

  if (indices.dtype() == mx::int32) {
    csr_arnoldi_cpu_impl<int32_t>(data, indices, indptr, v0, outputs[0],
                                  outputs[1], outputs[2], n_rows_, k_,
                                  stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_arnoldi_cpu_impl<int64_t>(data, indices, indptr, v0, outputs[0],
                                  outputs[1], outputs[2], n_rows_, k_,
                                  stream());
    return;
  }
  throw std::runtime_error("csr_arnoldi requires int32 or int64 indices.");
}

void CSRTriangularSolve::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];

  if (indices.dtype() == mx::int32) {
    csr_triangular_solve_cpu_impl<int32_t>(
        data, indices, indptr, b, outputs[0], n_rows_, lower_, unit_diagonal_,
        stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_triangular_solve_cpu_impl<int64_t>(
        data, indices, indptr, b, outputs[0], n_rows_, lower_, unit_diagonal_,
        stream());
    return;
  }
  throw std::runtime_error(
      "csr_triangular_solve requires int32 or int64 indices.");
}

void CSRVdot::eval_cpu(const std::vector<mx::array> &inputs,
                       std::vector<mx::array> &outputs) {
  auto &lhs_data = inputs[0];
  auto &lhs_indices = inputs[1];
  auto &lhs_indptr = inputs[2];
  auto &rhs_data = inputs[3];
  auto &rhs_indices = inputs[4];
  auto &rhs_indptr = inputs[5];

  if (lhs_data.dtype() == mx::float32 && lhs_indices.dtype() == mx::int32) {
    csr_vdot_cpu_impl<float, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        outputs[0], n_rows_, conjugate_lhs_, stream());
    return;
  }
  if (lhs_data.dtype() == mx::float32 && lhs_indices.dtype() == mx::int64) {
    csr_vdot_cpu_impl<float, int64_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        outputs[0], n_rows_, conjugate_lhs_, stream());
    return;
  }
  if (lhs_data.dtype() == mx::complex64 && lhs_indices.dtype() == mx::int32) {
    csr_vdot_cpu_impl<mx::complex64_t, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        outputs[0], n_rows_, conjugate_lhs_, stream());
    return;
  }
  if (lhs_data.dtype() == mx::complex64 && lhs_indices.dtype() == mx::int64) {
    csr_vdot_cpu_impl<mx::complex64_t, int64_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        outputs[0], n_rows_, conjugate_lhs_, stream());
    return;
  }
  throw std::runtime_error(
      "csr_vdot requires float32 or complex64 data with int32 or int64 indices.");
}

void CSRPermuteVector::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  csr_permute_vector_cpu_impl(inputs[0], inputs[1], outputs[0], stream());
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
  mx::array work(mx::allocator::malloc(static_cast<size_t>(3 * n_rows_) *
                                       sizeof(float)),
                 mx::Shape{3 * n_rows_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name = sparse_kernel_name("csr_cg", data.dtype(), indices.dtype());
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

void CSRLanczos::eval_gpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &v0 = inputs[3];
  auto &alphas = outputs[0];
  auto &betas = outputs[1];
  auto &basis = outputs[2];
  auto &actual = outputs[3];

  alphas.set_data(mx::allocator::malloc(alphas.nbytes()));
  betas.set_data(mx::allocator::malloc(betas.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));
  mx::array work(mx::allocator::malloc(static_cast<size_t>(n_rows_) *
                                       sizeof(float)),
                 mx::Shape{n_rows_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_lanczos", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(v0, 3);
  encoder.set_output_array(alphas, 4);
  encoder.set_output_array(betas, 5);
  encoder.set_output_array(basis, 6);
  encoder.set_output_array(actual, 7);
  encoder.set_output_array(work, 8);
  encoder.set_bytes(n_rows_, 9);
  encoder.set_bytes(n_cols_, 10);
  encoder.set_bytes(k_, 11);
  int reorth = reorthogonalize_ ? 1 : 0;
  encoder.set_bytes(reorth, 12);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(work));
}

void CSRArnoldi::eval_gpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &v0 = inputs[3];
  auto &h = outputs[0];
  auto &basis = outputs[1];
  auto &actual = outputs[2];

  h.set_data(mx::allocator::malloc(h.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));
  mx::array work(mx::allocator::malloc(static_cast<size_t>(n_rows_) *
                                       sizeof(float)),
                 mx::Shape{n_rows_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_arnoldi", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(v0, 3);
  encoder.set_output_array(h, 4);
  encoder.set_output_array(basis, 5);
  encoder.set_output_array(actual, 6);
  encoder.set_output_array(work, 7);
  encoder.set_bytes(n_rows_, 8);
  encoder.set_bytes(n_cols_, 9);
  encoder.set_bytes(k_, 10);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(work));
}

void CSRTriangularSolve::eval_gpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x = outputs[0];

  x.set_data(mx::allocator::malloc(x.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_triangular_solve", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(b, 3);
  encoder.set_output_array(x, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(n_cols_, 6);
  int lower = lower_ ? 1 : 0;
  int unit_diagonal = unit_diagonal_ ? 1 : 0;
  encoder.set_bytes(lower, 7);
  encoder.set_bytes(unit_diagonal, 8);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

void CSRVdot::eval_gpu(const std::vector<mx::array> &inputs,
                       std::vector<mx::array> &outputs) {
  auto &lhs_data = inputs[0];
  auto &lhs_indices = inputs[1];
  auto &lhs_indptr = inputs[2];
  auto &rhs_data = inputs[3];
  auto &rhs_indices = inputs[4];
  auto &rhs_indptr = inputs[5];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name(conjugate_lhs_ ? "csr_vdot" : "csr_dot",
                         lhs_data.dtype(), lhs_indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(lhs_data, 0);
  encoder.set_input_array(lhs_indices, 1);
  encoder.set_input_array(lhs_indptr, 2);
  encoder.set_input_array(rhs_data, 3);
  encoder.set_input_array(rhs_indices, 4);
  encoder.set_input_array(rhs_indptr, 5);
  encoder.set_output_array(out, 6);
  encoder.set_bytes(n_rows_, 7);
  encoder.set_bytes(n_cols_, 8);
  encoder.dispatch_threads(MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
}

void CSRPermuteVector::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &x = inputs[0];
  auto &perm = inputs[1];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel("csr_permute_vector_float32", lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(perm, 1);
  encoder.set_output_array(out, 2);
  int size = static_cast<int>(out.size());
  encoder.set_bytes(size, 3);
  auto threads = std::max<size_t>(out.size(), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRCG::eval_gpu(const std::vector<mx::array> &,
                     std::vector<mx::array> &) {
  throw std::runtime_error("csr_cg has no GPU implementation in this build.");
}

void CSRLanczos::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_lanczos has no GPU implementation in this build.");
}

void CSRArnoldi::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_arnoldi has no GPU implementation in this build.");
}

void CSRTriangularSolve::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_triangular_solve has no GPU implementation in this build.");
}

void CSRVdot::eval_gpu(const std::vector<mx::array> &,
                       std::vector<mx::array> &) {
  throw std::runtime_error("csr_vdot has no GPU implementation in this build.");
}

void CSRPermuteVector::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_permute_vector has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_cg(const mx::array &data, const mx::array &indices,
       const mx::array &indptr, const mx::array &b, const mx::array &x0,
       int n_rows, int n_cols, float rtol, float atol, int maxiter,
       mx::StreamOrDevice s) {
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
  require_same_index_dtype(indices, indptr, "csr_cg indices",
                           "csr_cg indptr");
  require_size(indptr, n_rows + 1, "csr_cg indptr");
  require_size(b, n_rows, "csr_cg b");
  require_size(x0, n_cols, "csr_cg x0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument("csr_cg data and indices must have equal length.");
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

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres(const mx::array &data, const mx::array &indices,
          const mx::array &indptr, const mx::array &b, const mx::array &x0,
          int n_rows, int n_cols, float rtol, float atol, int restart,
          int maxiter) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_gmres requires a non-empty square matrix.");
  }
  if (restart <= 0 || maxiter < 0) {
    throw std::invalid_argument(
        "csr_gmres requires restart > 0 and maxiter >= 0.");
  }
  require_rank(data, 1, "csr_gmres data");
  require_rank(indices, 1, "csr_gmres indices");
  require_rank(indptr, 1, "csr_gmres indptr");
  require_rank(b, 1, "csr_gmres b");
  require_rank(x0, 1, "csr_gmres x0");
  require_linalg_float32(data, "csr_gmres data");
  require_linalg_float32(b, "csr_gmres b");
  require_linalg_float32(x0, "csr_gmres x0");
  require_same_index_dtype(indices, indptr, "csr_gmres indices",
                           "csr_gmres indptr");
  require_size(indptr, n_rows + 1, "csr_gmres indptr");
  require_size(b, n_rows, "csr_gmres b");
  require_size(x0, n_cols, "csr_gmres x0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_gmres data and indices must have equal length.");
  }
  if (indices.dtype() == mx::int32) {
    return csr_gmres_impl<int32_t>(data, indices, indptr, b, x0, n_rows, rtol,
                                   atol, restart, maxiter);
  }
  if (indices.dtype() == mx::int64) {
    return csr_gmres_impl<int64_t>(data, indices, indptr, b, x0, n_rows, rtol,
                                   atol, restart, maxiter);
  }
  throw std::runtime_error("csr_gmres requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_minres(const mx::array &data, const mx::array &indices,
           const mx::array &indptr, const mx::array &b, const mx::array &x0,
           int n_rows, int n_cols, float rtol, float atol, int maxiter) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_minres requires a non-empty square matrix.");
  }
  if (maxiter < 0) {
    throw std::invalid_argument("csr_minres maxiter must be non-negative.");
  }
  require_rank(data, 1, "csr_minres data");
  require_rank(indices, 1, "csr_minres indices");
  require_rank(indptr, 1, "csr_minres indptr");
  require_rank(b, 1, "csr_minres b");
  require_rank(x0, 1, "csr_minres x0");
  require_linalg_float32(data, "csr_minres data");
  require_linalg_float32(b, "csr_minres b");
  require_linalg_float32(x0, "csr_minres x0");
  require_same_index_dtype(indices, indptr, "csr_minres indices",
                           "csr_minres indptr");
  require_size(indptr, n_rows + 1, "csr_minres indptr");
  require_size(b, n_rows, "csr_minres b");
  require_size(x0, n_cols, "csr_minres x0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_minres data and indices must have equal length.");
  }
  if (indices.dtype() == mx::int32) {
    return csr_minres_impl<int32_t>(data, indices, indptr, b, x0, n_rows,
                                    rtol, atol, maxiter);
  }
  if (indices.dtype() == mx::int64) {
    return csr_minres_impl<int64_t>(data, indices, indptr, b, x0, n_rows,
                                    rtol, atol, maxiter);
  }
  throw std::runtime_error("csr_minres requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_lanczos(const mx::array &data, const mx::array &indices,
            const mx::array &indptr, const mx::array &v0, int n_rows,
            int n_cols, int k, bool reorthogonalize, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_lanczos requires a non-empty square matrix.");
  }
  if (k <= 0 || k > n_rows) {
    throw std::invalid_argument("csr_lanczos k must satisfy 0 < k <= n_rows.");
  }
  require_rank(data, 1, "csr_lanczos data");
  require_rank(indices, 1, "csr_lanczos indices");
  require_rank(indptr, 1, "csr_lanczos indptr");
  require_rank(v0, 1, "csr_lanczos v0");
  require_linalg_float32(data, "csr_lanczos data");
  require_linalg_float32(v0, "csr_lanczos v0");
  require_same_index_dtype(indices, indptr, "csr_lanczos indices",
                           "csr_lanczos indptr");
  require_size(indptr, n_rows + 1, "csr_lanczos indptr");
  require_size(v0, n_rows, "csr_lanczos v0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_lanczos data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto v0_contig = mx::contiguous(v0, false, stream);

  auto primitive =
      std::make_shared<CSRLanczos>(stream, n_rows, n_cols, k, reorthogonalize);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{k}, mx::Shape{k}, mx::Shape{n_rows, k}, mx::Shape{}},
      {mx::float32, mx::float32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, v0_contig});
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

std::tuple<mx::array, mx::array, mx::array>
csr_arnoldi(const mx::array &data, const mx::array &indices,
            const mx::array &indptr, const mx::array &v0, int n_rows,
            int n_cols, int k, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_arnoldi requires a non-empty square matrix.");
  }
  if (k <= 0 || k > n_rows) {
    throw std::invalid_argument("csr_arnoldi k must satisfy 0 < k <= n_rows.");
  }
  require_rank(data, 1, "csr_arnoldi data");
  require_rank(indices, 1, "csr_arnoldi indices");
  require_rank(indptr, 1, "csr_arnoldi indptr");
  require_rank(v0, 1, "csr_arnoldi v0");
  require_linalg_float32(data, "csr_arnoldi data");
  require_linalg_float32(v0, "csr_arnoldi v0");
  require_same_index_dtype(indices, indptr, "csr_arnoldi indices",
                           "csr_arnoldi indptr");
  require_size(indptr, n_rows + 1, "csr_arnoldi indptr");
  require_size(v0, n_rows, "csr_arnoldi v0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_arnoldi data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto v0_contig = mx::contiguous(v0, false, stream);

  auto primitive = std::make_shared<CSRArnoldi>(stream, n_rows, n_cols, k);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{k + 1, k}, mx::Shape{n_rows, k + 1}, mx::Shape{}},
      {mx::float32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, v0_contig});
  return {outputs[0], outputs[1], outputs[2]};
}

std::tuple<mx::array, mx::array>
csr_eigsh(const mx::array &data, const mx::array &indices,
          const mx::array &indptr, int n_rows, int n_cols, int k, int ncv,
          const std::string &which) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_eigsh requires a non-empty square matrix.");
  }
  if (k <= 0 || k >= n_rows) {
    throw std::invalid_argument("csr_eigsh k must satisfy 0 < k < n_rows.");
  }
  require_rank(data, 1, "csr_eigsh data");
  require_rank(indices, 1, "csr_eigsh indices");
  require_rank(indptr, 1, "csr_eigsh indptr");
  require_linalg_float32(data, "csr_eigsh data");
  require_same_index_dtype(indices, indptr, "csr_eigsh indices",
                           "csr_eigsh indptr");
  require_size(indptr, n_rows + 1, "csr_eigsh indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_eigsh data and indices must have equal length.");
  }
  ncv = std::min(n_rows, std::max(ncv, k + 1));
  if (indices.dtype() == mx::int32) {
    return csr_eigsh_impl<int32_t>(data, indices, indptr, n_rows, k, ncv,
                                   which);
  }
  if (indices.dtype() == mx::int64) {
    return csr_eigsh_impl<int64_t>(data, indices, indptr, n_rows, k, ncv,
                                   which);
  }
  throw std::runtime_error("csr_eigsh requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array>
csr_eigs(const mx::array &data, const mx::array &indices,
         const mx::array &indptr, int n_rows, int n_cols, int k, int ncv,
         const std::string &which) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_eigs requires a non-empty square matrix.");
  }
  if (k <= 0 || k >= n_rows) {
    throw std::invalid_argument("csr_eigs k must satisfy 0 < k < n_rows.");
  }
  require_rank(data, 1, "csr_eigs data");
  require_rank(indices, 1, "csr_eigs indices");
  require_rank(indptr, 1, "csr_eigs indptr");
  require_linalg_float32(data, "csr_eigs data");
  require_same_index_dtype(indices, indptr, "csr_eigs indices",
                           "csr_eigs indptr");
  require_size(indptr, n_rows + 1, "csr_eigs indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_eigs data and indices must have equal length.");
  }
  ncv = std::min(n_rows, std::max(ncv, k + 1));
  if (indices.dtype() == mx::int32) {
    return csr_eigs_impl<int32_t>(data, indices, indptr, n_rows, k, ncv,
                                  which);
  }
  if (indices.dtype() == mx::int64) {
    return csr_eigs_impl<int64_t>(data, indices, indptr, n_rows, k, ncv,
                                  which);
  }
  throw std::runtime_error("csr_eigs requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array, mx::array>
csr_svds(const mx::array &data, const mx::array &indices,
         const mx::array &indptr, int n_rows, int n_cols, int k, int ncv,
         const std::string &which) {
  if (n_rows <= 0 || n_cols <= 0) {
    throw std::invalid_argument("csr_svds requires a non-empty matrix.");
  }
  if (k <= 0 || k >= std::min(n_rows, n_cols)) {
    throw std::invalid_argument(
        "csr_svds k must satisfy 0 < k < min(shape).");
  }
  require_rank(data, 1, "csr_svds data");
  require_rank(indices, 1, "csr_svds indices");
  require_rank(indptr, 1, "csr_svds indptr");
  require_linalg_float32(data, "csr_svds data");
  require_same_index_dtype(indices, indptr, "csr_svds indices",
                           "csr_svds indptr");
  require_size(indptr, n_rows + 1, "csr_svds indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_svds data and indices must have equal length.");
  }
  ncv = std::min(n_cols, std::max(ncv, k + 1));
  if (indices.dtype() == mx::int32) {
    return csr_svds_impl<int32_t>(data, indices, indptr, n_rows, n_cols, k,
                                  ncv, which);
  }
  if (indices.dtype() == mx::int64) {
    return csr_svds_impl<int64_t>(data, indices, indptr, n_rows, n_cols, k,
                                  ncv, which);
  }
  throw std::runtime_error("csr_svds requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array, mx::array>
csr_cholesky(const mx::array &data, const mx::array &indices,
             const mx::array &indptr, int n_rows, int n_cols) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_cholesky requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_cholesky data");
  require_rank(indices, 1, "csr_cholesky indices");
  require_rank(indptr, 1, "csr_cholesky indptr");
  require_linalg_float32(data, "csr_cholesky data");
  require_same_index_dtype(indices, indptr, "csr_cholesky indices",
                           "csr_cholesky indptr");
  require_size(indptr, n_rows + 1, "csr_cholesky indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_cholesky data and indices must have equal length.");
  }
  if (indices.dtype() == mx::int32) {
    return csr_cholesky_impl<int32_t>(data, indices, indptr, n_rows, n_cols,
                                      mx::int32);
  }
  if (indices.dtype() == mx::int64) {
    return csr_cholesky_impl<int64_t>(data, indices, indptr, n_rows, n_cols,
                                      mx::int64);
  }
  throw std::runtime_error("csr_cholesky requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array,
           mx::array>
csr_lu(const mx::array &data, const mx::array &indices,
       const mx::array &indptr, int n_rows, int n_cols) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_lu requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_lu data");
  require_rank(indices, 1, "csr_lu indices");
  require_rank(indptr, 1, "csr_lu indptr");
  require_linalg_float32(data, "csr_lu data");
  require_same_index_dtype(indices, indptr, "csr_lu indices",
                           "csr_lu indptr");
  require_size(indptr, n_rows + 1, "csr_lu indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument("csr_lu data and indices must have equal length.");
  }
  if (indices.dtype() == mx::int32) {
    return csr_lu_impl<int32_t>(data, indices, indptr, n_rows, n_cols,
                                mx::int32);
  }
  if (indices.dtype() == mx::int64) {
    return csr_lu_impl<int64_t>(data, indices, indptr, n_rows, n_cols,
                                mx::int64);
  }
  throw std::runtime_error("csr_lu requires int32 or int64 indices.");
}

mx::array csr_triangular_solve(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, const mx::array &b,
                               int n_rows, int n_cols, bool lower,
                               bool unit_diagonal, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_triangular_solve requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_triangular_solve data");
  require_rank(indices, 1, "csr_triangular_solve indices");
  require_rank(indptr, 1, "csr_triangular_solve indptr");
  require_rank(b, 1, "csr_triangular_solve b");
  require_linalg_float32(data, "csr_triangular_solve data");
  require_linalg_float32(b, "csr_triangular_solve b");
  require_same_index_dtype(indices, indptr, "csr_triangular_solve indices",
                           "csr_triangular_solve indptr");
  require_size(indptr, n_rows + 1, "csr_triangular_solve indptr");
  require_size(b, n_rows, "csr_triangular_solve b");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_triangular_solve data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);

  return mx::array(
      mx::Shape{n_rows}, mx::float32,
      std::make_shared<CSRTriangularSolve>(stream, n_rows, n_cols, lower,
                                           unit_diagonal),
      {data_contig, indices_contig, indptr_contig, b_contig});
}

mx::array csr_inner_product(const mx::array &lhs_data,
                            const mx::array &lhs_indices,
                            const mx::array &lhs_indptr,
                            const mx::array &rhs_data,
                            const mx::array &rhs_indices,
                            const mx::array &rhs_indptr, int n_rows,
                            int n_cols, bool conjugate_lhs,
                            mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr sparse inner product shape dimensions must be non-negative.");
  }
  require_rank(lhs_data, 1, "csr inner product lhs_data");
  require_rank(lhs_indices, 1, "csr inner product lhs_indices");
  require_rank(lhs_indptr, 1, "csr inner product lhs_indptr");
  require_rank(rhs_data, 1, "csr inner product rhs_data");
  require_rank(rhs_indices, 1, "csr inner product rhs_indices");
  require_rank(rhs_indptr, 1, "csr inner product rhs_indptr");
  require_inner_product_dtype(lhs_data, "csr inner product lhs_data");
  require_inner_product_dtype(rhs_data, "csr inner product rhs_data");
  if (lhs_data.dtype() != rhs_data.dtype()) {
    throw std::invalid_argument(
        "csr sparse inner product operands must use the same value dtype.");
  }
  require_same_index_dtype(lhs_indices, lhs_indptr,
                           "csr inner product lhs_indices",
                           "csr inner product lhs_indptr");
  require_same_index_dtype(rhs_indices, rhs_indptr,
                           "csr inner product rhs_indices",
                           "csr inner product rhs_indptr");
  if (lhs_indices.dtype() != rhs_indices.dtype()) {
    throw std::invalid_argument(
        "csr sparse inner product operands must use the same index dtype.");
  }
  require_size(lhs_indptr, n_rows + 1, "csr inner product lhs_indptr");
  require_size(rhs_indptr, n_rows + 1, "csr inner product rhs_indptr");
  if (lhs_indices.size() != lhs_data.size() ||
      rhs_indices.size() != rhs_data.size()) {
    throw std::invalid_argument(
        "csr sparse inner product data and indices must have equal lengths.");
  }

  auto stream = mx::to_stream(s);
  auto lhs_data_contig = mx::contiguous(lhs_data, false, stream);
  auto lhs_indices_contig = mx::contiguous(lhs_indices, false, stream);
  auto lhs_indptr_contig = mx::contiguous(lhs_indptr, false, stream);
  auto rhs_data_contig = mx::contiguous(rhs_data, false, stream);
  auto rhs_indices_contig = mx::contiguous(rhs_indices, false, stream);
  auto rhs_indptr_contig = mx::contiguous(rhs_indptr, false, stream);

  return mx::array(
      mx::Shape{}, lhs_data.dtype(),
      std::make_shared<CSRVdot>(stream, n_rows, n_cols, conjugate_lhs),
      {lhs_data_contig, lhs_indices_contig, lhs_indptr_contig,
       rhs_data_contig, rhs_indices_contig, rhs_indptr_contig});
}

mx::array csr_vdot(const mx::array &lhs_data, const mx::array &lhs_indices,
                   const mx::array &lhs_indptr, const mx::array &rhs_data,
                   const mx::array &rhs_indices,
                   const mx::array &rhs_indptr, int n_rows, int n_cols,
                   mx::StreamOrDevice s) {
  return csr_inner_product(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                           rhs_indices, rhs_indptr, n_rows, n_cols, true, s);
}

mx::array csr_dot(const mx::array &lhs_data, const mx::array &lhs_indices,
                  const mx::array &lhs_indptr, const mx::array &rhs_data,
                  const mx::array &rhs_indices,
                  const mx::array &rhs_indptr, int n_rows, int n_cols,
                  mx::StreamOrDevice s) {
  return csr_inner_product(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                           rhs_indices, rhs_indptr, n_rows, n_cols, false, s);
}

mx::array csr_permute_vector(const mx::array &x, const mx::array &perm,
                             mx::StreamOrDevice s) {
  require_rank(x, 1, "csr_permute_vector x");
  require_rank(perm, 1, "csr_permute_vector perm");
  require_linalg_float32(x, "csr_permute_vector x");
  if (perm.dtype() != mx::int32) {
    throw std::invalid_argument("csr_permute_vector perm must have dtype int32.");
  }
  require_size(perm, static_cast<int>(x.size()), "csr_permute_vector perm");

  auto stream = mx::to_stream(s);
  auto x_contig = mx::contiguous(x, false, stream);
  auto perm_contig = mx::contiguous(perm, false, stream);
  return mx::array(mx::Shape{static_cast<int>(x.size())}, mx::float32,
                   std::make_shared<CSRPermuteVector>(stream),
                   {x_contig, perm_contig});
}

} // namespace mlx_sparse
