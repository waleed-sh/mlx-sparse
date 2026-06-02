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

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "common/common.h"

namespace mlx_sparse::linalg_detail {

constexpr int kSolverThreads = 256;

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

inline double dot_double(const std::vector<float> &lhs,
                         const std::vector<float> &rhs) {
  double acc = 0.0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    acc += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
  }
  return acc;
}

inline float dot_float(const std::vector<float> &lhs,
                       const std::vector<float> &rhs) {
  return static_cast<float>(dot_double(lhs, rhs));
}

inline float dot_column_float(const float *basis, const float *w, int n,
                              int stride, int col) {
  double acc = 0.0;
  for (int row = 0; row < n; ++row) {
    acc += static_cast<double>(basis[static_cast<size_t>(row) * stride + col]) *
           static_cast<double>(w[row]);
  }
  return static_cast<float>(acc);
}

inline float norm_float(const std::vector<float> &x) {
  return std::sqrt(std::max(dot_float(x, x), 0.0f));
}

inline std::vector<double> solve_dense_system(std::vector<double> a,
                                              std::vector<double> b, int n) {
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double pivot_abs = std::abs(a[static_cast<size_t>(col) * n + col]);
    for (int row = col + 1; row < n; ++row) {
      const double candidate = std::abs(a[static_cast<size_t>(row) * n + col]);
      if (candidate > pivot_abs) {
        pivot_abs = candidate;
        pivot = row;
      }
    }
    if (pivot_abs <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          "small dense solve encountered a singular matrix.");
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
      sum -=
          a[static_cast<size_t>(row) * n + col] * x[static_cast<size_t>(col)];
    }
    x[static_cast<size_t>(row)] = sum / a[static_cast<size_t>(row) * n + row];
  }
  return x;
}

inline std::vector<double>
least_squares_normal_equations(const std::vector<double> &a,
                               const std::vector<double> &b, int rows,
                               int cols) {
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

inline std::vector<double>
least_squares_upper_hessenberg_givens_qr(const std::vector<double> &a,
                                         const std::vector<double> &b, int rows,
                                         int cols) {
  if (rows != cols + 1 || rows <= 1 || cols <= 0) {
    throw std::invalid_argument(
        "upper-Hessenberg Givens least-squares solve requires rows == cols + "
        "1.");
  }
  if (a.size() != static_cast<size_t>(rows) * cols ||
      b.size() != static_cast<size_t>(rows)) {
    throw std::invalid_argument(
        "upper-Hessenberg Givens least-squares solve received inconsistent "
        "dimensions.");
  }

  std::vector<double> r = a;
  std::vector<double> rhs = b;
  double scale = 0.0;
  for (double value : r) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "upper-Hessenberg Givens least-squares solve received a non-finite "
          "matrix value.");
    }
    scale = std::max(scale, std::abs(value));
  }
  for (double value : rhs) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "upper-Hessenberg Givens least-squares solve received a non-finite "
          "right-hand side.");
    }
  }
  const double rank_tol = std::numeric_limits<double>::epsilon() *
                          static_cast<double>(std::max(rows, cols)) *
                          std::max(1.0, scale);
  if (scale <= rank_tol) {
    throw std::runtime_error(
        "upper-Hessenberg Givens least-squares solve encountered a "
        "rank-deficient matrix.");
  }

  for (int col = 0; col < cols; ++col) {
    const int row = col + 1;
    const double pivot = r[static_cast<size_t>(col) * cols + col];
    const double entry = r[static_cast<size_t>(row) * cols + col];
    if (entry == 0.0) {
      continue;
    }
    const double radius = std::hypot(pivot, entry);
    if (!std::isfinite(radius) || radius <= rank_tol) {
      throw std::runtime_error(
          "upper-Hessenberg Givens least-squares solve encountered a "
          "degenerate rotation.");
    }
    const double c = pivot / radius;
    const double s = entry / radius;
    for (int j = col; j < cols; ++j) {
      const double top = r[static_cast<size_t>(col) * cols + j];
      const double bottom = r[static_cast<size_t>(row) * cols + j];
      r[static_cast<size_t>(col) * cols + j] = c * top + s * bottom;
      r[static_cast<size_t>(row) * cols + j] = -s * top + c * bottom;
    }
    const double rhs_top = rhs[static_cast<size_t>(col)];
    const double rhs_bottom = rhs[static_cast<size_t>(row)];
    rhs[static_cast<size_t>(col)] = c * rhs_top + s * rhs_bottom;
    rhs[static_cast<size_t>(row)] = -s * rhs_top + c * rhs_bottom;
  }

  std::vector<double> x(static_cast<size_t>(cols), 0.0);
  for (int row = cols - 1; row >= 0; --row) {
    double sum = rhs[static_cast<size_t>(row)];
    for (int col = row + 1; col < cols; ++col) {
      sum -= r[static_cast<size_t>(row) * cols + col] *
             x[static_cast<size_t>(col)];
    }
    const double diag = r[static_cast<size_t>(row) * cols + row];
    if (!std::isfinite(diag) || std::abs(diag) <= rank_tol) {
      throw std::runtime_error(
          "upper-Hessenberg Givens least-squares solve encountered a "
          "rank-deficient R factor.");
    }
    x[static_cast<size_t>(row)] = sum / diag;
    if (!std::isfinite(x[static_cast<size_t>(row)])) {
      throw std::runtime_error(
          "upper-Hessenberg Givens least-squares solve produced a non-finite "
          "solution.");
    }
  }
  return x;
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

inline std::pair<std::vector<float>, std::vector<float>>
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
    const float t = (tau >= 0.0f ? 1.0f : -1.0f) /
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

inline std::vector<int> select_ritz_indices(const std::vector<float> &values,
                                            int k, const std::string &which) {
  std::vector<int> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  if (which == "SM" || which == "SA" || which == "SR") {
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      if (which == "SM") {
        return std::abs(values[static_cast<size_t>(lhs)]) <
               std::abs(values[static_cast<size_t>(rhs)]);
      }
      return values[static_cast<size_t>(lhs)] <
             values[static_cast<size_t>(rhs)];
    });
  } else {
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      if (which == "LM") {
        return std::abs(values[static_cast<size_t>(lhs)]) >
               std::abs(values[static_cast<size_t>(rhs)]);
      }
      return values[static_cast<size_t>(lhs)] >
             values[static_cast<size_t>(rhs)];
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
      q[static_cast<size_t>(row)] = basis[static_cast<size_t>(row) * steps + j];
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

inline void require_linalg_float32(const mx::array &array, const char *name) {
  if (array.dtype() != mx::float32) {
    throw std::invalid_argument(std::string(name) +
                                " currently requires dtype float32.");
  }
}

inline void require_inner_product_dtype(const mx::array &array,
                                        const char *name) {
  if (array.dtype() != mx::float32 && array.dtype() != mx::complex64) {
    throw std::invalid_argument(std::string(name) +
                                " requires dtype float32 or complex64.");
  }
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array>
make_csr_arrays_float32(const std::vector<float> &data,
                        const std::vector<I> &indices,
                        const std::vector<I> &indptr, mx::Dtype index_dtype) {
  return {mx::array(data.begin(), mx::Shape{static_cast<int>(data.size())},
                    mx::float32),
          mx::array(indices.begin(),
                    mx::Shape{static_cast<int>(indices.size())}, index_dtype),
          mx::array(indptr.begin(), mx::Shape{static_cast<int>(indptr.size())},
                    index_dtype)};
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

} // namespace mlx_sparse::linalg_detail
