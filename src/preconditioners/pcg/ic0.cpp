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
#include "mlx/ops.h"

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

mx::array vector_to_mx_array(const std::vector<float> &values) {
  return mx::array(values.begin(), mx::Shape{static_cast<int>(values.size())},
                   mx::float32);
}

template <typename I>
bool csr_lower_solve_host(const float *data, const I *indices, const I *indptr,
                          const std::vector<float> &rhs,
                          std::vector<float> &out, int n_rows) {
  const float eps = std::numeric_limits<float>::epsilon();
  for (int row = 0; row < n_rows; ++row) {
    double sum = static_cast<double>(rhs[static_cast<size_t>(row)]);
    float diag = 0.0f;
    bool has_diag = false;
    double row_scale = 1.0;
    for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
      const int col = static_cast<int>(indices[p]);
      const float value = data[p];
      row_scale = std::max(row_scale, std::abs(static_cast<double>(value)));
      if (!finite_float(value) || col < 0 || col >= n_rows || col > row) {
        return false;
      }
      if (col < row) {
        sum -= static_cast<double>(value) *
               static_cast<double>(out[static_cast<size_t>(col)]);
      } else {
        diag = value;
        has_diag = true;
      }
    }
    const float threshold = eps * static_cast<float>(std::max(1.0, row_scale));
    if (!has_diag || !finite_float(diag) || diag <= threshold) {
      return false;
    }
    const float solved = static_cast<float>(sum) / diag;
    if (!finite_float(solved)) {
      return false;
    }
    out[static_cast<size_t>(row)] = solved;
  }
  return true;
}

template <typename I>
bool csr_upper_solve_host(const float *data, const I *indices, const I *indptr,
                          const std::vector<float> &rhs,
                          std::vector<float> &out, int n_rows) {
  const float eps = std::numeric_limits<float>::epsilon();
  for (int row = n_rows - 1; row >= 0; --row) {
    double sum = static_cast<double>(rhs[static_cast<size_t>(row)]);
    float diag = 0.0f;
    bool has_diag = false;
    double row_scale = 1.0;
    for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
      const int col = static_cast<int>(indices[p]);
      const float value = data[p];
      row_scale = std::max(row_scale, std::abs(static_cast<double>(value)));
      if (!finite_float(value) || col < 0 || col >= n_rows || col < row) {
        return false;
      }
      if (col > row) {
        sum -= static_cast<double>(value) *
               static_cast<double>(out[static_cast<size_t>(col)]);
      } else {
        diag = value;
        has_diag = true;
      }
    }
    const float threshold = eps * static_cast<float>(std::max(1.0, row_scale));
    if (!has_diag || !finite_float(diag) || diag <= threshold) {
      return false;
    }
    const float solved = static_cast<float>(sum) / diag;
    if (!finite_float(solved)) {
      return false;
    }
    out[static_cast<size_t>(row)] = solved;
  }
  return true;
}

template <typename I>
bool apply_ic0_host(const float *l_data, const I *l_indices, const I *l_indptr,
                    const float *lt_data, const I *lt_indices,
                    const I *lt_indptr, const std::vector<float> &rhs,
                    std::vector<float> &out, int n_rows) {
  std::vector<float> y(static_cast<size_t>(n_rows), 0.0f);
  std::fill(out.begin(), out.end(), 0.0f);
  return csr_lower_solve_host(l_data, l_indices, l_indptr, rhs, y, n_rows) &&
         csr_upper_solve_host(lt_data, lt_indices, lt_indptr, y, out, n_rows);
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_pcg_ic0_impl(mx::array data, mx::array indices, mx::array indptr,
                 mx::array b, mx::array x0, mx::array l_data,
                 mx::array l_indices, mx::array l_indptr, mx::array lt_data,
                 mx::array lt_indices, mx::array lt_indptr, int n_rows,
                 float rtol, float atol, int maxiter) {
  data.eval();
  indices.eval();
  indptr.eval();
  b.eval();
  x0.eval();
  l_data.eval();
  l_indices.eval();
  l_indptr.eval();
  lt_data.eval();
  lt_indices.eval();
  lt_indptr.eval();

  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  const auto *b_ptr = b.data<float>();
  const auto *x0_ptr = x0.data<float>();
  const auto *l_data_ptr = l_data.data<float>();
  const auto *l_indices_ptr = l_indices.data<I>();
  const auto *l_indptr_ptr = l_indptr.data<I>();
  const auto *lt_data_ptr = lt_data.data<float>();
  const auto *lt_indices_ptr = lt_indices.data<I>();
  const auto *lt_indptr_ptr = lt_indptr.data<I>();

  std::vector<float> x(x0_ptr, x0_ptr + n_rows);
  std::vector<float> r(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> z(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> p(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> ap(static_cast<size_t>(n_rows), 0.0f);

  csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, x.data(), ap.data(),
                 n_rows);
  double b_norm2 = 0.0;
  bool finite = true;
  for (int i = 0; i < n_rows; ++i) {
    const float ri = b_ptr[i] - ap[static_cast<size_t>(i)];
    r[static_cast<size_t>(i)] = ri;
    b_norm2 += static_cast<double>(b_ptr[i]) * static_cast<double>(b_ptr[i]);
    finite = finite && finite_float(ri) && finite_float(b_ptr[i]) &&
             finite_float(x[static_cast<size_t>(i)]);
  }

  float true_rr = dot_float(r, r);
  const float b_norm = std::sqrt(std::max(b_norm2, 0.0));
  const float tol = std::max(atol, rtol * b_norm);
  float true_residual = std::sqrt(std::max(true_rr, 0.0f));
  if (!finite || !finite_float(true_rr) || !finite_float(true_residual)) {
    return {vector_to_mx_array(x), mx::array(-3, mx::int32),
            mx::array(true_residual, mx::float32), mx::array(0, mx::int32)};
  }
  if (true_residual <= tol) {
    return {vector_to_mx_array(x), mx::array(0, mx::int32),
            mx::array(true_residual, mx::float32), mx::array(0, mx::int32)};
  }

  if (!apply_ic0_host(l_data_ptr, l_indices_ptr, l_indptr_ptr, lt_data_ptr,
                      lt_indices_ptr, lt_indptr_ptr, r, z, n_rows) ||
      !finite_vector(z)) {
    return {vector_to_mx_array(x), mx::array(-2, mx::int32),
            mx::array(true_residual, mx::float32), mx::array(0, mx::int32)};
  }
  p = z;
  float rho = dot_float(r, z);
  if (!finite_float(rho) || rho <= 0.0f) {
    return {vector_to_mx_array(x), mx::array(-2, mx::int32),
            mx::array(true_residual, mx::float32), mx::array(0, mx::int32)};
  }

  const float eps = std::numeric_limits<float>::epsilon();
  int status = maxiter > 0 ? maxiter : 1;
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
      x[static_cast<size_t>(i)] += alpha * p[static_cast<size_t>(i)];
      r[static_cast<size_t>(i)] -= alpha * ap[static_cast<size_t>(i)];
      true_rr_acc += static_cast<double>(r[static_cast<size_t>(i)]) *
                     static_cast<double>(r[static_cast<size_t>(i)]);
      finite = finite && finite_float(x[static_cast<size_t>(i)]) &&
               finite_float(r[static_cast<size_t>(i)]);
    }
    true_rr = static_cast<float>(true_rr_acc);
    true_residual = std::sqrt(std::max(true_rr, 0.0f));
    completed = it;
    if (!finite || !finite_float(true_rr) || !finite_float(true_residual)) {
      status = -3;
      break;
    }
    if (true_residual <= tol) {
      status = 0;
      break;
    }

    if (!apply_ic0_host(l_data_ptr, l_indices_ptr, l_indptr_ptr, lt_data_ptr,
                        lt_indices_ptr, lt_indptr_ptr, r, z, n_rows) ||
        !finite_vector(z)) {
      status = -2;
      break;
    }
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

  return {vector_to_mx_array(x), mx::array(status, mx::int32),
          mx::array(true_residual, mx::float32),
          mx::array(completed, mx::int32)};
}

} // namespace

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_pcg_ic0(const mx::array &data, const mx::array &indices,
            const mx::array &indptr, const mx::array &b, const mx::array &x0,
            const mx::array &l_data, const mx::array &l_indices,
            const mx::array &l_indptr, const mx::array &lt_data,
            const mx::array &lt_indices, const mx::array &lt_indptr, int n_rows,
            int n_cols, float rtol, float atol, int maxiter,
            mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_pcg_ic0 requires a non-empty square matrix.");
  }
  if (maxiter < 0) {
    throw std::invalid_argument("csr_pcg_ic0 maxiter must be non-negative.");
  }
  if (!std::isfinite(rtol) || !std::isfinite(atol) || rtol < 0.0f ||
      atol < 0.0f) {
    throw std::invalid_argument(
        "csr_pcg_ic0 requires finite non-negative tolerances.");
  }
  require_rank(data, 1, "csr_pcg_ic0 data");
  require_rank(indices, 1, "csr_pcg_ic0 indices");
  require_rank(indptr, 1, "csr_pcg_ic0 indptr");
  require_rank(b, 1, "csr_pcg_ic0 b");
  require_rank(x0, 1, "csr_pcg_ic0 x0");
  require_rank(l_data, 1, "csr_pcg_ic0 L data");
  require_rank(l_indices, 1, "csr_pcg_ic0 L indices");
  require_rank(l_indptr, 1, "csr_pcg_ic0 L indptr");
  require_rank(lt_data, 1, "csr_pcg_ic0 LT data");
  require_rank(lt_indices, 1, "csr_pcg_ic0 LT indices");
  require_rank(lt_indptr, 1, "csr_pcg_ic0 LT indptr");
  require_linalg_float32(data, "csr_pcg_ic0 data");
  require_linalg_float32(b, "csr_pcg_ic0 b");
  require_linalg_float32(x0, "csr_pcg_ic0 x0");
  require_linalg_float32(l_data, "csr_pcg_ic0 L data");
  require_linalg_float32(lt_data, "csr_pcg_ic0 LT data");
  require_same_index_dtype(indices, indptr, "csr_pcg_ic0 indices",
                           "csr_pcg_ic0 indptr");
  require_same_index_dtype(l_indices, l_indptr, "csr_pcg_ic0 L indices",
                           "csr_pcg_ic0 L indptr");
  require_same_index_dtype(lt_indices, lt_indptr, "csr_pcg_ic0 LT indices",
                           "csr_pcg_ic0 LT indptr");
  if (indices.dtype() != l_indices.dtype() ||
      indices.dtype() != lt_indices.dtype()) {
    throw std::invalid_argument(
        "csr_pcg_ic0 matrix and factor index dtypes must match.");
  }
  require_size(indptr, n_rows + 1, "csr_pcg_ic0 indptr");
  require_size(l_indptr, n_rows + 1, "csr_pcg_ic0 L indptr");
  require_size(lt_indptr, n_rows + 1, "csr_pcg_ic0 LT indptr");
  require_size(b, n_rows, "csr_pcg_ic0 b");
  require_size(x0, n_cols, "csr_pcg_ic0 x0");
  if (indices.size() != data.size() || l_indices.size() != l_data.size() ||
      lt_indices.size() != lt_data.size()) {
    throw std::invalid_argument(
        "csr_pcg_ic0 data and indices must have equal lengths.");
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

  if (indices.dtype() == mx::int32) {
    return csr_pcg_ic0_impl<int32_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig,
        l_data_contig, l_indices_contig, l_indptr_contig, lt_data_contig,
        lt_indices_contig, lt_indptr_contig, n_rows, rtol, atol, maxiter);
  }
  if (indices.dtype() == mx::int64) {
    return csr_pcg_ic0_impl<int64_t>(
        data_contig, indices_contig, indptr_contig, b_contig, x0_contig,
        l_data_contig, l_indices_contig, l_indptr_contig, lt_data_contig,
        lt_indices_contig, lt_indptr_contig, n_rows, rtol, atol, maxiter);
  }
  throw std::runtime_error("csr_pcg_ic0 requires int32 or int64 indices.");
}

} // namespace mlx_sparse
