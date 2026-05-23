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

#include "linalg/minres/minres.h"

#include "linalg/lanczos/lanczos.h"
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

template <typename I>
std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_minres_impl(mx::array data, mx::array indices, mx::array indptr,
                mx::array b, mx::array x0, int n_rows, float rtol, float atol,
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
  auto [alphas_mx, betas_mx, basis_mx, actual_k_mx] = csr_lanczos(
      data, indices, indptr, v0, n_rows, n_rows, steps, true, stream);
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

} // namespace

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_minres(const mx::array &data, const mx::array &indices,
           const mx::array &indptr, const mx::array &b, const mx::array &x0,
           int n_rows, int n_cols, float rtol, float atol, int maxiter) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_minres requires a non-empty square matrix.");
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
    return csr_minres_impl<int32_t>(data, indices, indptr, b, x0, n_rows, rtol,
                                    atol, maxiter);
  }
  if (indices.dtype() == mx::int64) {
    return csr_minres_impl<int64_t>(data, indices, indptr, b, x0, n_rows, rtol,
                                    atol, maxiter);
  }
  throw std::runtime_error("csr_minres requires int32 or int64 indices.");
}

} // namespace mlx_sparse
