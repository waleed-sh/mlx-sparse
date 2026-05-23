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

#include "linalg/gmres/gmres.h"

#include "linalg/arnoldi/arnoldi.h"
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
        update += basis_ptr[static_cast<size_t>(row) * (steps + 1) + col] *
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

} // namespace

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_gmres(const mx::array &data, const mx::array &indices,
          const mx::array &indptr, const mx::array &b, const mx::array &x0,
          int n_rows, int n_cols, float rtol, float atol, int restart,
          int maxiter) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_gmres requires a non-empty square matrix.");
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

} // namespace mlx_sparse
