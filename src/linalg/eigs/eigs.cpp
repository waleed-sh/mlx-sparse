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

#include "linalg/eigs/eigs.h"

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
          coeff += q[static_cast<size_t>(row) * n + prev] *
                   v[static_cast<size_t>(row)];
        }
        r[static_cast<size_t>(prev) * n + col] = static_cast<float>(coeff);
        for (int row = 0; row < n; ++row) {
          v[static_cast<size_t>(row)] -= static_cast<float>(coeff) *
                                         q[static_cast<size_t>(row) * n + prev];
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

} // namespace

std::tuple<mx::array, mx::array> csr_eigs(const mx::array &data,
                                          const mx::array &indices,
                                          const mx::array &indptr, int n_rows,
                                          int n_cols, int k, int ncv,
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
    return csr_eigs_impl<int32_t>(data, indices, indptr, n_rows, k, ncv, which);
  }
  if (indices.dtype() == mx::int64) {
    return csr_eigs_impl<int64_t>(data, indices, indptr, n_rows, k, ncv, which);
  }
  throw std::runtime_error("csr_eigs requires int32 or int64 indices.");
}

} // namespace mlx_sparse
