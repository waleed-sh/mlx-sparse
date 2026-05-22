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

#include "linalg/eigsh/eigsh.h"

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

} // namespace

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

} // namespace mlx_sparse
