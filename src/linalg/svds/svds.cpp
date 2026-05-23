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

#include "linalg/svds/svds.h"

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
  auto [tridiagonal, basis, used] =
      host_lanczos_operator(n_cols, steps, [&](const std::vector<float> &x) {
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

std::tuple<mx::array, mx::array, mx::array>
csr_svds(const mx::array &data, const mx::array &indices,
         const mx::array &indptr, int n_rows, int n_cols, int k, int ncv,
         const std::string &which) {
  if (n_rows <= 0 || n_cols <= 0) {
    throw std::invalid_argument("csr_svds requires a non-empty matrix.");
  }
  if (k <= 0 || k >= std::min(n_rows, n_cols)) {
    throw std::invalid_argument("csr_svds k must satisfy 0 < k < min(shape).");
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
    return csr_svds_impl<int32_t>(data, indices, indptr, n_rows, n_cols, k, ncv,
                                  which);
  }
  if (indices.dtype() == mx::int64) {
    return csr_svds_impl<int64_t>(data, indices, indptr, n_rows, n_cols, k, ncv,
                                  which);
  }
  throw std::runtime_error("csr_svds requires int32 or int64 indices.");
}

} // namespace mlx_sparse
