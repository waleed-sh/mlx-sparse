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

#include "linalg/cholesky/cholesky.h"

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
csr_cholesky_impl(mx::array data, mx::array indices, mx::array indptr,
                  int n_rows, int n_cols, mx::Dtype index_dtype) {
  auto input_rows = read_csr_rows_float32<I>(
      std::move(data), std::move(indices), std::move(indptr), n_rows);
  std::vector<std::map<int, float>> lower(static_cast<size_t>(n_rows));
  for (int row = 0; row < n_rows; ++row) {
    for (const auto &[col, value] : input_rows[static_cast<size_t>(row)]) {
      if (col < 0 || col >= n_cols) {
        throw std::invalid_argument(
            "csr_cholesky input contains an out-of-bounds column.");
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
      const float factor = it->second / diag[static_cast<size_t>(pivot_col)];
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
      throw std::runtime_error(
          "csr_cholesky requires a positive-definite matrix.");
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
    out_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(out_data.size());
  }
  return make_csr_arrays_float32(out_data, out_indices, out_indptr,
                                 index_dtype);
}

} // namespace

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

} // namespace mlx_sparse
