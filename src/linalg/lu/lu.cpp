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

#include "linalg/lu/lu.h"

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
      const float value =
          found == rows[static_cast<size_t>(row)].end() ? 0.0f : found->second;
      if (std::abs(value) > pivot_abs) {
        pivot_abs = std::abs(value);
        pivot_row = row;
      }
    }
    if (pivot_abs <= eps) {
      throw std::runtime_error(
          "csr_lu encountered a structurally singular pivot.");
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
    l_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(l_data.size());
    for (const auto &[col, value] : U[static_cast<size_t>(row)]) {
      if (col >= row && std::abs(value) > eps) {
        u_data.push_back(value);
        u_indices.push_back(static_cast<I>(col));
      }
    }
    u_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(u_data.size());
  }

  auto permutation = mx::array(
      perm.begin(), mx::Shape{static_cast<int>(perm.size())}, mx::int32);
  auto [l_data_array, l_indices_array, l_indptr_array] =
      make_csr_arrays_float32(l_data, l_indices, l_indptr, index_dtype);
  auto [u_data_array, u_indices_array, u_indptr_array] =
      make_csr_arrays_float32(u_data, u_indices, u_indptr, index_dtype);
  return {permutation,  l_data_array,    l_indices_array, l_indptr_array,
          u_data_array, u_indices_array, u_indptr_array};
}

} // namespace

std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array,
           mx::array>
csr_lu(const mx::array &data, const mx::array &indices, const mx::array &indptr,
       int n_rows, int n_cols) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_lu requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_lu data");
  require_rank(indices, 1, "csr_lu indices");
  require_rank(indptr, 1, "csr_lu indptr");
  require_linalg_float32(data, "csr_lu data");
  require_same_index_dtype(indices, indptr, "csr_lu indices", "csr_lu indptr");
  require_size(indptr, n_rows + 1, "csr_lu indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_lu data and indices must have equal length.");
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

} // namespace mlx_sparse
