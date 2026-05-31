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

using SparseRow = std::vector<std::pair<int, float>>;

void sort_and_sum_row(SparseRow &row) {
  std::sort(row.begin(), row.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  size_t write = 0;
  for (const auto &[col, value] : row) {
    if (write > 0 && row[write - 1].first == col) {
      row[write - 1].second += value;
    } else {
      row[write++] = {col, value};
    }
  }
  row.resize(write);
}

SparseRow::iterator find_row_col(SparseRow &row, int col) {
  if (row.size() <= 16) {
    return std::find_if(row.begin(), row.end(), [col](const auto &entry) {
      return entry.first >= col;
    });
  }
  return std::lower_bound(
      row.begin(), row.end(), col,
      [](const auto &entry, int target) { return entry.first < target; });
}

SparseRow::const_iterator find_row_col(const SparseRow &row, int col) {
  if (row.size() <= 16) {
    return std::find_if(row.begin(), row.end(), [col](const auto &entry) {
      return entry.first >= col;
    });
  }
  return std::lower_bound(
      row.begin(), row.end(), col,
      [](const auto &entry, int target) { return entry.first < target; });
}

float row_value(const SparseRow &row, int col) {
  auto pos = find_row_col(row, col);
  return (pos != row.end() && pos->first == col) ? pos->second : 0.0f;
}

void set_row_value(SparseRow &row, int col, float value) {
  auto pos = find_row_col(row, col);
  if (value == 0.0f) {
    if (pos != row.end() && pos->first == col) {
      row.erase(pos);
    }
    return;
  }
  if (pos != row.end() && pos->first == col) {
    pos->second = value;
  } else {
    row.insert(pos, {col, value});
  }
}

void add_row_value(SparseRow &row, int col, float delta, float eps) {
  auto pos = find_row_col(row, col);
  if (pos != row.end() && pos->first == col) {
    pos->second += delta;
    if (std::abs(pos->second) <= eps) {
      row.erase(pos);
    }
  } else if (std::abs(delta) > eps) {
    row.insert(pos, {col, delta});
  }
}

template <typename I>
std::vector<SparseRow>
read_csr_sparse_rows_float32(mx::array data, mx::array indices,
                             mx::array indptr, int n_rows, int n_cols) {
  data.eval();
  indices.eval();
  indptr.eval();
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  std::vector<SparseRow> rows(static_cast<size_t>(n_rows));
  for (int row = 0; row < n_rows; ++row) {
    auto &entries = rows[static_cast<size_t>(row)];
    const I begin = indptr_ptr[row];
    const I end = indptr_ptr[row + 1];
    entries.reserve(static_cast<size_t>(end - begin));
    for (I p = begin; p < end; ++p) {
      const int col = static_cast<int>(indices_ptr[p]);
      if (col < 0 || col >= n_cols) {
        throw std::invalid_argument(
            "csr_lu input contains an out-of-bounds column.");
      }
      entries.push_back({col, data_ptr[p]});
    }
    sort_and_sum_row(entries);
  }
  return rows;
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array,
           mx::array>
csr_lu_impl(mx::array data, mx::array indices, mx::array indptr, int n_rows,
            int n_cols, mx::Dtype index_dtype) {
  if (n_rows != n_cols) {
    throw std::invalid_argument("csr_lu requires a square matrix.");
  }
  auto rows = read_csr_sparse_rows_float32<I>(
      std::move(data), std::move(indices), std::move(indptr), n_rows, n_cols);
  std::vector<SparseRow> L(static_cast<size_t>(n_rows));
  std::vector<SparseRow> U(static_cast<size_t>(n_rows));
  std::vector<int32_t> perm(static_cast<size_t>(n_rows));
  std::iota(perm.begin(), perm.end(), 0);
  const float eps = std::numeric_limits<float>::epsilon();

  for (int k = 0; k < n_rows; ++k) {
    int pivot_row = k;
    float pivot_abs = 0.0f;
    for (int row = k; row < n_rows; ++row) {
      const float value = row_value(rows[static_cast<size_t>(row)], k);
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
        const float pivot_value =
            row_value(L[static_cast<size_t>(pivot_row)], col);
        const float current_value = row_value(L[static_cast<size_t>(k)], col);
        set_row_value(L[static_cast<size_t>(pivot_row)], col, current_value);
        set_row_value(L[static_cast<size_t>(k)], col, pivot_value);
      }
    }

    set_row_value(L[static_cast<size_t>(k)], k, 1.0f);
    auto &upper_row = U[static_cast<size_t>(k)];
    upper_row.clear();
    upper_row.reserve(rows[static_cast<size_t>(k)].size());
    for (const auto &[col, value] : rows[static_cast<size_t>(k)]) {
      if (col >= k && std::abs(value) > eps) {
        upper_row.push_back({col, value});
      }
    }
    const float pivot = row_value(upper_row, k);
    for (int row = k + 1; row < n_rows; ++row) {
      auto &work_row = rows[static_cast<size_t>(row)];
      auto entry = find_row_col(work_row, k);
      if (entry == work_row.end() || entry->first != k ||
          std::abs(entry->second) <= eps) {
        continue;
      }
      const float factor = entry->second / pivot;
      set_row_value(L[static_cast<size_t>(row)], k, factor);
      work_row.erase(entry);
      for (const auto &[col, upper_value] : upper_row) {
        if (col > k) {
          add_row_value(work_row, col, -factor * upper_value, eps);
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
  size_t l_nnz = 0;
  size_t u_nnz = 0;
  for (int row = 0; row < n_rows; ++row) {
    l_nnz += L[static_cast<size_t>(row)].size();
    u_nnz += U[static_cast<size_t>(row)].size();
  }
  l_data.reserve(l_nnz);
  l_indices.reserve(l_nnz);
  u_data.reserve(u_nnz);
  u_indices.reserve(u_nnz);
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
