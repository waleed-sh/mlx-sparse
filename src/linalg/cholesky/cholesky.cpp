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

void insert_mirrored_entry_if_missing(SparseRow &row, int col, float value) {
  auto pos = find_row_col(row, col);
  if (pos == row.end() || pos->first != col) {
    row.insert(pos, {col, value});
  }
}

void insert_active_column(std::vector<int> &active, size_t start, int col) {
  auto begin = active.begin() + static_cast<std::ptrdiff_t>(start);
  auto pos = std::lower_bound(begin, active.end(), col);
  if (pos == active.end() || *pos != col) {
    active.insert(pos, col);
  }
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array>
csr_cholesky_impl(mx::array data, mx::array indices, mx::array indptr,
                  int n_rows, int n_cols, mx::Dtype index_dtype) {
  data.eval();
  indices.eval();
  indptr.eval();
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();

  std::vector<SparseRow> lower(static_cast<size_t>(n_rows));
  std::vector<SparseRow> mirrored_upper(static_cast<size_t>(n_rows));
  for (int row = 0; row < n_rows; ++row) {
    for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
      const int col = static_cast<int>(indices_ptr[p]);
      if (col < 0 || col >= n_cols) {
        throw std::invalid_argument(
            "csr_cholesky input contains an out-of-bounds column.");
      }
      const float value = data_ptr[p];
      if (row >= col) {
        lower[static_cast<size_t>(row)].push_back({col, value});
      } else {
        mirrored_upper[static_cast<size_t>(col)].push_back({row, value});
      }
    }
  }
  for (int row = 0; row < n_rows; ++row) {
    auto &lower_row = lower[static_cast<size_t>(row)];
    sort_and_sum_row(lower_row);

    auto &upper_row = mirrored_upper[static_cast<size_t>(row)];
    sort_and_sum_row(upper_row);
    for (const auto &[col, value] : upper_row) {
      insert_mirrored_entry_if_missing(lower_row, col, value);
    }
  }

  std::vector<SparseRow> columns(static_cast<size_t>(n_rows));
  std::vector<float> diag(static_cast<size_t>(n_rows), 0.0f);
  std::vector<float> work(static_cast<size_t>(n_rows), 0.0f);
  std::vector<int> marker(static_cast<size_t>(n_rows), -1);
  std::vector<int> active;
  const float eps = std::numeric_limits<float>::epsilon();

  std::vector<float> out_data;
  std::vector<I> out_indices;
  std::vector<I> out_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  out_data.reserve(data.size());
  out_indices.reserve(indices.size());

  for (int row = 0; row < n_rows; ++row) {
    active.clear();
    const auto &input_row = lower[static_cast<size_t>(row)];
    active.reserve(std::max(active.capacity(), input_row.size() + 1));
    for (const auto &[col, value] : input_row) {
      if (col <= row) {
        marker[static_cast<size_t>(col)] = row;
        work[static_cast<size_t>(col)] = value;
        active.push_back(col);
      }
    }
    if (marker[static_cast<size_t>(row)] != row) {
      marker[static_cast<size_t>(row)] = row;
      work[static_cast<size_t>(row)] = 0.0f;
      insert_active_column(active, 0, row);
    }

    for (size_t active_pos = 0; active_pos < active.size(); ++active_pos) {
      const int pivot_col = active[active_pos];
      if (pivot_col >= row) {
        break;
      }
      if (std::abs(diag[static_cast<size_t>(pivot_col)]) <= eps) {
        throw std::runtime_error("csr_cholesky encountered a zero pivot.");
      }
      const float factor = work[static_cast<size_t>(pivot_col)] /
                           diag[static_cast<size_t>(pivot_col)];
      work[static_cast<size_t>(pivot_col)] = factor;
      for (const auto &[update_col, update_value] :
           columns[static_cast<size_t>(pivot_col)]) {
        if (update_col < row) {
          if (marker[static_cast<size_t>(update_col)] != row) {
            marker[static_cast<size_t>(update_col)] = row;
            work[static_cast<size_t>(update_col)] = 0.0f;
            insert_active_column(active, active_pos + 1, update_col);
          }
          work[static_cast<size_t>(update_col)] -= factor * update_value;
        }
      }
      work[static_cast<size_t>(row)] -= factor * factor;
      columns[static_cast<size_t>(pivot_col)].push_back({row, factor});
    }
    const float diag_value = work[static_cast<size_t>(row)];
    if (diag_value <= eps) {
      throw std::runtime_error(
          "csr_cholesky requires a positive-definite matrix.");
    }
    diag[static_cast<size_t>(row)] = std::sqrt(diag_value);
    work[static_cast<size_t>(row)] = diag[static_cast<size_t>(row)];

    for (const int col : active) {
      const float value = work[static_cast<size_t>(col)];
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
