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

#include "preconditioners/ic0/ic0.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "linalg/common/common.h"
#include "linalg/triangular_solve/triangular_solve.h"
#include "mlx/ops.h"

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

struct SparseEntry {
  int col;
  float value;
};

using SparseRow = std::vector<SparseEntry>;

void sort_and_sum_row(SparseRow &row) {
  std::sort(row.begin(), row.end(),
            [](const SparseEntry &lhs, const SparseEntry &rhs) {
              return lhs.col < rhs.col;
            });
  size_t write = 0;
  for (const auto &entry : row) {
    if (write > 0 && row[write - 1].col == entry.col) {
      row[write - 1].value += entry.value;
    } else {
      row[write++] = entry;
    }
  }
  row.resize(write);
}

SparseRow::iterator find_entry(SparseRow &row, int col) {
  return std::lower_bound(
      row.begin(), row.end(), col,
      [](const SparseEntry &entry, int target) { return entry.col < target; });
}

SparseRow::const_iterator find_entry(const SparseRow &row, int col) {
  return std::lower_bound(
      row.begin(), row.end(), col,
      [](const SparseEntry &entry, int target) { return entry.col < target; });
}

float entry_value(const SparseRow &row, int col) {
  auto pos = find_entry(row, col);
  if (pos == row.end() || pos->col != col) {
    return 0.0f;
  }
  return pos->value;
}

float row_abs_sum(const SparseRow &row) {
  double sum = 0.0;
  for (const auto &entry : row) {
    sum += std::abs(static_cast<double>(entry.value));
  }
  return static_cast<float>(sum);
}

float pivot_threshold(const SparseRow &row, bool check) {
  if (!check) {
    return 0.0f;
  }
  const float scale = std::max(1.0f, row_abs_sum(row));
  return std::numeric_limits<float>::epsilon() * scale;
}

float symmetry_threshold(float lhs, float rhs) {
  const float scale = std::max({1.0f, std::abs(lhs), std::abs(rhs)});
  return 16.0f * std::numeric_limits<float>::epsilon() * scale;
}

template <typename I>
std::vector<SparseRow>
read_canonical_csr_rows(mx::array data, mx::array indices, mx::array indptr,
                        int n_rows, int n_cols) {
  data.eval();
  indices.eval();
  indptr.eval();
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  const size_t nnz = data.size();

  if (indptr_ptr[n_rows] < I{0} ||
      static_cast<size_t>(indptr_ptr[n_rows]) != nnz) {
    throw std::invalid_argument(
        "csr_ic0 input terminal row offset must equal nnz.");
  }

  std::vector<SparseRow> rows(static_cast<size_t>(n_rows));
  for (int row = 0; row < n_rows; ++row) {
    const I begin = indptr_ptr[row];
    const I end = indptr_ptr[row + 1];
    if (begin > end || begin < I{0} || static_cast<size_t>(end) > nnz) {
      throw std::invalid_argument("csr_ic0 input has invalid CSR row offsets.");
    }
    auto &entries = rows[static_cast<size_t>(row)];
    entries.reserve(static_cast<size_t>(end - begin));
    for (I p = begin; p < end; ++p) {
      const I raw_col = indices_ptr[p];
      if (raw_col < I{0} || raw_col >= static_cast<I>(n_cols)) {
        throw std::invalid_argument(
            "csr_ic0 input contains an out-of-bounds column.");
      }
      const float value = data_ptr[p];
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "csr_ic0 input contains a non-finite value.");
      }
      entries.push_back({static_cast<int>(raw_col), value});
    }
    sort_and_sum_row(entries);
  }
  return rows;
}

std::vector<SparseRow>
build_symmetric_lower_pattern(const std::vector<SparseRow> &rows, bool check) {
  const int n_rows = static_cast<int>(rows.size());
  std::vector<SparseRow> lower(static_cast<size_t>(n_rows));

  for (int row = 0; row < n_rows; ++row) {
    for (const auto &entry : rows[static_cast<size_t>(row)]) {
      if (entry.col <= row) {
        lower[static_cast<size_t>(row)].push_back(entry);
      }
    }
  }
  for (auto &row : lower) {
    sort_and_sum_row(row);
  }

  for (int row = 0; row < n_rows; ++row) {
    for (const auto &entry : rows[static_cast<size_t>(row)]) {
      const int col = entry.col;
      if (col <= row) {
        continue;
      }
      auto &mirrored_row = lower[static_cast<size_t>(col)];
      auto direct = find_entry(mirrored_row, row);
      if (direct == mirrored_row.end() || direct->col != row) {
        mirrored_row.push_back({row, entry.value});
        continue;
      }
      if (check && std::abs(direct->value - entry.value) >
                       symmetry_threshold(direct->value, entry.value)) {
        throw std::runtime_error(
            "csr_ic0 requires symmetric numeric values when check=True.");
      }
    }
  }

  for (auto &row : lower) {
    sort_and_sum_row(row);
  }
  return lower;
}

void validate_diagonal_and_apply_shift(std::vector<SparseRow> &lower,
                                       float shift) {
  if (!std::isfinite(shift)) {
    throw std::invalid_argument("csr_ic0 shift must be finite.");
  }
  if (shift < 0.0f) {
    throw std::invalid_argument("csr_ic0 shift must be non-negative.");
  }
  for (int row = 0; row < static_cast<int>(lower.size()); ++row) {
    auto &entries = lower[static_cast<size_t>(row)];
    auto diag = find_entry(entries, row);
    if (diag == entries.end() || diag->col != row) {
      throw std::runtime_error(
          "csr_ic0 requires every row to contain an explicit diagonal entry.");
    }
    diag->value += shift;
    if (!std::isfinite(diag->value)) {
      throw std::runtime_error(
          "csr_ic0 shifted diagonal produced a non-finite value.");
    }
  }
}

float computed_entry(const SparseRow &row, int col) {
  auto pos = find_entry(row, col);
  if (pos == row.end() || pos->col != col) {
    return 0.0f;
  }
  return pos->value;
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array>
csr_ic0_impl(mx::array data, mx::array indices, mx::array indptr, int n_rows,
             int n_cols, float shift, bool check, mx::Dtype index_dtype) {
  auto rows = read_canonical_csr_rows<I>(std::move(data), std::move(indices),
                                         std::move(indptr), n_rows, n_cols);
  auto lower_pattern = build_symmetric_lower_pattern(rows, check);
  validate_diagonal_and_apply_shift(lower_pattern, shift);

  std::vector<SparseRow> factor(static_cast<size_t>(n_rows));
  for (int row = 0; row < n_rows; ++row) {
    const auto &pattern_row = lower_pattern[static_cast<size_t>(row)];
    SparseRow l_row;
    l_row.reserve(pattern_row.size());

    for (const auto &entry : pattern_row) {
      const int col = entry.col;
      if (col >= row) {
        break;
      }
      double sum = static_cast<double>(entry.value);
      for (const auto &known : l_row) {
        const int k = known.col;
        if (k >= col) {
          break;
        }
        const float ljk = computed_entry(factor[static_cast<size_t>(col)], k);
        if (ljk != 0.0f) {
          sum -= static_cast<double>(known.value) * static_cast<double>(ljk);
        }
      }
      const float diag_j = entry_value(factor[static_cast<size_t>(col)], col);
      const float threshold =
          pivot_threshold(factor[static_cast<size_t>(col)], check);
      if (!std::isfinite(diag_j) || diag_j <= threshold) {
        throw std::runtime_error(
            "csr_ic0 encountered a non-positive or near-zero pivot.");
      }
      const float value = static_cast<float>(sum) / diag_j;
      if (!std::isfinite(value)) {
        throw std::runtime_error(
            "csr_ic0 produced a non-finite lower factor entry.");
      }
      l_row.push_back({col, value});
    }

    auto diag_entry = find_entry(pattern_row, row);
    if (diag_entry == pattern_row.end() || diag_entry->col != row) {
      throw std::runtime_error("csr_ic0 encountered a missing diagonal pivot.");
    }
    double diag = static_cast<double>(diag_entry->value);
    for (const auto &known : l_row) {
      diag -=
          static_cast<double>(known.value) * static_cast<double>(known.value);
    }
    const float diag_threshold = pivot_threshold(pattern_row, check);
    if (!std::isfinite(diag) || diag <= static_cast<double>(diag_threshold)) {
      throw std::runtime_error(
          "csr_ic0 encountered a non-positive or near-zero pivot.");
    }
    const float diag_factor = std::sqrt(static_cast<float>(diag));
    if (!std::isfinite(diag_factor)) {
      throw std::runtime_error(
          "csr_ic0 produced a non-finite diagonal factor.");
    }
    l_row.push_back({row, diag_factor});
    factor[static_cast<size_t>(row)] = std::move(l_row);
  }

  std::vector<float> l_data;
  std::vector<I> l_indices;
  std::vector<I> l_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  size_t nnz = 0;
  for (const auto &row : factor) {
    nnz += row.size();
  }
  l_data.reserve(nnz);
  l_indices.reserve(nnz);
  for (int row = 0; row < n_rows; ++row) {
    const auto &entries = factor[static_cast<size_t>(row)];
    for (const auto &entry : entries) {
      if (entry.col > row) {
        throw std::runtime_error("csr_ic0 produced a non-lower factor entry.");
      }
      l_data.push_back(entry.value);
      l_indices.push_back(static_cast<I>(entry.col));
    }
    l_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(l_data.size());
  }

  return make_csr_arrays_float32(l_data, l_indices, l_indptr, index_dtype);
}

void validate_ic0_factor_csr(const mx::array &data, const mx::array &indices,
                             const mx::array &indptr, int n_rows, int n_cols,
                             const char *context) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(std::string(context) +
                                " requires a non-empty square factor.");
  }
  require_rank(data, 1, context);
  require_rank(indices, 1, context);
  require_rank(indptr, 1, context);
  require_linalg_float32(data, context);
  require_same_index_dtype(indices, indptr, context, context);
  require_size(indptr, n_rows + 1, context);
  if (indices.size() != data.size()) {
    throw std::invalid_argument(std::string(context) +
                                " data and indices must have equal length.");
  }
}

void validate_ic0_rhs(const mx::array &rhs, int n_rows, const char *context) {
  if (rhs.ndim() != 1 && rhs.ndim() != 2) {
    throw std::invalid_argument(std::string(context) +
                                " rhs must be rank-1 or rank-2.");
  }
  require_linalg_float32(rhs, context);
  if (rhs.shape(0) != n_rows) {
    throw std::invalid_argument(std::string(context) +
                                " rhs has incompatible leading dimension.");
  }
  if (rhs.ndim() == 2 && rhs.shape(1) <= 0) {
    throw std::invalid_argument(
        std::string(context) + " rank-2 rhs must include at least one column.");
  }
}

} // namespace

std::tuple<mx::array, mx::array, mx::array> csr_ic0(const mx::array &data,
                                                    const mx::array &indices,
                                                    const mx::array &indptr,
                                                    int n_rows, int n_cols,
                                                    float shift, bool check) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_ic0 requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_ic0 data");
  require_rank(indices, 1, "csr_ic0 indices");
  require_rank(indptr, 1, "csr_ic0 indptr");
  require_linalg_float32(data, "csr_ic0 data");
  require_same_index_dtype(indices, indptr, "csr_ic0 indices",
                           "csr_ic0 indptr");
  require_size(indptr, n_rows + 1, "csr_ic0 indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_ic0 data and indices must have equal length.");
  }
  if (indices.dtype() == mx::int32) {
    return csr_ic0_impl<int32_t>(data, indices, indptr, n_rows, n_cols, shift,
                                 check, mx::int32);
  }
  if (indices.dtype() == mx::int64) {
    return csr_ic0_impl<int64_t>(data, indices, indptr, n_rows, n_cols, shift,
                                 check, mx::int64);
  }
  throw std::runtime_error("csr_ic0 requires int32 or int64 indices.");
}

mx::array csr_ic0_preconditioner_apply(
    const mx::array &l_data, const mx::array &l_indices,
    const mx::array &l_indptr, const mx::array &lt_data,
    const mx::array &lt_indices, const mx::array &lt_indptr,
    const mx::array &rhs, int n_rows, int n_cols, mx::StreamOrDevice s) {
  validate_ic0_factor_csr(l_data, l_indices, l_indptr, n_rows, n_cols,
                          "csr_ic0_preconditioner_apply L");
  validate_ic0_factor_csr(lt_data, lt_indices, lt_indptr, n_rows, n_cols,
                          "csr_ic0_preconditioner_apply LT");
  if (l_indices.dtype() != lt_indices.dtype() ||
      l_indptr.dtype() != lt_indptr.dtype()) {
    throw std::invalid_argument(
        "csr_ic0_preconditioner_apply L and LT index dtypes must match.");
  }
  validate_ic0_rhs(rhs, n_rows, "csr_ic0_preconditioner_apply");

  auto stream = mx::to_stream(s);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  auto y = csr_triangular_solve(l_data, l_indices, l_indptr, rhs_contig, n_rows,
                                n_cols, true, false, stream);
  return csr_triangular_solve(lt_data, lt_indices, lt_indptr, y, n_rows, n_cols,
                              false, false, stream);
}

} // namespace mlx_sparse
