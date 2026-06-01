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

#include "preconditioners/ilu0/ilu0.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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

float row_abs_sum(const SparseRow &row) {
  double sum = 0.0;
  for (const auto &entry : row) {
    sum += std::abs(static_cast<double>(entry.value));
  }
  return static_cast<float>(sum);
}

bool has_row_entry(const SparseRow &row, int col) {
  auto pos = find_entry(row, col);
  return pos != row.end() && pos->col == col;
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
        "csr_ilu0 input terminal row offset must equal nnz.");
  }

  std::vector<SparseRow> rows(static_cast<size_t>(n_rows));
  for (int row = 0; row < n_rows; ++row) {
    const I begin = indptr_ptr[row];
    const I end = indptr_ptr[row + 1];
    if (begin > end || begin < I{0} || static_cast<size_t>(end) > nnz) {
      throw std::invalid_argument(
          "csr_ilu0 input has invalid CSR row offsets.");
    }
    auto &entries = rows[static_cast<size_t>(row)];
    entries.reserve(static_cast<size_t>(end - begin));
    for (I p = begin; p < end; ++p) {
      const I raw_col = indices_ptr[p];
      if (raw_col < I{0} || raw_col >= static_cast<I>(n_cols)) {
        throw std::invalid_argument(
            "csr_ilu0 input contains an out-of-bounds column.");
      }
      const int col = static_cast<int>(raw_col);
      const float value = data_ptr[p];
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "csr_ilu0 input contains a non-finite value.");
      }
      entries.push_back({col, value});
    }
    sort_and_sum_row(entries);
  }
  return rows;
}

void validate_existing_diagonal_and_apply_shift(std::vector<SparseRow> &rows,
                                                float shift) {
  for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
    auto &entries = rows[static_cast<size_t>(row)];
    auto diag = find_entry(entries, row);
    if (diag == entries.end() || diag->col != row) {
      throw std::runtime_error("csr_ilu0 requires every row to contain an "
                               "explicit diagonal entry.");
    }
    diag->value += shift;
    if (!std::isfinite(diag->value)) {
      throw std::runtime_error(
          "csr_ilu0 shifted diagonal produced a non-finite value.");
    }
  }
}

float pivot_threshold(const SparseRow &row, bool check) {
  if (!check) {
    return 0.0f;
  }
  const float scale = std::max(1.0f, row_abs_sum(row));
  return std::numeric_limits<float>::epsilon() * scale;
}

template <typename I>
std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array>
csr_ilu0_impl(mx::array data, mx::array indices, mx::array indptr, int n_rows,
              int n_cols, float shift, bool check, mx::Dtype index_dtype) {
  if (!std::isfinite(shift)) {
    throw std::invalid_argument("csr_ilu0 shift must be finite.");
  }
  auto rows = read_canonical_csr_rows<I>(std::move(data), std::move(indices),
                                         std::move(indptr), n_rows, n_cols);
  validate_existing_diagonal_and_apply_shift(rows, shift);

  std::vector<SparseRow> lower(static_cast<size_t>(n_rows));
  std::vector<SparseRow> upper(static_cast<size_t>(n_rows));
  std::vector<float> upper_diagonal(static_cast<size_t>(n_rows), 0.0f);

  for (int row = 0; row < n_rows; ++row) {
    SparseRow work = rows[static_cast<size_t>(row)];

    for (auto &entry : work) {
      const int k = entry.col;
      if (k >= row) {
        break;
      }
      const float pivot = upper_diagonal[static_cast<size_t>(k)];
      if (!std::isfinite(pivot) ||
          std::abs(pivot) <=
              pivot_threshold(upper[static_cast<size_t>(k)], check)) {
        throw std::runtime_error(
            "csr_ilu0 encountered a zero or near-zero pivot.");
      }

      const float factor = entry.value / pivot;
      if (!std::isfinite(factor)) {
        throw std::runtime_error(
            "csr_ilu0 produced a non-finite lower factor entry.");
      }
      entry.value = factor;

      const auto &upper_k = upper[static_cast<size_t>(k)];
      for (const auto &u_entry : upper_k) {
        const int col = u_entry.col;
        if (col <= k) {
          continue;
        }
        auto pos = find_entry(work, col);
        if (pos != work.end() && pos->col == col) {
          pos->value -= factor * u_entry.value;
          if (!std::isfinite(pos->value)) {
            throw std::runtime_error(
                "csr_ilu0 update produced a non-finite factor entry.");
          }
        }
      }
    }

    auto diag = find_entry(work, row);
    if (diag == work.end() || diag->col != row) {
      throw std::runtime_error(
          "csr_ilu0 encountered a structurally singular pivot.");
    }
    const float threshold = pivot_threshold(work, check);
    if (!std::isfinite(diag->value) || std::abs(diag->value) <= threshold) {
      throw std::runtime_error(
          "csr_ilu0 encountered a zero or near-zero pivot.");
    }
    upper_diagonal[static_cast<size_t>(row)] = diag->value;

    auto &l_row = lower[static_cast<size_t>(row)];
    auto &u_row = upper[static_cast<size_t>(row)];
    l_row.reserve(work.size());
    u_row.reserve(work.size());
    for (const auto &entry : work) {
      if (entry.col < row) {
        l_row.push_back(entry);
      } else {
        u_row.push_back(entry);
      }
    }
    if (!has_row_entry(l_row, row)) {
      l_row.push_back({row, 1.0f});
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
    l_nnz += lower[static_cast<size_t>(row)].size();
    u_nnz += upper[static_cast<size_t>(row)].size();
  }
  l_data.reserve(l_nnz);
  l_indices.reserve(l_nnz);
  u_data.reserve(u_nnz);
  u_indices.reserve(u_nnz);

  for (int row = 0; row < n_rows; ++row) {
    auto &l_row = lower[static_cast<size_t>(row)];
    sort_and_sum_row(l_row);
    for (const auto &entry : l_row) {
      if (entry.col <= row) {
        l_data.push_back(entry.value);
        l_indices.push_back(static_cast<I>(entry.col));
      }
    }
    l_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(l_data.size());

    auto &u_row = upper[static_cast<size_t>(row)];
    sort_and_sum_row(u_row);
    for (const auto &entry : u_row) {
      if (entry.col >= row) {
        u_data.push_back(entry.value);
        u_indices.push_back(static_cast<I>(entry.col));
      }
    }
    u_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(u_data.size());
  }

  auto [l_data_array, l_indices_array, l_indptr_array] =
      make_csr_arrays_float32(l_data, l_indices, l_indptr, index_dtype);
  auto [u_data_array, u_indices_array, u_indptr_array] =
      make_csr_arrays_float32(u_data, u_indices, u_indptr, index_dtype);
  return {l_data_array, l_indices_array, l_indptr_array,
          u_data_array, u_indices_array, u_indptr_array};
}

void validate_ilu0_factor_csr(const mx::array &data, const mx::array &indices,
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

void validate_ilu0_apply_inputs(const mx::array &l_data,
                                const mx::array &l_indices,
                                const mx::array &l_indptr,
                                const mx::array &u_data,
                                const mx::array &u_indices,
                                const mx::array &u_indptr, const mx::array &rhs,
                                int n_rows, int n_cols) {
  validate_ilu0_factor_csr(l_data, l_indices, l_indptr, n_rows, n_cols,
                           "csr_ilu0_preconditioner_apply L");
  validate_ilu0_factor_csr(u_data, u_indices, u_indptr, n_rows, n_cols,
                           "csr_ilu0_preconditioner_apply U");
  if (l_indices.dtype() != u_indices.dtype() ||
      l_indptr.dtype() != u_indptr.dtype()) {
    throw std::invalid_argument(
        "csr_ilu0_preconditioner_apply L and U index dtypes must match.");
  }
  if (rhs.ndim() != 1 && rhs.ndim() != 2) {
    throw std::invalid_argument(
        "csr_ilu0_preconditioner_apply rhs must be rank-1 or rank-2.");
  }
  require_linalg_float32(rhs, "csr_ilu0_preconditioner_apply rhs");
  if (rhs.shape(0) != n_rows) {
    throw std::invalid_argument("csr_ilu0_preconditioner_apply rhs has "
                                "incompatible leading dimension.");
  }
  if (rhs.ndim() == 2 && rhs.shape(1) <= 0) {
    throw std::invalid_argument(
        "csr_ilu0_preconditioner_apply rank-2 rhs must include at least one "
        "column.");
  }
}

} // namespace

std::tuple<mx::array, mx::array, mx::array, mx::array, mx::array, mx::array>
csr_ilu0(const mx::array &data, const mx::array &indices,
         const mx::array &indptr, int n_rows, int n_cols, float shift,
         bool check) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_ilu0 requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_ilu0 data");
  require_rank(indices, 1, "csr_ilu0 indices");
  require_rank(indptr, 1, "csr_ilu0 indptr");
  require_linalg_float32(data, "csr_ilu0 data");
  require_same_index_dtype(indices, indptr, "csr_ilu0 indices",
                           "csr_ilu0 indptr");
  require_size(indptr, n_rows + 1, "csr_ilu0 indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_ilu0 data and indices must have equal length.");
  }
  if (indices.dtype() == mx::int32) {
    return csr_ilu0_impl<int32_t>(data, indices, indptr, n_rows, n_cols, shift,
                                  check, mx::int32);
  }
  if (indices.dtype() == mx::int64) {
    return csr_ilu0_impl<int64_t>(data, indices, indptr, n_rows, n_cols, shift,
                                  check, mx::int64);
  }
  throw std::runtime_error("csr_ilu0 requires int32 or int64 indices.");
}

mx::array csr_ilu0_preconditioner_apply(
    const mx::array &l_data, const mx::array &l_indices,
    const mx::array &l_indptr, const mx::array &u_data,
    const mx::array &u_indices, const mx::array &u_indptr, const mx::array &rhs,
    int n_rows, int n_cols, mx::StreamOrDevice s) {
  validate_ilu0_apply_inputs(l_data, l_indices, l_indptr, u_data, u_indices,
                             u_indptr, rhs, n_rows, n_cols);
  auto stream = mx::to_stream(s);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  auto y = csr_triangular_solve(l_data, l_indices, l_indptr, rhs_contig, n_rows,
                                n_cols, true, true, stream);
  return csr_triangular_solve(u_data, u_indices, u_indptr, y, n_rows, n_cols,
                              false, false, stream);
}

} // namespace mlx_sparse
