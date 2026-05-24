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

#include "sparse/coo_matmat/coo_matmat.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlx_sparse {

namespace {

template <typename T> bool nonzero(T value) { return value != T{}; }

template <typename I>
int prefix_counts(const std::vector<I> &counts, std::vector<I> &indptr) {
  indptr.resize(counts.size() + 1);
  indptr[0] = I{0};
  long long total = 0;
  for (size_t i = 0; i < counts.size(); ++i) {
    if (counts[i] < I{0}) {
      throw std::runtime_error("coo_matmat produced a negative row count.");
    }
    total += static_cast<long long>(counts[i]);
    if (total > static_cast<long long>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "coo_matmat output nnz exceeds MLX shape limits.");
    }
    if constexpr (std::is_same_v<I, int32_t>) {
      if (total > static_cast<long long>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error(
            "coo_matmat output nnz exceeds int32 index capacity.");
      }
    }
    indptr[i + 1] = static_cast<I>(total);
  }
  return static_cast<int>(total);
}

std::vector<int> offsets_from_counts(const std::vector<int> &counts) {
  std::vector<int> offsets(counts.size() + 1, 0);
  for (size_t i = 0; i < counts.size(); ++i) {
    const long long next = static_cast<long long>(offsets[i]) + counts[i];
    if (next > static_cast<long long>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "coo_matmat input nnz exceeds supported limits.");
    }
    offsets[i + 1] = static_cast<int>(next);
  }
  return offsets;
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
std::tuple<mx::array, mx::array, mx::array>
coo_matmat_impl(mx::array lhs_data, mx::array lhs_row, mx::array lhs_col,
                mx::array rhs_data, mx::array rhs_row, mx::array rhs_col,
                int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
                mx::Dtype out_index_dtype) {
  using AccT = typename Accumulator<T>::Type;

  lhs_data.eval();
  lhs_row.eval();
  lhs_col.eval();
  rhs_data.eval();
  rhs_row.eval();
  rhs_col.eval();

  if (lhs_data.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      rhs_data.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("coo_matmat input nnz exceeds supported limits.");
  }

  const int lhs_nnz = static_cast<int>(lhs_data.size());
  const int rhs_nnz = static_cast<int>(rhs_data.size());
  const auto *lhs_data_ptr = lhs_data.data<T>();
  const auto *lhs_row_ptr = lhs_row.data<LhsI>();
  const auto *lhs_col_ptr = lhs_col.data<LhsI>();
  const auto *rhs_data_ptr = rhs_data.data<T>();
  const auto *rhs_row_ptr = rhs_row.data<RhsI>();
  const auto *rhs_col_ptr = rhs_col.data<RhsI>();

  std::vector<int> lhs_counts(static_cast<size_t>(lhs_n_rows), 0);
  std::vector<int> rhs_counts(static_cast<size_t>(rhs_n_rows), 0);
  for (int p = 0; p < lhs_nnz; ++p) {
    const int row = static_cast<int>(lhs_row_ptr[p]);
    const int col = static_cast<int>(lhs_col_ptr[p]);
    if (row < 0 || row >= lhs_n_rows || col < 0 || col >= lhs_n_cols) {
      throw std::invalid_argument(
          "coo_matmat lhs coordinates contain an out-of-bounds entry.");
    }
    lhs_counts[static_cast<size_t>(row)] += 1;
  }
  for (int p = 0; p < rhs_nnz; ++p) {
    const int row = static_cast<int>(rhs_row_ptr[p]);
    const int col = static_cast<int>(rhs_col_ptr[p]);
    if (row < 0 || row >= rhs_n_rows || col < 0 || col >= rhs_n_cols) {
      throw std::invalid_argument(
          "coo_matmat rhs coordinates contain an out-of-bounds entry.");
    }
    rhs_counts[static_cast<size_t>(row)] += 1;
  }

  const auto lhs_offsets = offsets_from_counts(lhs_counts);
  const auto rhs_offsets = offsets_from_counts(rhs_counts);
  std::vector<int> lhs_positions(static_cast<size_t>(lhs_nnz));
  std::vector<int> rhs_positions(static_cast<size_t>(rhs_nnz));
  auto lhs_cursor = lhs_offsets;
  auto rhs_cursor = rhs_offsets;
  for (int p = 0; p < lhs_nnz; ++p) {
    const int row = static_cast<int>(lhs_row_ptr[p]);
    lhs_positions[static_cast<size_t>(lhs_cursor[static_cast<size_t>(row)]++)] =
        p;
  }
  for (int p = 0; p < rhs_nnz; ++p) {
    const int row = static_cast<int>(rhs_row_ptr[p]);
    rhs_positions[static_cast<size_t>(rhs_cursor[static_cast<size_t>(row)]++)] =
        p;
  }

  std::vector<int> marker(static_cast<size_t>(rhs_n_cols), -1);
  std::vector<OutI> candidate_counts(static_cast<size_t>(lhs_n_rows), OutI{0});
  std::vector<int> columns;
  for (int row = 0; row < lhs_n_rows; ++row) {
    int row_count = 0;
    for (int lp = lhs_offsets[static_cast<size_t>(row)];
         lp < lhs_offsets[static_cast<size_t>(row) + 1]; ++lp) {
      const int lhs_pos = lhs_positions[static_cast<size_t>(lp)];
      const int rhs_row = static_cast<int>(lhs_col_ptr[lhs_pos]);
      for (int rp = rhs_offsets[static_cast<size_t>(rhs_row)];
           rp < rhs_offsets[static_cast<size_t>(rhs_row) + 1]; ++rp) {
        const int rhs_pos = rhs_positions[static_cast<size_t>(rp)];
        const int col = static_cast<int>(rhs_col_ptr[rhs_pos]);
        if (marker[static_cast<size_t>(col)] != row) {
          marker[static_cast<size_t>(col)] = row;
          row_count += 1;
        }
      }
    }
    candidate_counts[static_cast<size_t>(row)] = static_cast<OutI>(row_count);
  }

  std::vector<OutI> candidate_indptr;
  const int candidate_nnz = prefix_counts(candidate_counts, candidate_indptr);
  std::vector<T> candidate_data(static_cast<size_t>(candidate_nnz));
  std::vector<OutI> candidate_col(static_cast<size_t>(candidate_nnz));

  std::fill(marker.begin(), marker.end(), -1);
  std::vector<AccT> accum(static_cast<size_t>(rhs_n_cols),
                          Accumulator<T>::zero());
  for (int row = 0; row < lhs_n_rows; ++row) {
    columns.clear();
    for (int lp = lhs_offsets[static_cast<size_t>(row)];
         lp < lhs_offsets[static_cast<size_t>(row) + 1]; ++lp) {
      const int lhs_pos = lhs_positions[static_cast<size_t>(lp)];
      const int rhs_row = static_cast<int>(lhs_col_ptr[lhs_pos]);
      const T lhs_value = lhs_data_ptr[lhs_pos];
      for (int rp = rhs_offsets[static_cast<size_t>(rhs_row)];
           rp < rhs_offsets[static_cast<size_t>(rhs_row) + 1]; ++rp) {
        const int rhs_pos = rhs_positions[static_cast<size_t>(rp)];
        const int col = static_cast<int>(rhs_col_ptr[rhs_pos]);
        const auto col_index = static_cast<size_t>(col);
        if (marker[col_index] != row) {
          marker[col_index] = row;
          accum[col_index] = Accumulator<T>::zero();
          columns.push_back(col);
        }
        accum[col_index] +=
            multiply_accumulate<T>(lhs_value, rhs_data_ptr[rhs_pos]);
      }
    }

    std::sort(columns.begin(), columns.end());
    OutI write = candidate_indptr[static_cast<size_t>(row)];
    for (int col : columns) {
      candidate_col[static_cast<size_t>(write)] = static_cast<OutI>(col);
      candidate_data[static_cast<size_t>(write)] =
          Accumulator<T>::cast(accum[static_cast<size_t>(col)]);
      ++write;
    }
  }

  std::vector<OutI> out_counts(static_cast<size_t>(lhs_n_rows), OutI{0});
  for (int row = 0; row < lhs_n_rows; ++row) {
    OutI count = OutI{0};
    for (OutI p = candidate_indptr[static_cast<size_t>(row)];
         p < candidate_indptr[static_cast<size_t>(row) + 1]; ++p) {
      if (nonzero(candidate_data[static_cast<size_t>(p)])) {
        count += OutI{1};
      }
    }
    out_counts[static_cast<size_t>(row)] = count;
  }

  std::vector<OutI> out_indptr;
  const int out_nnz = prefix_counts(out_counts, out_indptr);
  std::vector<T> out_data;
  std::vector<OutI> out_row;
  std::vector<OutI> out_col;
  out_data.reserve(static_cast<size_t>(out_nnz));
  out_row.reserve(static_cast<size_t>(out_nnz));
  out_col.reserve(static_cast<size_t>(out_nnz));
  for (int row = 0; row < lhs_n_rows; ++row) {
    for (OutI p = candidate_indptr[static_cast<size_t>(row)];
         p < candidate_indptr[static_cast<size_t>(row) + 1]; ++p) {
      const auto value = candidate_data[static_cast<size_t>(p)];
      if (nonzero(value)) {
        out_data.push_back(value);
        out_row.push_back(static_cast<OutI>(row));
        out_col.push_back(candidate_col[static_cast<size_t>(p)]);
      }
    }
  }

  return {mx::array(out_data.begin(), mx::Shape{out_nnz}, lhs_data.dtype()),
          mx::array(out_row.begin(), mx::Shape{out_nnz}, out_index_dtype),
          mx::array(out_col.begin(), mx::Shape{out_nnz}, out_index_dtype)};
}

template <typename T, typename LhsI, typename RhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_out(mx::array lhs_data, mx::array lhs_row, mx::array lhs_col,
             mx::array rhs_data, mx::array rhs_row, mx::array rhs_col,
             int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
             mx::Dtype out_index_dtype) {
  if (out_index_dtype == mx::int32) {
    return coo_matmat_impl<T, LhsI, RhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
        std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return coo_matmat_impl<T, LhsI, RhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
      std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
      lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
}

template <typename T, typename LhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_rhs(mx::array lhs_data, mx::array lhs_row, mx::array lhs_col,
             mx::array rhs_data, mx::array rhs_row, mx::array rhs_col,
             int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
             mx::Dtype out_index_dtype) {
  if (rhs_row.dtype() == mx::int32) {
    return dispatch_out<T, LhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
        std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return dispatch_out<T, LhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
      std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
      lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
}

template <typename T>
std::tuple<mx::array, mx::array, mx::array>
dispatch_lhs(mx::array lhs_data, mx::array lhs_row, mx::array lhs_col,
             mx::array rhs_data, mx::array rhs_row, mx::array rhs_col,
             int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
             mx::Dtype out_index_dtype) {
  if (lhs_row.dtype() == mx::int32) {
    return dispatch_rhs<T, int32_t>(
        std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
        std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return dispatch_rhs<T, int64_t>(
      std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
      std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
      lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
}

} // namespace

std::tuple<mx::array, mx::array, mx::array>
coo_matmat(const mx::array &lhs_data, const mx::array &lhs_row,
           const mx::array &lhs_col, const mx::array &rhs_data,
           const mx::array &rhs_row, const mx::array &rhs_col, int lhs_n_rows,
           int lhs_n_cols, int rhs_n_rows, int rhs_n_cols) {
  if (lhs_n_rows < 0 || lhs_n_cols < 0 || rhs_n_rows < 0 || rhs_n_cols < 0) {
    throw std::invalid_argument(
        "coo_matmat shape dimensions must be non-negative.");
  }
  if (lhs_n_cols != rhs_n_rows) {
    throw std::invalid_argument("COO sparse-sparse matmul dimension mismatch.");
  }
  require_rank(lhs_data, 1, "coo_matmat lhs_data");
  require_rank(lhs_row, 1, "coo_matmat lhs_row");
  require_rank(lhs_col, 1, "coo_matmat lhs_col");
  require_rank(rhs_data, 1, "coo_matmat rhs_data");
  require_rank(rhs_row, 1, "coo_matmat rhs_row");
  require_rank(rhs_col, 1, "coo_matmat rhs_col");
  require_same_value_dtype(lhs_data, rhs_data, "coo_matmat lhs_data",
                           "coo_matmat rhs_data");
  require_same_index_dtype(lhs_row, lhs_col, "coo_matmat lhs_row",
                           "coo_matmat lhs_col");
  require_same_index_dtype(rhs_row, rhs_col, "coo_matmat rhs_row",
                           "coo_matmat rhs_col");
  if (lhs_data.size() != lhs_row.size() || lhs_data.size() != lhs_col.size() ||
      rhs_data.size() != rhs_row.size() || rhs_data.size() != rhs_col.size()) {
    throw std::invalid_argument(
        "coo_matmat data and coordinate arrays must have equal lengths.");
  }
  if (rhs_n_cols > std::numeric_limits<int>::max()) {
    throw std::overflow_error("coo_matmat n_cols exceeds supported limits.");
  }

  const auto out_index_dtype =
      lhs_row.dtype() == rhs_row.dtype() ? lhs_row.dtype() : mx::int64;
  if (out_index_dtype == mx::int32 &&
      (lhs_n_rows > std::numeric_limits<int32_t>::max() ||
       rhs_n_cols > std::numeric_limits<int32_t>::max())) {
    throw std::overflow_error(
        "coo_matmat output shape exceeds int32 index capacity.");
  }

  if (lhs_data.dtype() == mx::float32) {
    return dispatch_lhs<float>(lhs_data, lhs_row, lhs_col, rhs_data, rhs_row,
                               rhs_col, lhs_n_rows, lhs_n_cols, rhs_n_rows,
                               rhs_n_cols, out_index_dtype);
  }
  if (lhs_data.dtype() == mx::float16) {
    return dispatch_lhs<mx::float16_t>(lhs_data, lhs_row, lhs_col, rhs_data,
                                       rhs_row, rhs_col, lhs_n_rows, lhs_n_cols,
                                       rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (lhs_data.dtype() == mx::bfloat16) {
    return dispatch_lhs<mx::bfloat16_t>(
        lhs_data, lhs_row, lhs_col, rhs_data, rhs_row, rhs_col, lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (lhs_data.dtype() == mx::complex64) {
    return dispatch_lhs<mx::complex64_t>(
        lhs_data, lhs_row, lhs_col, rhs_data, rhs_row, rhs_col, lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  throw std::runtime_error("coo_matmat unsupported value dtype.");
}

} // namespace mlx_sparse
