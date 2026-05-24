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

#include "sparse/csc_matmat/csc_matmat.h"

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
      throw std::runtime_error("csc_matmat produced a negative column count.");
    }
    total += static_cast<long long>(counts[i]);
    if (total > static_cast<long long>(std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "csc_matmat output nnz exceeds MLX shape limits.");
    }
    if constexpr (std::is_same_v<I, int32_t>) {
      if (total > static_cast<long long>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error(
            "csc_matmat output nnz exceeds int32 index capacity.");
      }
    }
    indptr[i + 1] = static_cast<I>(total);
  }
  return static_cast<int>(total);
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
std::tuple<mx::array, mx::array, mx::array>
csc_matmat_impl(mx::array lhs_data, mx::array lhs_indices, mx::array lhs_indptr,
                mx::array rhs_data, mx::array rhs_indices, mx::array rhs_indptr,
                int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
                mx::Dtype out_index_dtype) {
  using AccT = typename Accumulator<T>::Type;

  lhs_data.eval();
  lhs_indices.eval();
  lhs_indptr.eval();
  rhs_data.eval();
  rhs_indices.eval();
  rhs_indptr.eval();

  const auto *lhs_data_ptr = lhs_data.data<T>();
  const auto *lhs_indices_ptr = lhs_indices.data<LhsI>();
  const auto *lhs_indptr_ptr = lhs_indptr.data<LhsI>();
  const auto *rhs_data_ptr = rhs_data.data<T>();
  const auto *rhs_indices_ptr = rhs_indices.data<RhsI>();
  const auto *rhs_indptr_ptr = rhs_indptr.data<RhsI>();

  for (int col = 0; col < lhs_n_cols; ++col) {
    for (LhsI lhs_pos = lhs_indptr_ptr[col]; lhs_pos < lhs_indptr_ptr[col + 1];
         ++lhs_pos) {
      const int row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
      if (row < 0 || row >= lhs_n_rows) {
        throw std::invalid_argument(
            "csc_matmat lhs indices contain an out-of-bounds row.");
      }
    }
  }

  std::vector<int> marker(static_cast<size_t>(lhs_n_rows), -1);
  std::vector<OutI> candidate_counts(static_cast<size_t>(rhs_n_cols), OutI{0});
  for (int col = 0; col < rhs_n_cols; ++col) {
    int col_count = 0;
    for (RhsI rhs_pos = rhs_indptr_ptr[col]; rhs_pos < rhs_indptr_ptr[col + 1];
         ++rhs_pos) {
      const int lhs_col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
      if (lhs_col < 0 || lhs_col >= lhs_n_cols) {
        throw std::invalid_argument(
            "csc_matmat rhs indices contain an out-of-bounds row.");
      }
      for (LhsI lhs_pos = lhs_indptr_ptr[lhs_col];
           lhs_pos < lhs_indptr_ptr[lhs_col + 1]; ++lhs_pos) {
        const int row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
        if (row < 0 || row >= lhs_n_rows) {
          throw std::invalid_argument(
              "csc_matmat lhs indices contain an out-of-bounds row.");
        }
        if (marker[static_cast<size_t>(row)] != col) {
          marker[static_cast<size_t>(row)] = col;
          col_count += 1;
        }
      }
    }
    candidate_counts[static_cast<size_t>(col)] = static_cast<OutI>(col_count);
  }

  std::vector<OutI> candidate_indptr;
  const int candidate_nnz = prefix_counts(candidate_counts, candidate_indptr);
  std::vector<T> candidate_data(static_cast<size_t>(candidate_nnz));
  std::vector<OutI> candidate_indices(static_cast<size_t>(candidate_nnz));

  std::fill(marker.begin(), marker.end(), -1);
  std::vector<AccT> accum(static_cast<size_t>(lhs_n_rows),
                          Accumulator<T>::zero());
  std::vector<int> rows;
  for (int col = 0; col < rhs_n_cols; ++col) {
    rows.clear();
    for (RhsI rhs_pos = rhs_indptr_ptr[col]; rhs_pos < rhs_indptr_ptr[col + 1];
         ++rhs_pos) {
      const int lhs_col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
      const T rhs_value = rhs_data_ptr[rhs_pos];
      for (LhsI lhs_pos = lhs_indptr_ptr[lhs_col];
           lhs_pos < lhs_indptr_ptr[lhs_col + 1]; ++lhs_pos) {
        const int row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
        const auto row_index = static_cast<size_t>(row);
        if (marker[row_index] != col) {
          marker[row_index] = col;
          accum[row_index] = Accumulator<T>::zero();
          rows.push_back(row);
        }
        accum[row_index] +=
            multiply_accumulate<T>(lhs_data_ptr[lhs_pos], rhs_value);
      }
    }

    std::sort(rows.begin(), rows.end());
    OutI write = candidate_indptr[static_cast<size_t>(col)];
    for (int row : rows) {
      candidate_indices[static_cast<size_t>(write)] = static_cast<OutI>(row);
      candidate_data[static_cast<size_t>(write)] =
          Accumulator<T>::cast(accum[static_cast<size_t>(row)]);
      ++write;
    }
  }

  std::vector<OutI> out_counts(static_cast<size_t>(rhs_n_cols), OutI{0});
  for (int col = 0; col < rhs_n_cols; ++col) {
    OutI count = OutI{0};
    for (OutI p = candidate_indptr[static_cast<size_t>(col)];
         p < candidate_indptr[static_cast<size_t>(col) + 1]; ++p) {
      if (nonzero(candidate_data[static_cast<size_t>(p)])) {
        count += OutI{1};
      }
    }
    out_counts[static_cast<size_t>(col)] = count;
  }

  std::vector<OutI> out_indptr;
  const int out_nnz = prefix_counts(out_counts, out_indptr);
  std::vector<T> out_data;
  std::vector<OutI> out_indices;
  out_data.reserve(static_cast<size_t>(out_nnz));
  out_indices.reserve(static_cast<size_t>(out_nnz));
  for (int col = 0; col < rhs_n_cols; ++col) {
    for (OutI p = candidate_indptr[static_cast<size_t>(col)];
         p < candidate_indptr[static_cast<size_t>(col) + 1]; ++p) {
      const auto value = candidate_data[static_cast<size_t>(p)];
      if (nonzero(value)) {
        out_data.push_back(value);
        out_indices.push_back(candidate_indices[static_cast<size_t>(p)]);
      }
    }
  }

  return {mx::array(out_data.begin(), mx::Shape{out_nnz}, lhs_data.dtype()),
          mx::array(out_indices.begin(), mx::Shape{out_nnz}, out_index_dtype),
          mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    out_index_dtype)};
}

template <typename T, typename LhsI, typename RhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_out(mx::array lhs_data, mx::array lhs_indices, mx::array lhs_indptr,
             mx::array rhs_data, mx::array rhs_indices, mx::array rhs_indptr,
             int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
             mx::Dtype out_index_dtype) {
  if (out_index_dtype == mx::int32) {
    return csc_matmat_impl<T, LhsI, RhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return csc_matmat_impl<T, LhsI, RhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
}

template <typename T, typename LhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_rhs(mx::array lhs_data, mx::array lhs_indices, mx::array lhs_indptr,
             mx::array rhs_data, mx::array rhs_indices, mx::array rhs_indptr,
             int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
             mx::Dtype out_index_dtype) {
  if (rhs_indices.dtype() == mx::int32) {
    return dispatch_out<T, LhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return dispatch_out<T, LhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
}

template <typename T>
std::tuple<mx::array, mx::array, mx::array>
dispatch_lhs(mx::array lhs_data, mx::array lhs_indices, mx::array lhs_indptr,
             mx::array rhs_data, mx::array rhs_indices, mx::array rhs_indptr,
             int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
             mx::Dtype out_index_dtype) {
  if (lhs_indices.dtype() == mx::int32) {
    return dispatch_rhs<T, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return dispatch_rhs<T, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
}

} // namespace

std::tuple<mx::array, mx::array, mx::array>
csc_matmat(const mx::array &lhs_data, const mx::array &lhs_indices,
           const mx::array &lhs_indptr, const mx::array &rhs_data,
           const mx::array &rhs_indices, const mx::array &rhs_indptr,
           int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols) {
  if (lhs_n_rows < 0 || lhs_n_cols < 0 || rhs_n_rows < 0 || rhs_n_cols < 0) {
    throw std::invalid_argument(
        "csc_matmat shape dimensions must be non-negative.");
  }
  if (lhs_n_cols != rhs_n_rows) {
    throw std::invalid_argument("CSC sparse-sparse matmul dimension mismatch.");
  }
  require_rank(lhs_data, 1, "csc_matmat lhs_data");
  require_rank(lhs_indices, 1, "csc_matmat lhs_indices");
  require_rank(lhs_indptr, 1, "csc_matmat lhs_indptr");
  require_rank(rhs_data, 1, "csc_matmat rhs_data");
  require_rank(rhs_indices, 1, "csc_matmat rhs_indices");
  require_rank(rhs_indptr, 1, "csc_matmat rhs_indptr");
  require_same_value_dtype(lhs_data, rhs_data, "csc_matmat lhs_data",
                           "csc_matmat rhs_data");
  require_same_index_dtype(lhs_indices, lhs_indptr, "csc_matmat lhs_indices",
                           "csc_matmat lhs_indptr");
  require_same_index_dtype(rhs_indices, rhs_indptr, "csc_matmat rhs_indices",
                           "csc_matmat rhs_indptr");
  require_size(lhs_indptr, lhs_n_cols + 1, "csc_matmat lhs_indptr");
  require_size(rhs_indptr, rhs_n_cols + 1, "csc_matmat rhs_indptr");
  if (lhs_data.size() != lhs_indices.size() ||
      rhs_data.size() != rhs_indices.size()) {
    throw std::invalid_argument(
        "csc_matmat data and indices must have equal lengths.");
  }
  if (lhs_n_rows > std::numeric_limits<int>::max()) {
    throw std::overflow_error("csc_matmat n_rows exceeds supported limits.");
  }

  const auto out_index_dtype = lhs_indices.dtype() == rhs_indices.dtype()
                                   ? lhs_indices.dtype()
                                   : mx::int64;
  if (out_index_dtype == mx::int32 &&
      lhs_n_rows > std::numeric_limits<int32_t>::max()) {
    throw std::overflow_error(
        "csc_matmat n_rows exceeds int32 output index capacity.");
  }

  if (lhs_data.dtype() == mx::float32) {
    return dispatch_lhs<float>(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                               rhs_indices, rhs_indptr, lhs_n_rows, lhs_n_cols,
                               rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (lhs_data.dtype() == mx::float16) {
    return dispatch_lhs<mx::float16_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (lhs_data.dtype() == mx::bfloat16) {
    return dispatch_lhs<mx::bfloat16_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (lhs_data.dtype() == mx::complex64) {
    return dispatch_lhs<mx::complex64_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  throw std::runtime_error("csc_matmat unsupported value dtype.");
}

} // namespace mlx_sparse
