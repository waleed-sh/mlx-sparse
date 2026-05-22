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

#include "sparse/csr_matmat/csr_matmat.h"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

namespace mlx_sparse {

namespace {

// CSR x CSR has a data-dependent output sparsity pattern. Keep this as an
// eager native assembly routine rather than pretending it is a fixed-shape MLX
// primitive.
template <typename T, typename LhsI, typename RhsI, typename OutI>
std::tuple<mx::array, mx::array, mx::array>
csr_matmat_impl(mx::array lhs_data, mx::array lhs_indices, mx::array lhs_indptr,
                mx::array rhs_data, mx::array rhs_indices, mx::array rhs_indptr,
                int lhs_n_rows, int rhs_n_cols, mx::Dtype out_index_dtype) {
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
  const int rhs_n_rows = static_cast<int>(rhs_indptr.size()) - 1;

  std::vector<T> out_data;
  std::vector<OutI> out_indices;
  std::vector<OutI> out_indptr(static_cast<size_t>(lhs_n_rows) + 1, OutI{0});

  for (int row = 0; row < lhs_n_rows; ++row) {
    std::map<int, AccT> accum;
    for (LhsI lhs_pos = lhs_indptr_ptr[row]; lhs_pos < lhs_indptr_ptr[row + 1];
         ++lhs_pos) {
      const int rhs_row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
      if (rhs_row < 0 || rhs_row >= rhs_n_rows) {
        throw std::invalid_argument(
            "csr_matmat lhs indices contain an out-of-bounds column.");
      }
      const auto lhs_value = lhs_data_ptr[lhs_pos];
      for (RhsI rhs_pos = rhs_indptr_ptr[rhs_row];
           rhs_pos < rhs_indptr_ptr[rhs_row + 1]; ++rhs_pos) {
        const int col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
        if (col < 0 || col >= rhs_n_cols) {
          throw std::invalid_argument(
              "csr_matmat rhs indices contain an out-of-bounds column.");
        }
        accum[col] += multiply_accumulate<T>(lhs_value, rhs_data_ptr[rhs_pos]);
      }
    }

    for (const auto &[col, value] : accum) {
      if (value != AccT{}) {
        out_indices.push_back(static_cast<OutI>(col));
        out_data.push_back(Accumulator<T>::cast(value));
      }
    }
    out_indptr[static_cast<size_t>(row) + 1] =
        static_cast<OutI>(out_data.size());
  }

  return {mx::array(out_data.begin(), mx::Shape{static_cast<int>(out_data.size())},
                    lhs_data.dtype()),
          mx::array(out_indices.begin(),
                    mx::Shape{static_cast<int>(out_indices.size())},
                    out_index_dtype),
          mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    out_index_dtype)};
}

template <typename T, typename LhsI, typename RhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_output_index(mx::array lhs_data, mx::array lhs_indices,
                      mx::array lhs_indptr, mx::array rhs_data,
                      mx::array rhs_indices, mx::array rhs_indptr,
                      int lhs_n_rows, int rhs_n_cols,
                      mx::Dtype out_index_dtype) {
  if (out_index_dtype == mx::int32) {
    return csr_matmat_impl<T, LhsI, RhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return csr_matmat_impl<T, LhsI, RhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      lhs_n_rows, rhs_n_cols, out_index_dtype);
}

template <typename T, typename LhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_rhs_index(mx::array lhs_data, mx::array lhs_indices,
                   mx::array lhs_indptr, mx::array rhs_data,
                   mx::array rhs_indices, mx::array rhs_indptr,
                   int lhs_n_rows, int rhs_n_cols, mx::Dtype out_index_dtype) {
  if (rhs_indices.dtype() == mx::int32) {
    return dispatch_output_index<T, LhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (rhs_indices.dtype() == mx::int64) {
    return dispatch_output_index<T, LhsI, int64_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  throw std::runtime_error("csr_matmat requires int32 or int64 rhs indices.");
}

template <typename T>
std::tuple<mx::array, mx::array, mx::array>
dispatch_lhs_index(mx::array lhs_data, mx::array lhs_indices,
                   mx::array lhs_indptr, mx::array rhs_data,
                   mx::array rhs_indices, mx::array rhs_indptr,
                   int lhs_n_rows, int rhs_n_cols, mx::Dtype out_index_dtype) {
  if (lhs_indices.dtype() == mx::int32) {
    return dispatch_rhs_index<T, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (lhs_indices.dtype() == mx::int64) {
    return dispatch_rhs_index<T, int64_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  throw std::runtime_error("csr_matmat requires int32 or int64 lhs indices.");
}

} // namespace

std::tuple<mx::array, mx::array, mx::array>
csr_matmat(const mx::array &lhs_data, const mx::array &lhs_indices,
           const mx::array &lhs_indptr, const mx::array &rhs_data,
           const mx::array &rhs_indices, const mx::array &rhs_indptr,
           int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols) {
  if (lhs_n_rows < 0 || lhs_n_cols < 0 || rhs_n_rows < 0 || rhs_n_cols < 0) {
    throw std::invalid_argument(
        "csr_matmat shape dimensions must be non-negative.");
  }
  if (lhs_n_cols != rhs_n_rows) {
    throw std::invalid_argument("CSR sparse-sparse matmul dimension mismatch.");
  }
  require_rank(lhs_data, 1, "csr_matmat lhs_data");
  require_rank(lhs_indices, 1, "csr_matmat lhs_indices");
  require_rank(lhs_indptr, 1, "csr_matmat lhs_indptr");
  require_rank(rhs_data, 1, "csr_matmat rhs_data");
  require_rank(rhs_indices, 1, "csr_matmat rhs_indices");
  require_rank(rhs_indptr, 1, "csr_matmat rhs_indptr");
  require_same_value_dtype(lhs_data, rhs_data, "csr_matmat lhs_data",
                           "csr_matmat rhs_data");
  require_same_index_dtype(lhs_indices, lhs_indptr, "csr_matmat lhs_indices",
                           "csr_matmat lhs_indptr");
  require_same_index_dtype(rhs_indices, rhs_indptr, "csr_matmat rhs_indices",
                           "csr_matmat rhs_indptr");
  require_size(lhs_indptr, lhs_n_rows + 1, "csr_matmat lhs_indptr");
  require_size(rhs_indptr, rhs_n_rows + 1, "csr_matmat rhs_indptr");
  if (lhs_indices.size() != lhs_data.size() ||
      rhs_indices.size() != rhs_data.size()) {
    throw std::invalid_argument(
        "csr_matmat data and indices must have equal lengths.");
  }

  const auto out_index_dtype =
      lhs_indices.dtype() == rhs_indices.dtype() ? lhs_indices.dtype() : mx::int64;
  if (lhs_data.dtype() == mx::float32) {
    return dispatch_lhs_index<float>(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                                     rhs_indices, rhs_indptr, lhs_n_rows,
                                     rhs_n_cols, out_index_dtype);
  }
  if (lhs_data.dtype() == mx::float16) {
    return dispatch_lhs_index<mx::float16_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        lhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (lhs_data.dtype() == mx::bfloat16) {
    return dispatch_lhs_index<mx::bfloat16_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        lhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (lhs_data.dtype() == mx::complex64) {
    return dispatch_lhs_index<mx::complex64_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        lhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  throw std::runtime_error("csr_matmat unsupported value dtype.");
}

} // namespace mlx_sparse
