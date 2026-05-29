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
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/cpu_parallel.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "sparse/coo_tocsr/coo_tocsr.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

template <typename T> bool nonzero(T value) { return value != T{}; }

std::string coo_matmat_index_kernel_name(const std::string &prefix,
                                         mx::Dtype lhs_index_dtype,
                                         mx::Dtype rhs_index_dtype,
                                         mx::Dtype out_index_dtype) {
  return prefix + "_" + index_kernel_suffix(lhs_index_dtype) + "_" +
         index_kernel_suffix(rhs_index_dtype) + "_" +
         index_kernel_suffix(out_index_dtype);
}

std::string coo_matmat_numeric_kernel_name(mx::Dtype value_dtype,
                                           mx::Dtype lhs_index_dtype,
                                           mx::Dtype rhs_index_dtype,
                                           mx::Dtype out_index_dtype) {
  return "coo_matmat_numeric_" + value_kernel_suffix(value_dtype) + "_" +
         index_kernel_suffix(lhs_index_dtype) + "_" +
         index_kernel_suffix(rhs_index_dtype) + "_" +
         index_kernel_suffix(out_index_dtype);
}

bool use_experimental_metal_spgemm() {
  const char *flag = std::getenv("MLX_SPARSE_EXPERIMENTAL_METAL_SPGEMM");
  return flag != nullptr && std::string(flag) == "1" &&
         mx::default_device().type == mx::Device::gpu;
}

class COOMatmatSymbolic : public mx::Primitive {
public:
  COOMatmatSymbolic(mx::Stream stream, int lhs_n_rows, int rhs_n_rows,
                    int rhs_n_cols)
      : Primitive(stream), lhs_n_rows_(lhs_n_rows), rhs_n_rows_(rhs_n_rows),
        rhs_n_cols_(rhs_n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOMatmatSymbolic"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOMatmatSymbolic &>(other);
    return lhs_n_rows_ == rhs.lhs_n_rows_ && rhs_n_rows_ == rhs.rhs_n_rows_ &&
           rhs_n_cols_ == rhs.rhs_n_cols_;
  }

private:
  int lhs_n_rows_;
  int rhs_n_rows_;
  int rhs_n_cols_;
};

class COOMatmatNumeric : public mx::Primitive {
public:
  COOMatmatNumeric(mx::Stream stream, int lhs_n_rows, int rhs_n_rows,
                   int rhs_n_cols)
      : Primitive(stream), lhs_n_rows_(lhs_n_rows), rhs_n_rows_(rhs_n_rows),
        rhs_n_cols_(rhs_n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOMatmatNumeric"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOMatmatNumeric &>(other);
    return lhs_n_rows_ == rhs.lhs_n_rows_ && rhs_n_rows_ == rhs.rhs_n_rows_ &&
           rhs_n_cols_ == rhs.rhs_n_cols_;
  }

private:
  int lhs_n_rows_;
  int rhs_n_rows_;
  int rhs_n_cols_;
};

class COOMatmatPruneCounts : public mx::Primitive {
public:
  explicit COOMatmatPruneCounts(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOMatmatPruneCounts"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

class COOMatmatPruneFill : public mx::Primitive {
public:
  explicit COOMatmatPruneFill(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOMatmatPruneFill"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

template <typename LhsI, typename RhsI, typename OutI>
void symbolic_cpu_impl(const mx::array &lhs_indices,
                       const mx::array &lhs_indptr,
                       const mx::array &rhs_indices,
                       const mx::array &rhs_indptr, mx::array &counts,
                       int lhs_n_rows, int rhs_n_rows, int rhs_n_cols,
                       mx::Stream stream) {
  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(lhs_indptr);
  encoder.set_input_array(rhs_indices);
  encoder.set_input_array(rhs_indptr);
  encoder.set_output_array(counts);

  encoder.dispatch([lhs_indices = mx::array::unsafe_weak_copy(lhs_indices),
                    lhs_indptr = mx::array::unsafe_weak_copy(lhs_indptr),
                    rhs_indices = mx::array::unsafe_weak_copy(rhs_indices),
                    rhs_indptr = mx::array::unsafe_weak_copy(rhs_indptr),
                    counts = mx::array::unsafe_weak_copy(counts), lhs_n_rows,
                    rhs_n_rows, rhs_n_cols]() mutable {
    const auto *lhs_indices_ptr = lhs_indices.data<LhsI>();
    const auto *lhs_indptr_ptr = lhs_indptr.data<LhsI>();
    const auto *rhs_indices_ptr = rhs_indices.data<RhsI>();
    const auto *rhs_indptr_ptr = rhs_indptr.data<RhsI>();
    auto *counts_ptr = counts.data<OutI>();

    std::vector<int> marker(static_cast<size_t>(rhs_n_cols), -1);
    for (int row = 0; row < lhs_n_rows; ++row) {
      int row_count = 0;
      for (LhsI lhs_pos = lhs_indptr_ptr[row];
           lhs_pos < lhs_indptr_ptr[row + 1]; ++lhs_pos) {
        const int rhs_row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
        if (rhs_row < 0 || rhs_row >= rhs_n_rows) {
          throw std::invalid_argument(
              "coo_matmat lhs columns contain an out-of-bounds entry.");
        }
        for (RhsI rhs_pos = rhs_indptr_ptr[rhs_row];
             rhs_pos < rhs_indptr_ptr[rhs_row + 1]; ++rhs_pos) {
          const int col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
          if (col < 0 || col >= rhs_n_cols) {
            throw std::invalid_argument(
                "coo_matmat rhs columns contain an out-of-bounds entry.");
          }
          if (marker[static_cast<size_t>(col)] != row) {
            marker[static_cast<size_t>(col)] = row;
            row_count += 1;
          }
        }
      }
      counts_ptr[row] = static_cast<OutI>(row_count);
    }
  });
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
void numeric_cpu_impl(const mx::array &lhs_data, const mx::array &lhs_indices,
                      const mx::array &lhs_indptr, const mx::array &rhs_data,
                      const mx::array &rhs_indices, const mx::array &rhs_indptr,
                      const mx::array &out_indptr, mx::array &out_data,
                      mx::array &out_col, int lhs_n_rows, int rhs_n_rows,
                      int rhs_n_cols, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(lhs_data);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(lhs_indptr);
  encoder.set_input_array(rhs_data);
  encoder.set_input_array(rhs_indices);
  encoder.set_input_array(rhs_indptr);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_col);

  encoder.dispatch([lhs_data = mx::array::unsafe_weak_copy(lhs_data),
                    lhs_indices = mx::array::unsafe_weak_copy(lhs_indices),
                    lhs_indptr = mx::array::unsafe_weak_copy(lhs_indptr),
                    rhs_data = mx::array::unsafe_weak_copy(rhs_data),
                    rhs_indices = mx::array::unsafe_weak_copy(rhs_indices),
                    rhs_indptr = mx::array::unsafe_weak_copy(rhs_indptr),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_col = mx::array::unsafe_weak_copy(out_col), lhs_n_rows,
                    rhs_n_rows, rhs_n_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;

    const auto *lhs_data_ptr = lhs_data.data<T>();
    const auto *lhs_indices_ptr = lhs_indices.data<LhsI>();
    const auto *lhs_indptr_ptr = lhs_indptr.data<LhsI>();
    const auto *rhs_data_ptr = rhs_data.data<T>();
    const auto *rhs_indices_ptr = rhs_indices.data<RhsI>();
    const auto *rhs_indptr_ptr = rhs_indptr.data<RhsI>();
    const auto *out_indptr_ptr = out_indptr.data<OutI>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_col_ptr = out_col.data<OutI>();

    std::vector<int> marker(static_cast<size_t>(rhs_n_cols), -1);
    std::vector<AccT> accum(static_cast<size_t>(rhs_n_cols),
                            Accumulator<T>::zero());
    std::vector<int> columns;

    for (int row = 0; row < lhs_n_rows; ++row) {
      columns.clear();
      for (LhsI lhs_pos = lhs_indptr_ptr[row];
           lhs_pos < lhs_indptr_ptr[row + 1]; ++lhs_pos) {
        const int rhs_row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
        if (rhs_row < 0 || rhs_row >= rhs_n_rows) {
          throw std::invalid_argument(
              "coo_matmat lhs columns contain an out-of-bounds entry.");
        }
        const auto lhs_value = lhs_data_ptr[lhs_pos];
        for (RhsI rhs_pos = rhs_indptr_ptr[rhs_row];
             rhs_pos < rhs_indptr_ptr[rhs_row + 1]; ++rhs_pos) {
          const int col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
          if (col < 0 || col >= rhs_n_cols) {
            throw std::invalid_argument(
                "coo_matmat rhs columns contain an out-of-bounds entry.");
          }
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
      OutI write = out_indptr_ptr[row];
      for (int col : columns) {
        const auto col_index = static_cast<size_t>(col);
        out_col_ptr[write] = static_cast<OutI>(col);
        out_data_ptr[write] = Accumulator<T>::cast(accum[col_index]);
        ++write;
      }
    }
  });
}

template <typename T, typename I>
void prune_counts_cpu_impl(const mx::array &data, const mx::array &indptr,
                           mx::array &counts, mx::Stream stream) {
  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indptr);
  encoder.set_output_array(counts);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    counts = mx::array::unsafe_weak_copy(counts)]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *counts_ptr = counts.data<I>();
    const int n_rows = static_cast<int>(indptr.size()) - 1;

    for (int row = 0; row < n_rows; ++row) {
      I count = I{0};
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        if (nonzero(data_ptr[p])) {
          count += I{1};
        }
      }
      counts_ptr[row] = count;
    }
  });
}

template <typename T, typename I>
void prune_fill_cpu_impl(const mx::array &data, const mx::array &col,
                         const mx::array &indptr, const mx::array &out_indptr,
                         mx::array &out_data, mx::array &out_row,
                         mx::array &out_col, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_row.set_data(mx::allocator::malloc(out_row.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(col);
  encoder.set_input_array(indptr);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_row);
  encoder.set_output_array(out_col);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    col = mx::array::unsafe_weak_copy(col),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_row = mx::array::unsafe_weak_copy(out_row),
                    out_col = mx::array::unsafe_weak_copy(out_col)]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *col_ptr = col.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *out_indptr_ptr = out_indptr.data<I>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_row_ptr = out_row.data<I>();
    auto *out_col_ptr = out_col.data<I>();
    const int n_rows = static_cast<int>(indptr.size()) - 1;

    for (int row = 0; row < n_rows; ++row) {
      I write = out_indptr_ptr[row];
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const T value = data_ptr[p];
        if (nonzero(value)) {
          out_data_ptr[write] = value;
          out_row_ptr[write] = static_cast<I>(row);
          out_col_ptr[write] = col_ptr[p];
          ++write;
        }
      }
    }
  });
}

template <typename I>
std::pair<mx::array, int> build_indptr_from_counts(const mx::array &counts,
                                                   int n_rows,
                                                   mx::Dtype index_dtype) {
  const auto *counts_ptr = counts.data<I>();
  std::vector<I> out_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  int64_t total = 0;
  for (int row = 0; row < n_rows; ++row) {
    const auto count = static_cast<int64_t>(counts_ptr[row]);
    if (count < 0) {
      throw std::runtime_error("coo_matmat produced a negative row count.");
    }
    total += count;
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error(
          "coo_matmat output nnz exceeds MLX shape limits.");
    }
    if (total > static_cast<int64_t>(std::numeric_limits<I>::max())) {
      throw std::overflow_error(
          "coo_matmat output nnz exceeds index dtype capacity.");
    }
    out_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(total);
  }

  return {mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    index_dtype),
          static_cast<int>(total)};
}

std::pair<mx::array, int> build_indptr_from_counts(const mx::array &counts,
                                                   int n_rows,
                                                   mx::Dtype index_dtype) {
  if (index_dtype == mx::int32) {
    return build_indptr_from_counts<int32_t>(counts, n_rows, index_dtype);
  }
  return build_indptr_from_counts<int64_t>(counts, n_rows, index_dtype);
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

void sort_touched_indices(std::vector<int> &indices) {
  constexpr size_t kInsertionSortLimit = 32;
  if (indices.size() <= 1) {
    return;
  }
  if (indices.size() <= kInsertionSortLimit) {
    for (size_t i = 1; i < indices.size(); ++i) {
      const int value = indices[i];
      size_t j = i;
      while (j > 0 && value < indices[j - 1]) {
        indices[j] = indices[j - 1];
        --j;
      }
      indices[j] = value;
    }
    return;
  }
  std::sort(indices.begin(), indices.end());
}

bool use_dense_ordered_scan(size_t touched_count, int dimension,
                            int disorder_count) {
  constexpr size_t kMinDenseScanTouched = 64;
  constexpr size_t kDenseScanFactor = 32;
  constexpr int kMinDenseScanDisorder = 8;
  return touched_count >= kMinDenseScanTouched && dimension > 0 &&
         disorder_count >= kMinDenseScanDisorder &&
         static_cast<size_t>(dimension) <= touched_count * kDenseScanFactor;
}

size_t spgemm_reserve_hint(int outer_dim, int inner_dim, int result_dim,
                           size_t lhs_nnz, size_t rhs_nnz) {
  if (outer_dim <= 0 || inner_dim <= 0 || result_dim <= 0 || lhs_nnz == 0 ||
      rhs_nnz == 0) {
    return 0;
  }

  constexpr long double kPathologicalWorkFactor = 32.0L;
  constexpr long double kMaxReserveHint = 64.0L * 1024.0L * 1024.0L;

  const long double linear_input =
      static_cast<long double>(lhs_nnz) + static_cast<long double>(rhs_nnz);
  const long double average_rhs_row_nnz =
      static_cast<long double>(rhs_nnz) / static_cast<long double>(inner_dim);
  const long double estimated_products =
      static_cast<long double>(lhs_nnz) * average_rhs_row_nnz;
  const long double dense_bound = static_cast<long double>(outer_dim) *
                                  static_cast<long double>(result_dim);

  long double estimate = std::min(estimated_products, dense_bound);
  if (estimate > kPathologicalWorkFactor * linear_input) {
    estimate = linear_input;
  }
  estimate = std::min(estimate, kMaxReserveHint);
  if (estimate <= 0.0L) {
    return 0;
  }
  return static_cast<size_t>(estimate);
}

template <typename OutI>
void check_output_nnz(size_t nnz, const char *op_name) {
  if (nnz > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(op_name) +
                              " output nnz exceeds MLX shape limits.");
  }
  if (nnz > static_cast<size_t>(std::numeric_limits<OutI>::max())) {
    throw std::overflow_error(std::string(op_name) +
                              " output nnz exceeds index dtype capacity.");
  }
}

int64_t saturated_add(int64_t lhs, int64_t rhs) {
  if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return std::numeric_limits<int64_t>::max();
  }
  if (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs) {
    return std::numeric_limits<int64_t>::min();
  }
  return lhs + rhs;
}

size_t local_reserve_hint(size_t global_hint, int64_t local_work,
                          int64_t total_work) {
  if (global_hint == 0 || local_work <= 0 || total_work <= 0) {
    return 0;
  }
  const long double fraction = static_cast<long double>(local_work) /
                               static_cast<long double>(total_work);
  const auto estimate =
      static_cast<size_t>(std::max<long double>(1.0L, global_hint * fraction));
  return std::min(global_hint, estimate);
}

template <typename T, typename OutI> struct LocalCooSpgemmOutput {
  std::vector<T> data;
  std::vector<OutI> col;
};

template <typename LhsI>
std::vector<int64_t> coo_row_work(const std::vector<int> &lhs_offsets,
                                  const std::vector<int> &lhs_positions,
                                  const LhsI *lhs_col_ptr,
                                  const std::vector<int> &rhs_offsets,
                                  int lhs_n_rows, int rhs_n_rows) {
  std::vector<int64_t> work(static_cast<size_t>(lhs_n_rows), 0);
  for (int row = 0; row < lhs_n_rows; ++row) {
    int64_t row_work = 0;
    for (int lp = lhs_offsets[static_cast<size_t>(row)];
         lp < lhs_offsets[static_cast<size_t>(row) + 1]; ++lp) {
      const int lhs_pos = lhs_positions[static_cast<size_t>(lp)];
      const int rhs_row = static_cast<int>(lhs_col_ptr[lhs_pos]);
      if (rhs_row < 0 || rhs_row >= rhs_n_rows) {
        throw std::invalid_argument(
            "coo_matmat lhs coordinates contain an out-of-bounds column.");
      }
      row_work = saturated_add(
          row_work,
          static_cast<int64_t>(rhs_offsets[static_cast<size_t>(rhs_row) + 1] -
                               rhs_offsets[static_cast<size_t>(rhs_row)]));
    }
    work[static_cast<size_t>(row)] = row_work;
  }
  return work;
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
  std::vector<AccT> accum(static_cast<size_t>(rhs_n_cols),
                          Accumulator<T>::zero());
  std::vector<int> columns;
  std::vector<T> out_data;
  std::vector<OutI> out_row;
  std::vector<OutI> out_col;
  const size_t reserve_hint = spgemm_reserve_hint(
      lhs_n_rows, rhs_n_rows, rhs_n_cols, lhs_data.size(), rhs_data.size());
  out_data.reserve(reserve_hint);
  out_row.reserve(reserve_hint);
  out_col.reserve(reserve_hint);

  for (int row = 0; row < lhs_n_rows; ++row) {
    columns.clear();
    bool columns_sorted = true;
    int disorder_count = 0;
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
        const AccT product =
            multiply_accumulate<T>(lhs_value, rhs_data_ptr[rhs_pos]);
        if (marker[col_index] != row) {
          marker[col_index] = row;
          accum[col_index] = product;
          if (!columns.empty() && col < columns.back()) {
            columns_sorted = false;
            disorder_count += 1;
          }
          columns.push_back(col);
        } else {
          accum[col_index] += product;
        }
      }
    }

    if (!columns_sorted &&
        use_dense_ordered_scan(columns.size(), rhs_n_cols, disorder_count)) {
      for (int col = 0; col < rhs_n_cols; ++col) {
        const auto col_index = static_cast<size_t>(col);
        if (marker[col_index] != row) {
          continue;
        }
        const auto value = Accumulator<T>::cast(accum[col_index]);
        if (nonzero(value)) {
          out_data.push_back(value);
          out_row.push_back(static_cast<OutI>(row));
          out_col.push_back(static_cast<OutI>(col));
        }
      }
    } else {
      if (!columns_sorted) {
        sort_touched_indices(columns);
      }
      for (int col : columns) {
        const auto value =
            Accumulator<T>::cast(accum[static_cast<size_t>(col)]);
        if (nonzero(value)) {
          out_data.push_back(value);
          out_row.push_back(static_cast<OutI>(row));
          out_col.push_back(static_cast<OutI>(col));
        }
      }
    }
    check_output_nnz<OutI>(out_data.size(), "coo_matmat");
  }

  const int out_nnz = static_cast<int>(out_data.size());
  return {mx::array(out_data.begin(), mx::Shape{out_nnz}, lhs_data.dtype()),
          mx::array(out_row.begin(), mx::Shape{out_nnz}, out_index_dtype),
          mx::array(out_col.begin(), mx::Shape{out_nnz}, out_index_dtype)};
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
std::tuple<mx::array, mx::array, mx::array>
coo_matmat_parallel_impl(mx::array lhs_data, mx::array lhs_row,
                         mx::array lhs_col, mx::array rhs_data,
                         mx::array rhs_row, mx::array rhs_col, int lhs_n_rows,
                         int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
                         mx::Dtype out_index_dtype, int requested_workers) {
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

  const auto row_work = coo_row_work(lhs_offsets, lhs_positions, lhs_col_ptr,
                                     rhs_offsets, lhs_n_rows, rhs_n_rows);
  const auto ranges = cpu_ranges_for_output_work(row_work, requested_workers);
  std::vector<OutI> row_counts(static_cast<size_t>(lhs_n_rows), OutI{0});
  std::vector<LocalCooSpgemmOutput<T, OutI>> local_outputs(ranges.size());
  const size_t reserve_hint = spgemm_reserve_hint(
      lhs_n_rows, rhs_n_rows, rhs_n_cols, lhs_data.size(), rhs_data.size());
  int64_t total_work = 0;
  for (const auto work : row_work) {
    total_work = saturated_add(total_work, work);
  }

  parallel_for_cpu_ranges_indexed(ranges, [&](size_t worker, CpuRange range) {
    auto &local = local_outputs[worker];
    int64_t local_work = 0;
    for (int row = range.begin; row < range.end; ++row) {
      local_work =
          saturated_add(local_work, row_work[static_cast<size_t>(row)]);
    }
    const auto reserve =
        local_reserve_hint(reserve_hint, local_work, total_work);
    local.data.reserve(reserve);
    local.col.reserve(reserve);

    std::vector<int> marker(static_cast<size_t>(rhs_n_cols), -1);
    std::vector<AccT> accum(static_cast<size_t>(rhs_n_cols),
                            Accumulator<T>::zero());
    std::vector<int> columns;

    for (int row = range.begin; row < range.end; ++row) {
      columns.clear();
      bool columns_sorted = true;
      int disorder_count = 0;
      const auto before = local.data.size();
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
          const AccT product =
              multiply_accumulate<T>(lhs_value, rhs_data_ptr[rhs_pos]);
          if (marker[col_index] != row) {
            marker[col_index] = row;
            accum[col_index] = product;
            if (!columns.empty() && col < columns.back()) {
              columns_sorted = false;
              disorder_count += 1;
            }
            columns.push_back(col);
          } else {
            accum[col_index] += product;
          }
        }
      }

      if (!columns_sorted &&
          use_dense_ordered_scan(columns.size(), rhs_n_cols, disorder_count)) {
        for (int col = 0; col < rhs_n_cols; ++col) {
          const auto col_index = static_cast<size_t>(col);
          if (marker[col_index] != row) {
            continue;
          }
          const auto value = Accumulator<T>::cast(accum[col_index]);
          if (nonzero(value)) {
            local.data.push_back(value);
            local.col.push_back(static_cast<OutI>(col));
          }
        }
      } else {
        if (!columns_sorted) {
          sort_touched_indices(columns);
        }
        for (int col : columns) {
          const auto value =
              Accumulator<T>::cast(accum[static_cast<size_t>(col)]);
          if (nonzero(value)) {
            local.data.push_back(value);
            local.col.push_back(static_cast<OutI>(col));
          }
        }
      }

      const auto row_nnz = local.data.size() - before;
      check_output_nnz<OutI>(row_nnz, "coo_matmat row");
      row_counts[static_cast<size_t>(row)] = static_cast<OutI>(row_nnz);
    }
  });

  size_t total_nnz = 0;
  std::vector<size_t> row_offsets(static_cast<size_t>(lhs_n_rows) + 1, 0);
  for (int row = 0; row < lhs_n_rows; ++row) {
    total_nnz += static_cast<size_t>(row_counts[static_cast<size_t>(row)]);
    check_output_nnz<OutI>(total_nnz, "coo_matmat");
    row_offsets[static_cast<size_t>(row) + 1] = total_nnz;
  }

  std::vector<T> out_data(total_nnz);
  std::vector<OutI> out_row(total_nnz);
  std::vector<OutI> out_col(total_nnz);
  for (size_t worker = 0; worker < ranges.size(); ++worker) {
    const auto &range = ranges[worker];
    const auto &local = local_outputs[worker];
    size_t read = 0;
    for (int row = range.begin; row < range.end; ++row) {
      const auto count =
          static_cast<size_t>(row_counts[static_cast<size_t>(row)]);
      const auto write = row_offsets[static_cast<size_t>(row)];
      std::copy(local.data.begin() + static_cast<std::ptrdiff_t>(read),
                local.data.begin() + static_cast<std::ptrdiff_t>(read + count),
                out_data.begin() + static_cast<std::ptrdiff_t>(write));
      std::fill(out_row.begin() + static_cast<std::ptrdiff_t>(write),
                out_row.begin() + static_cast<std::ptrdiff_t>(write + count),
                static_cast<OutI>(row));
      std::copy(local.col.begin() + static_cast<std::ptrdiff_t>(read),
                local.col.begin() + static_cast<std::ptrdiff_t>(read + count),
                out_col.begin() + static_cast<std::ptrdiff_t>(write));
      read += count;
    }
    if (read != local.data.size() || read != local.col.size()) {
      throw std::runtime_error("coo_matmat internal parallel count mismatch.");
    }
  }

  const int out_nnz = static_cast<int>(total_nnz);
  return {mx::array(out_data.begin(), mx::Shape{out_nnz}, lhs_data.dtype()),
          mx::array(out_row.begin(), mx::Shape{out_nnz}, out_index_dtype),
          mx::array(out_col.begin(), mx::Shape{out_nnz}, out_index_dtype)};
}

template <typename T, typename LhsI, typename RhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_out(mx::array lhs_data, mx::array lhs_row, mx::array lhs_col,
             mx::array rhs_data, mx::array rhs_row, mx::array rhs_col,
             int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
             mx::Dtype out_index_dtype, int requested_workers) {
  if (out_index_dtype == mx::int32) {
    if (requested_workers > 1) {
      return coo_matmat_parallel_impl<T, LhsI, RhsI, int32_t>(
          std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
          std::move(rhs_data), std::move(rhs_row), std::move(rhs_col),
          lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
          requested_workers);
    }
    return coo_matmat_impl<T, LhsI, RhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
        std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (requested_workers > 1) {
    return coo_matmat_parallel_impl<T, LhsI, RhsI, int64_t>(
        std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
        std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype, requested_workers);
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
             mx::Dtype out_index_dtype, int requested_workers) {
  if (rhs_row.dtype() == mx::int32) {
    return dispatch_out<T, LhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
        std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype, requested_workers);
  }
  return dispatch_out<T, LhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
      std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
      lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype, requested_workers);
}

template <typename T>
std::tuple<mx::array, mx::array, mx::array>
dispatch_lhs(mx::array lhs_data, mx::array lhs_row, mx::array lhs_col,
             mx::array rhs_data, mx::array rhs_row, mx::array rhs_col,
             int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
             mx::Dtype out_index_dtype, int requested_workers) {
  if (lhs_row.dtype() == mx::int32) {
    return dispatch_rhs<T, int32_t>(
        std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
        std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype, requested_workers);
  }
  return dispatch_rhs<T, int64_t>(
      std::move(lhs_data), std::move(lhs_row), std::move(lhs_col),
      std::move(rhs_data), std::move(rhs_row), std::move(rhs_col), lhs_n_rows,
      lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype, requested_workers);
}

mx::array symbolic_counts(const mx::array &lhs_indices,
                          const mx::array &lhs_indptr,
                          const mx::array &rhs_indices,
                          const mx::array &rhs_indptr, int lhs_n_rows,
                          int rhs_n_rows, int rhs_n_cols,
                          mx::Dtype out_index_dtype, mx::Stream stream) {
  auto primitive = std::make_shared<COOMatmatSymbolic>(stream, lhs_n_rows,
                                                       rhs_n_rows, rhs_n_cols);
  return mx::array(mx::Shape{lhs_n_rows}, out_index_dtype, primitive,
                   {lhs_indices, lhs_indptr, rhs_indices, rhs_indptr});
}

std::tuple<mx::array, mx::array>
numeric_fill(const mx::array &lhs_data, const mx::array &lhs_indices,
             const mx::array &lhs_indptr, const mx::array &rhs_data,
             const mx::array &rhs_indices, const mx::array &rhs_indptr,
             const mx::array &out_indptr, int out_nnz, int lhs_n_rows,
             int rhs_n_rows, int rhs_n_cols, mx::Dtype out_index_dtype,
             mx::Stream stream) {
  auto primitive = std::make_shared<COOMatmatNumeric>(stream, lhs_n_rows,
                                                      rhs_n_rows, rhs_n_cols);
  auto outputs =
      mx::array::make_arrays({mx::Shape{out_nnz}, mx::Shape{out_nnz}},
                             {lhs_data.dtype(), out_index_dtype}, primitive,
                             {lhs_data, lhs_indices, lhs_indptr, rhs_data,
                              rhs_indices, rhs_indptr, out_indptr});
  return {outputs[0], outputs[1]};
}

mx::array prune_counts(const mx::array &data, const mx::array &indptr,
                       int n_rows, mx::Stream stream) {
  auto primitive = std::make_shared<COOMatmatPruneCounts>(stream);
  return mx::array(mx::Shape{n_rows}, indptr.dtype(), primitive,
                   {data, indptr});
}

std::tuple<mx::array, mx::array, mx::array>
prune_fill(const mx::array &data, const mx::array &col, const mx::array &indptr,
           const mx::array &out_indptr, int out_nnz, mx::Stream stream) {
  auto primitive = std::make_shared<COOMatmatPruneFill>(stream);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}, mx::Shape{out_nnz}},
      {data.dtype(), col.dtype(), col.dtype()}, primitive,
      {data, col, indptr, out_indptr});
  return {outputs[0], outputs[1], outputs[2]};
}

template <typename LhsI, typename RhsI>
void dispatch_symbolic_out(const mx::array &lhs_indices,
                           const mx::array &lhs_indptr,
                           const mx::array &rhs_indices,
                           const mx::array &rhs_indptr, mx::array &counts,
                           int lhs_n_rows, int rhs_n_rows, int rhs_n_cols,
                           mx::Stream stream) {
  if (counts.dtype() == mx::int32) {
    symbolic_cpu_impl<LhsI, RhsI, int32_t>(lhs_indices, lhs_indptr, rhs_indices,
                                           rhs_indptr, counts, lhs_n_rows,
                                           rhs_n_rows, rhs_n_cols, stream);
    return;
  }
  symbolic_cpu_impl<LhsI, RhsI, int64_t>(lhs_indices, lhs_indptr, rhs_indices,
                                         rhs_indptr, counts, lhs_n_rows,
                                         rhs_n_rows, rhs_n_cols, stream);
}

template <typename LhsI>
void dispatch_symbolic_rhs(const mx::array &lhs_indices,
                           const mx::array &lhs_indptr,
                           const mx::array &rhs_indices,
                           const mx::array &rhs_indptr, mx::array &counts,
                           int lhs_n_rows, int rhs_n_rows, int rhs_n_cols,
                           mx::Stream stream) {
  if (rhs_indices.dtype() == mx::int32) {
    dispatch_symbolic_out<LhsI, int32_t>(lhs_indices, lhs_indptr, rhs_indices,
                                         rhs_indptr, counts, lhs_n_rows,
                                         rhs_n_rows, rhs_n_cols, stream);
    return;
  }
  dispatch_symbolic_out<LhsI, int64_t>(lhs_indices, lhs_indptr, rhs_indices,
                                       rhs_indptr, counts, lhs_n_rows,
                                       rhs_n_rows, rhs_n_cols, stream);
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
void numeric_cpu_dispatch_out(
    const mx::array &lhs_data, const mx::array &lhs_indices,
    const mx::array &lhs_indptr, const mx::array &rhs_data,
    const mx::array &rhs_indices, const mx::array &rhs_indptr,
    const mx::array &out_indptr, mx::array &out_data, mx::array &out_col,
    int lhs_n_rows, int rhs_n_rows, int rhs_n_cols, mx::Stream stream) {
  numeric_cpu_impl<T, LhsI, RhsI, OutI>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_col, lhs_n_rows, rhs_n_rows, rhs_n_cols,
      stream);
}

template <typename T, typename LhsI, typename RhsI>
void numeric_cpu_dispatch_index(
    const mx::array &lhs_data, const mx::array &lhs_indices,
    const mx::array &lhs_indptr, const mx::array &rhs_data,
    const mx::array &rhs_indices, const mx::array &rhs_indptr,
    const mx::array &out_indptr, mx::array &out_data, mx::array &out_col,
    int lhs_n_rows, int rhs_n_rows, int rhs_n_cols, mx::Stream stream) {
  if (out_col.dtype() == mx::int32) {
    numeric_cpu_dispatch_out<T, LhsI, RhsI, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        out_indptr, out_data, out_col, lhs_n_rows, rhs_n_rows, rhs_n_cols,
        stream);
    return;
  }
  numeric_cpu_dispatch_out<T, LhsI, RhsI, int64_t>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_col, lhs_n_rows, rhs_n_rows, rhs_n_cols,
      stream);
}

template <typename T, typename LhsI>
void numeric_cpu_dispatch_rhs(
    const mx::array &lhs_data, const mx::array &lhs_indices,
    const mx::array &lhs_indptr, const mx::array &rhs_data,
    const mx::array &rhs_indices, const mx::array &rhs_indptr,
    const mx::array &out_indptr, mx::array &out_data, mx::array &out_col,
    int lhs_n_rows, int rhs_n_rows, int rhs_n_cols, mx::Stream stream) {
  if (rhs_indices.dtype() == mx::int32) {
    numeric_cpu_dispatch_index<T, LhsI, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        out_indptr, out_data, out_col, lhs_n_rows, rhs_n_rows, rhs_n_cols,
        stream);
    return;
  }
  numeric_cpu_dispatch_index<T, LhsI, int64_t>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_col, lhs_n_rows, rhs_n_rows, rhs_n_cols,
      stream);
}

template <typename T>
void numeric_cpu_dispatch_lhs(
    const mx::array &lhs_data, const mx::array &lhs_indices,
    const mx::array &lhs_indptr, const mx::array &rhs_data,
    const mx::array &rhs_indices, const mx::array &rhs_indptr,
    const mx::array &out_indptr, mx::array &out_data, mx::array &out_col,
    int lhs_n_rows, int rhs_n_rows, int rhs_n_cols, mx::Stream stream) {
  if (lhs_indices.dtype() == mx::int32) {
    numeric_cpu_dispatch_rhs<T, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        out_indptr, out_data, out_col, lhs_n_rows, rhs_n_rows, rhs_n_cols,
        stream);
    return;
  }
  numeric_cpu_dispatch_rhs<T, int64_t>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_col, lhs_n_rows, rhs_n_rows, rhs_n_cols,
      stream);
}

template <typename T>
void dispatch_prune_counts(const mx::array &data, const mx::array &indptr,
                           mx::array &counts, mx::Stream stream) {
  if (indptr.dtype() == mx::int32) {
    prune_counts_cpu_impl<T, int32_t>(data, indptr, counts, stream);
    return;
  }
  prune_counts_cpu_impl<T, int64_t>(data, indptr, counts, stream);
}

template <typename T>
void dispatch_prune_fill(const mx::array &data, const mx::array &col,
                         const mx::array &indptr, const mx::array &out_indptr,
                         mx::array &out_data, mx::array &out_row,
                         mx::array &out_col, mx::Stream stream) {
  if (col.dtype() == mx::int32) {
    prune_fill_cpu_impl<T, int32_t>(data, col, indptr, out_indptr, out_data,
                                    out_row, out_col, stream);
    return;
  }
  prune_fill_cpu_impl<T, int64_t>(data, col, indptr, out_indptr, out_data,
                                  out_row, out_col, stream);
}

void COOMatmatSymbolic::eval_cpu(const std::vector<mx::array> &inputs,
                                 std::vector<mx::array> &outputs) {
  const auto &lhs_indices = inputs[0];
  const auto &lhs_indptr = inputs[1];
  const auto &rhs_indices = inputs[2];
  const auto &rhs_indptr = inputs[3];
  auto &counts = outputs[0];

  if (lhs_indices.dtype() == mx::int32) {
    dispatch_symbolic_rhs<int32_t>(lhs_indices, lhs_indptr, rhs_indices,
                                   rhs_indptr, counts, lhs_n_rows_, rhs_n_rows_,
                                   rhs_n_cols_, stream());
    return;
  }
  dispatch_symbolic_rhs<int64_t>(lhs_indices, lhs_indptr, rhs_indices,
                                 rhs_indptr, counts, lhs_n_rows_, rhs_n_rows_,
                                 rhs_n_cols_, stream());
}

#ifdef _METAL_
void COOMatmatSymbolic::eval_gpu(const std::vector<mx::array> &inputs,
                                 std::vector<mx::array> &outputs) {
  const auto &lhs_indices = inputs[0];
  const auto &lhs_indptr = inputs[1];
  const auto &rhs_indices = inputs[2];
  const auto &rhs_indptr = inputs[3];
  auto &counts = outputs[0];

  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      coo_matmat_index_kernel_name("coo_matmat_symbolic", lhs_indices.dtype(),
                                   rhs_indices.dtype(), counts.dtype()),
      lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(lhs_indices, 0);
  encoder.set_input_array(lhs_indptr, 1);
  encoder.set_input_array(rhs_indices, 2);
  encoder.set_input_array(rhs_indptr, 3);
  encoder.set_output_array(counts, 4);
  encoder.set_bytes(lhs_n_rows_, 5);
  encoder.set_bytes(rhs_n_rows_, 6);
  encoder.set_bytes(rhs_n_cols_, 7);

  auto threads = std::max<size_t>(static_cast<size_t>(lhs_n_rows_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOMatmatSymbolic::eval_gpu(const std::vector<mx::array> &,
                                 std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_matmat has no GPU implementation in this build.");
}
#endif

void COOMatmatNumeric::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &lhs_indices = inputs[1];
  const auto &lhs_indptr = inputs[2];
  const auto &rhs_data = inputs[3];
  const auto &rhs_indices = inputs[4];
  const auto &rhs_indptr = inputs[5];
  const auto &out_indptr = inputs[6];

#define DISPATCH_COO_MATMAT_NUMERIC_VALUE(DTYPE, TYPE)                         \
  if (lhs_data.dtype() == DTYPE) {                                             \
    numeric_cpu_dispatch_lhs<TYPE>(                                            \
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,  \
        out_indptr, outputs[0], outputs[1], lhs_n_rows_, rhs_n_rows_,          \
        rhs_n_cols_, stream());                                                \
    return;                                                                    \
  }

  DISPATCH_COO_MATMAT_NUMERIC_VALUE(mx::float32, float)
  DISPATCH_COO_MATMAT_NUMERIC_VALUE(mx::float16, mx::float16_t)
  DISPATCH_COO_MATMAT_NUMERIC_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_MATMAT_NUMERIC_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_MATMAT_NUMERIC_VALUE

  throw std::runtime_error("coo_matmat unsupported value dtype.");
}

#ifdef _METAL_
void COOMatmatNumeric::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &lhs_indices = inputs[1];
  const auto &lhs_indptr = inputs[2];
  const auto &rhs_data = inputs[3];
  const auto &rhs_indices = inputs[4];
  const auto &rhs_indptr = inputs[5];
  const auto &out_indptr = inputs[6];
  auto &out_data = outputs[0];
  auto &out_col = outputs[1];

  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      coo_matmat_numeric_kernel_name(lhs_data.dtype(), lhs_indices.dtype(),
                                     rhs_indices.dtype(), out_col.dtype()),
      lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(lhs_data, 0);
  encoder.set_input_array(lhs_indices, 1);
  encoder.set_input_array(lhs_indptr, 2);
  encoder.set_input_array(rhs_data, 3);
  encoder.set_input_array(rhs_indices, 4);
  encoder.set_input_array(rhs_indptr, 5);
  encoder.set_input_array(out_indptr, 6);
  encoder.set_output_array(out_data, 7);
  encoder.set_output_array(out_col, 8);
  encoder.set_bytes(lhs_n_rows_, 9);
  encoder.set_bytes(rhs_n_rows_, 10);
  encoder.set_bytes(rhs_n_cols_, 11);

  auto threads = std::max<size_t>(static_cast<size_t>(lhs_n_rows_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOMatmatNumeric::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_matmat has no GPU implementation in this build.");
}
#endif

void COOMatmatPruneCounts::eval_cpu(const std::vector<mx::array> &inputs,
                                    std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indptr = inputs[1];

#define DISPATCH_COO_MATMAT_PRUNE_COUNTS(DTYPE, TYPE)                          \
  if (data.dtype() == DTYPE) {                                                 \
    dispatch_prune_counts<TYPE>(data, indptr, outputs[0], stream());           \
    return;                                                                    \
  }

  DISPATCH_COO_MATMAT_PRUNE_COUNTS(mx::float32, float)
  DISPATCH_COO_MATMAT_PRUNE_COUNTS(mx::float16, mx::float16_t)
  DISPATCH_COO_MATMAT_PRUNE_COUNTS(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_MATMAT_PRUNE_COUNTS(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_MATMAT_PRUNE_COUNTS

  throw std::runtime_error("coo_matmat unsupported value dtype.");
}

#ifdef _METAL_
void COOMatmatPruneCounts::eval_gpu(const std::vector<mx::array> &inputs,
                                    std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indptr = inputs[1];
  auto &counts = outputs[0];

  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel =
      device.get_kernel(sparse_kernel_name("coo_matmat_prune_counts",
                                           data.dtype(), indptr.dtype()),
                        lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indptr, 1);
  encoder.set_output_array(counts, 2);
  const int n_rows = static_cast<int>(indptr.size()) - 1;
  encoder.set_bytes(n_rows, 3);

  auto threads = std::max<size_t>(static_cast<size_t>(n_rows), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOMatmatPruneCounts::eval_gpu(const std::vector<mx::array> &,
                                    std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_matmat has no GPU implementation in this build.");
}
#endif

void COOMatmatPruneFill::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &col = inputs[1];
  const auto &indptr = inputs[2];
  const auto &out_indptr = inputs[3];

#define DISPATCH_COO_MATMAT_PRUNE_FILL(DTYPE, TYPE)                            \
  if (data.dtype() == DTYPE) {                                                 \
    dispatch_prune_fill<TYPE>(data, col, indptr, out_indptr, outputs[0],       \
                              outputs[1], outputs[2], stream());               \
    return;                                                                    \
  }

  DISPATCH_COO_MATMAT_PRUNE_FILL(mx::float32, float)
  DISPATCH_COO_MATMAT_PRUNE_FILL(mx::float16, mx::float16_t)
  DISPATCH_COO_MATMAT_PRUNE_FILL(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_MATMAT_PRUNE_FILL(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_MATMAT_PRUNE_FILL

  throw std::runtime_error("coo_matmat unsupported value dtype.");
}

#ifdef _METAL_
void COOMatmatPruneFill::eval_gpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &col = inputs[1];
  const auto &indptr = inputs[2];
  const auto &out_indptr = inputs[3];
  auto &out_data = outputs[0];
  auto &out_row = outputs[1];
  auto &out_col = outputs[2];

  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_row.set_data(mx::allocator::malloc(out_row.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      sparse_kernel_name("coo_matmat_prune_fill", data.dtype(), col.dtype()),
      lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(col, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(out_indptr, 3);
  encoder.set_output_array(out_data, 4);
  encoder.set_output_array(out_row, 5);
  encoder.set_output_array(out_col, 6);
  const int n_rows = static_cast<int>(indptr.size()) - 1;
  encoder.set_bytes(n_rows, 7);

  auto threads = std::max<size_t>(static_cast<size_t>(n_rows), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOMatmatPruneFill::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_matmat has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
coo_matmat_staged(const mx::array &lhs_data, const mx::array &lhs_row,
                  const mx::array &lhs_col, const mx::array &rhs_data,
                  const mx::array &rhs_row, const mx::array &rhs_col,
                  int lhs_n_rows, int lhs_n_cols, int rhs_n_rows,
                  int rhs_n_cols, mx::Dtype out_index_dtype) {
  auto stream = mx::default_stream(mx::default_device());

  auto [lhs_bucket_data, lhs_bucket_col, lhs_indptr] =
      coo_tocsr(lhs_data, lhs_row, lhs_col, lhs_n_rows, lhs_n_cols, stream);
  auto [rhs_bucket_data, rhs_bucket_col, rhs_indptr] =
      coo_tocsr(rhs_data, rhs_row, rhs_col, rhs_n_rows, rhs_n_cols, stream);

  auto counts = symbolic_counts(lhs_bucket_col, lhs_indptr, rhs_bucket_col,
                                rhs_indptr, lhs_n_rows, rhs_n_rows, rhs_n_cols,
                                out_index_dtype, stream);
  mx::eval(counts);
  auto [candidate_indptr, candidate_nnz] =
      build_indptr_from_counts(counts, lhs_n_rows, out_index_dtype);

  auto [candidate_data, candidate_col] =
      numeric_fill(lhs_bucket_data, lhs_bucket_col, lhs_indptr, rhs_bucket_data,
                   rhs_bucket_col, rhs_indptr, candidate_indptr, candidate_nnz,
                   lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype, stream);

  auto nonzero_counts =
      prune_counts(candidate_data, candidate_indptr, lhs_n_rows, stream);
  mx::eval(nonzero_counts);
  auto [out_indptr, out_nnz] =
      build_indptr_from_counts(nonzero_counts, lhs_n_rows, out_index_dtype);
  return prune_fill(candidate_data, candidate_col, candidate_indptr, out_indptr,
                    out_nnz, stream);
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

  if (use_experimental_metal_spgemm()) {
    return coo_matmat_staged(lhs_data, lhs_row, lhs_col, rhs_data, rhs_row,
                             rhs_col, lhs_n_rows, lhs_n_cols, rhs_n_rows,
                             rhs_n_cols, out_index_dtype);
  }

  const int requested_workers = configured_spgemm_worker_count();
  if (lhs_data.dtype() == mx::float32) {
    return dispatch_lhs<float>(lhs_data, lhs_row, lhs_col, rhs_data, rhs_row,
                               rhs_col, lhs_n_rows, lhs_n_cols, rhs_n_rows,
                               rhs_n_cols, out_index_dtype, requested_workers);
  }
  if (lhs_data.dtype() == mx::float16) {
    return dispatch_lhs<mx::float16_t>(
        lhs_data, lhs_row, lhs_col, rhs_data, rhs_row, rhs_col, lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype, requested_workers);
  }
  if (lhs_data.dtype() == mx::bfloat16) {
    return dispatch_lhs<mx::bfloat16_t>(
        lhs_data, lhs_row, lhs_col, rhs_data, rhs_row, rhs_col, lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype, requested_workers);
  }
  if (lhs_data.dtype() == mx::complex64) {
    return dispatch_lhs<mx::complex64_t>(
        lhs_data, lhs_row, lhs_col, rhs_data, rhs_row, rhs_col, lhs_n_rows,
        lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype, requested_workers);
  }
  throw std::runtime_error("coo_matmat unsupported value dtype.");
}

} // namespace mlx_sparse
