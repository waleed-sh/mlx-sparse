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

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

template <typename T> bool nonzero(T value) { return value != T{}; }

std::string csc_matmat_index_kernel_name(const std::string &prefix,
                                         mx::Dtype lhs_index_dtype,
                                         mx::Dtype rhs_index_dtype,
                                         mx::Dtype out_index_dtype) {
  return prefix + "_" + index_kernel_suffix(lhs_index_dtype) + "_" +
         index_kernel_suffix(rhs_index_dtype) + "_" +
         index_kernel_suffix(out_index_dtype);
}

std::string csc_matmat_numeric_kernel_name(mx::Dtype value_dtype,
                                           mx::Dtype lhs_index_dtype,
                                           mx::Dtype rhs_index_dtype,
                                           mx::Dtype out_index_dtype) {
  return "csc_matmat_numeric_" + value_kernel_suffix(value_dtype) + "_" +
         index_kernel_suffix(lhs_index_dtype) + "_" +
         index_kernel_suffix(rhs_index_dtype) + "_" +
         index_kernel_suffix(out_index_dtype);
}

bool use_experimental_metal_spgemm() {
  const char *flag = std::getenv("MLX_SPARSE_EXPERIMENTAL_METAL_SPGEMM");
  return flag != nullptr && std::string(flag) == "1" &&
         mx::default_device().type == mx::Device::gpu;
}

class CSCMatmatSymbolic : public mx::Primitive {
public:
  CSCMatmatSymbolic(mx::Stream stream, int lhs_n_rows, int lhs_n_cols,
                    int rhs_n_cols)
      : Primitive(stream), lhs_n_rows_(lhs_n_rows), lhs_n_cols_(lhs_n_cols),
        rhs_n_cols_(rhs_n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCMatmatSymbolic"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCMatmatSymbolic &>(other);
    return lhs_n_rows_ == rhs.lhs_n_rows_ && lhs_n_cols_ == rhs.lhs_n_cols_ &&
           rhs_n_cols_ == rhs.rhs_n_cols_;
  }

private:
  int lhs_n_rows_;
  int lhs_n_cols_;
  int rhs_n_cols_;
};

class CSCMatmatNumeric : public mx::Primitive {
public:
  CSCMatmatNumeric(mx::Stream stream, int lhs_n_rows, int lhs_n_cols,
                   int rhs_n_cols)
      : Primitive(stream), lhs_n_rows_(lhs_n_rows), lhs_n_cols_(lhs_n_cols),
        rhs_n_cols_(rhs_n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCMatmatNumeric"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCMatmatNumeric &>(other);
    return lhs_n_rows_ == rhs.lhs_n_rows_ && lhs_n_cols_ == rhs.lhs_n_cols_ &&
           rhs_n_cols_ == rhs.rhs_n_cols_;
  }

private:
  int lhs_n_rows_;
  int lhs_n_cols_;
  int rhs_n_cols_;
};

class CSCMatmatPruneCounts : public mx::Primitive {
public:
  explicit CSCMatmatPruneCounts(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCMatmatPruneCounts"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

class CSCMatmatPruneFill : public mx::Primitive {
public:
  explicit CSCMatmatPruneFill(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCMatmatPruneFill"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

template <typename LhsI, typename RhsI, typename OutI>
void symbolic_cpu_impl(const mx::array &lhs_indices,
                       const mx::array &lhs_indptr,
                       const mx::array &rhs_indices,
                       const mx::array &rhs_indptr, mx::array &counts,
                       int lhs_n_rows, int lhs_n_cols, int rhs_n_cols,
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
                    lhs_n_cols, rhs_n_cols]() mutable {
    const auto *lhs_indices_ptr = lhs_indices.data<LhsI>();
    const auto *lhs_indptr_ptr = lhs_indptr.data<LhsI>();
    const auto *rhs_indices_ptr = rhs_indices.data<RhsI>();
    const auto *rhs_indptr_ptr = rhs_indptr.data<RhsI>();
    auto *counts_ptr = counts.data<OutI>();

    std::vector<int> marker(static_cast<size_t>(lhs_n_rows), -1);
    for (int col = 0; col < rhs_n_cols; ++col) {
      int col_count = 0;
      for (RhsI rhs_pos = rhs_indptr_ptr[col];
           rhs_pos < rhs_indptr_ptr[col + 1]; ++rhs_pos) {
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
      counts_ptr[col] = static_cast<OutI>(col_count);
    }
  });
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
void numeric_cpu_impl(const mx::array &lhs_data, const mx::array &lhs_indices,
                      const mx::array &lhs_indptr, const mx::array &rhs_data,
                      const mx::array &rhs_indices, const mx::array &rhs_indptr,
                      const mx::array &out_indptr, mx::array &out_data,
                      mx::array &out_indices, int lhs_n_rows, int lhs_n_cols,
                      int rhs_n_cols, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(lhs_data);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(lhs_indptr);
  encoder.set_input_array(rhs_data);
  encoder.set_input_array(rhs_indices);
  encoder.set_input_array(rhs_indptr);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_indices);

  encoder.dispatch([lhs_data = mx::array::unsafe_weak_copy(lhs_data),
                    lhs_indices = mx::array::unsafe_weak_copy(lhs_indices),
                    lhs_indptr = mx::array::unsafe_weak_copy(lhs_indptr),
                    rhs_data = mx::array::unsafe_weak_copy(rhs_data),
                    rhs_indices = mx::array::unsafe_weak_copy(rhs_indices),
                    rhs_indptr = mx::array::unsafe_weak_copy(rhs_indptr),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_indices = mx::array::unsafe_weak_copy(out_indices),
                    lhs_n_rows, lhs_n_cols, rhs_n_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;

    const auto *lhs_data_ptr = lhs_data.data<T>();
    const auto *lhs_indices_ptr = lhs_indices.data<LhsI>();
    const auto *lhs_indptr_ptr = lhs_indptr.data<LhsI>();
    const auto *rhs_data_ptr = rhs_data.data<T>();
    const auto *rhs_indices_ptr = rhs_indices.data<RhsI>();
    const auto *rhs_indptr_ptr = rhs_indptr.data<RhsI>();
    const auto *out_indptr_ptr = out_indptr.data<OutI>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_indices_ptr = out_indices.data<OutI>();

    std::vector<int> marker(static_cast<size_t>(lhs_n_rows), -1);
    std::vector<AccT> accum(static_cast<size_t>(lhs_n_rows),
                            Accumulator<T>::zero());
    std::vector<int> rows;

    for (int col = 0; col < rhs_n_cols; ++col) {
      rows.clear();
      for (RhsI rhs_pos = rhs_indptr_ptr[col];
           rhs_pos < rhs_indptr_ptr[col + 1]; ++rhs_pos) {
        const int lhs_col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
        if (lhs_col < 0 || lhs_col >= lhs_n_cols) {
          throw std::invalid_argument(
              "csc_matmat rhs indices contain an out-of-bounds row.");
        }
        const T rhs_value = rhs_data_ptr[rhs_pos];
        for (LhsI lhs_pos = lhs_indptr_ptr[lhs_col];
             lhs_pos < lhs_indptr_ptr[lhs_col + 1]; ++lhs_pos) {
          const int row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
          if (row < 0 || row >= lhs_n_rows) {
            throw std::invalid_argument(
                "csc_matmat lhs indices contain an out-of-bounds row.");
          }
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
      OutI write = out_indptr_ptr[col];
      for (int row : rows) {
        const auto row_index = static_cast<size_t>(row);
        out_indices_ptr[write] = static_cast<OutI>(row);
        out_data_ptr[write] = Accumulator<T>::cast(accum[row_index]);
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
    const int n_cols = static_cast<int>(indptr.size()) - 1;

    for (int col = 0; col < n_cols; ++col) {
      I count = I{0};
      for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
        if (nonzero(data_ptr[p])) {
          count += I{1};
        }
      }
      counts_ptr[col] = count;
    }
  });
}

template <typename T, typename I>
void prune_fill_cpu_impl(const mx::array &data, const mx::array &indices,
                         const mx::array &indptr, const mx::array &out_indptr,
                         mx::array &out_data, mx::array &out_indices,
                         mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_indices);

  encoder.dispatch(
      [data = mx::array::unsafe_weak_copy(data),
       indices = mx::array::unsafe_weak_copy(indices),
       indptr = mx::array::unsafe_weak_copy(indptr),
       out_indptr = mx::array::unsafe_weak_copy(out_indptr),
       out_data = mx::array::unsafe_weak_copy(out_data),
       out_indices = mx::array::unsafe_weak_copy(out_indices)]() mutable {
        const auto *data_ptr = data.data<T>();
        const auto *indices_ptr = indices.data<I>();
        const auto *indptr_ptr = indptr.data<I>();
        const auto *out_indptr_ptr = out_indptr.data<I>();
        auto *out_data_ptr = out_data.data<T>();
        auto *out_indices_ptr = out_indices.data<I>();
        const int n_cols = static_cast<int>(indptr.size()) - 1;

        for (int col = 0; col < n_cols; ++col) {
          I write = out_indptr_ptr[col];
          for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
            const T value = data_ptr[p];
            if (nonzero(value)) {
              out_data_ptr[write] = value;
              out_indices_ptr[write] = indices_ptr[p];
              ++write;
            }
          }
        }
      });
}

template <typename I>
std::pair<mx::array, int> build_indptr_from_counts(const mx::array &counts,
                                                   int n_cols,
                                                   mx::Dtype index_dtype) {
  const auto *counts_ptr = counts.data<I>();
  std::vector<I> out_indptr(static_cast<size_t>(n_cols) + 1, I{0});
  int64_t total = 0;
  for (int col = 0; col < n_cols; ++col) {
    const auto count = static_cast<int64_t>(counts_ptr[col]);
    if (count < 0) {
      throw std::runtime_error("csc_matmat produced a negative column count.");
    }
    total += count;
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error(
          "csc_matmat output nnz exceeds MLX shape limits.");
    }
    if (total > static_cast<int64_t>(std::numeric_limits<I>::max())) {
      throw std::overflow_error(
          "csc_matmat output nnz exceeds index dtype capacity.");
    }
    out_indptr[static_cast<size_t>(col) + 1] = static_cast<I>(total);
  }

  return {mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    index_dtype),
          static_cast<int>(total)};
}

std::pair<mx::array, int> build_indptr_from_counts(const mx::array &counts,
                                                   int n_cols,
                                                   mx::Dtype index_dtype) {
  if (index_dtype == mx::int32) {
    return build_indptr_from_counts<int32_t>(counts, n_cols, index_dtype);
  }
  return build_indptr_from_counts<int64_t>(counts, n_cols, index_dtype);
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
  const long double average_lhs_col_nnz =
      static_cast<long double>(lhs_nnz) / static_cast<long double>(inner_dim);
  const long double estimated_products =
      static_cast<long double>(rhs_nnz) * average_lhs_col_nnz;
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

template <typename T, typename OutI> struct LocalCscSpgemmOutput {
  std::vector<T> data;
  std::vector<OutI> indices;
};

template <typename LhsI, typename RhsI>
std::vector<int64_t>
csc_col_work(const RhsI *rhs_indices_ptr, const RhsI *rhs_indptr_ptr,
             const LhsI *lhs_indptr_ptr, int lhs_n_cols, int rhs_n_cols) {
  std::vector<int64_t> work(static_cast<size_t>(rhs_n_cols), 0);
  for (int col = 0; col < rhs_n_cols; ++col) {
    int64_t col_work = 0;
    for (RhsI rhs_pos = rhs_indptr_ptr[col]; rhs_pos < rhs_indptr_ptr[col + 1];
         ++rhs_pos) {
      const int lhs_col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
      if (lhs_col < 0 || lhs_col >= lhs_n_cols) {
        throw std::invalid_argument(
            "csc_matmat rhs indices contain an out-of-bounds row.");
      }
      const auto segment = static_cast<int64_t>(lhs_indptr_ptr[lhs_col + 1]) -
                           static_cast<int64_t>(lhs_indptr_ptr[lhs_col]);
      if (segment < 0) {
        throw std::invalid_argument(
            "csc_matmat lhs indptr must be nondecreasing.");
      }
      col_work = saturated_add(col_work, segment);
    }
    work[static_cast<size_t>(col)] = col_work;
  }
  return work;
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
  std::vector<AccT> accum(static_cast<size_t>(lhs_n_rows),
                          Accumulator<T>::zero());
  std::vector<int> rows;
  std::vector<OutI> out_indptr(static_cast<size_t>(rhs_n_cols) + 1, OutI{0});
  std::vector<T> out_data;
  std::vector<OutI> out_indices;
  const size_t reserve_hint = spgemm_reserve_hint(
      rhs_n_cols, lhs_n_cols, lhs_n_rows, lhs_data.size(), rhs_data.size());
  out_data.reserve(reserve_hint);
  out_indices.reserve(reserve_hint);

  for (int col = 0; col < rhs_n_cols; ++col) {
    rows.clear();
    bool rows_sorted = true;
    int disorder_count = 0;
    for (RhsI rhs_pos = rhs_indptr_ptr[col]; rhs_pos < rhs_indptr_ptr[col + 1];
         ++rhs_pos) {
      const int lhs_col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
      if (lhs_col < 0 || lhs_col >= lhs_n_cols) {
        throw std::invalid_argument(
            "csc_matmat rhs indices contain an out-of-bounds row.");
      }
      const T rhs_value = rhs_data_ptr[rhs_pos];
      for (LhsI lhs_pos = lhs_indptr_ptr[lhs_col];
           lhs_pos < lhs_indptr_ptr[lhs_col + 1]; ++lhs_pos) {
        const int row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
        const auto row_index = static_cast<size_t>(row);
        const AccT product =
            multiply_accumulate<T>(lhs_data_ptr[lhs_pos], rhs_value);
        if (marker[row_index] != col) {
          marker[row_index] = col;
          accum[row_index] = product;
          if (!rows.empty() && row < rows.back()) {
            rows_sorted = false;
            disorder_count += 1;
          }
          rows.push_back(row);
        } else {
          accum[row_index] += product;
        }
      }
    }

    if (!rows_sorted &&
        use_dense_ordered_scan(rows.size(), lhs_n_rows, disorder_count)) {
      for (int row = 0; row < lhs_n_rows; ++row) {
        const auto row_index = static_cast<size_t>(row);
        if (marker[row_index] != col) {
          continue;
        }
        const auto value = Accumulator<T>::cast(accum[row_index]);
        if (nonzero(value)) {
          out_data.push_back(value);
          out_indices.push_back(static_cast<OutI>(row));
        }
      }
    } else {
      if (!rows_sorted) {
        sort_touched_indices(rows);
      }
      for (int row : rows) {
        const auto value =
            Accumulator<T>::cast(accum[static_cast<size_t>(row)]);
        if (nonzero(value)) {
          out_data.push_back(value);
          out_indices.push_back(static_cast<OutI>(row));
        }
      }
    }
    check_output_nnz<OutI>(out_data.size(), "csc_matmat");
    out_indptr[static_cast<size_t>(col) + 1] =
        static_cast<OutI>(out_data.size());
  }

  const int out_nnz = static_cast<int>(out_data.size());
  return {mx::array(out_data.begin(), mx::Shape{out_nnz}, lhs_data.dtype()),
          mx::array(out_indices.begin(), mx::Shape{out_nnz}, out_index_dtype),
          mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    out_index_dtype)};
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
std::tuple<mx::array, mx::array, mx::array> csc_matmat_parallel_impl(
    mx::array lhs_data, mx::array lhs_indices, mx::array lhs_indptr,
    mx::array rhs_data, mx::array rhs_indices, mx::array rhs_indptr,
    int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
    mx::Dtype out_index_dtype, int requested_workers) {
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

  const auto col_work = csc_col_work(rhs_indices_ptr, rhs_indptr_ptr,
                                     lhs_indptr_ptr, lhs_n_cols, rhs_n_cols);
  const auto ranges = cpu_ranges_for_output_work(col_work, requested_workers);
  std::vector<OutI> col_counts(static_cast<size_t>(rhs_n_cols), OutI{0});
  std::vector<LocalCscSpgemmOutput<T, OutI>> local_outputs(ranges.size());
  const size_t reserve_hint = spgemm_reserve_hint(
      rhs_n_cols, lhs_n_cols, lhs_n_rows, lhs_data.size(), rhs_data.size());
  int64_t total_work = 0;
  for (const auto work : col_work) {
    total_work = saturated_add(total_work, work);
  }

  parallel_for_cpu_ranges_indexed(ranges, [&](size_t worker, CpuRange range) {
    auto &local = local_outputs[worker];
    int64_t local_work = 0;
    for (int col = range.begin; col < range.end; ++col) {
      local_work =
          saturated_add(local_work, col_work[static_cast<size_t>(col)]);
    }
    const auto reserve =
        local_reserve_hint(reserve_hint, local_work, total_work);
    local.data.reserve(reserve);
    local.indices.reserve(reserve);

    std::vector<int> marker(static_cast<size_t>(lhs_n_rows), -1);
    std::vector<AccT> accum(static_cast<size_t>(lhs_n_rows),
                            Accumulator<T>::zero());
    std::vector<int> rows;

    for (int col = range.begin; col < range.end; ++col) {
      rows.clear();
      bool rows_sorted = true;
      int disorder_count = 0;
      const auto before = local.data.size();
      for (RhsI rhs_pos = rhs_indptr_ptr[col];
           rhs_pos < rhs_indptr_ptr[col + 1]; ++rhs_pos) {
        const int lhs_col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
        const T rhs_value = rhs_data_ptr[rhs_pos];
        for (LhsI lhs_pos = lhs_indptr_ptr[lhs_col];
             lhs_pos < lhs_indptr_ptr[lhs_col + 1]; ++lhs_pos) {
          const int row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
          const auto row_index = static_cast<size_t>(row);
          const AccT product =
              multiply_accumulate<T>(lhs_data_ptr[lhs_pos], rhs_value);
          if (marker[row_index] != col) {
            marker[row_index] = col;
            accum[row_index] = product;
            if (!rows.empty() && row < rows.back()) {
              rows_sorted = false;
              disorder_count += 1;
            }
            rows.push_back(row);
          } else {
            accum[row_index] += product;
          }
        }
      }

      if (!rows_sorted &&
          use_dense_ordered_scan(rows.size(), lhs_n_rows, disorder_count)) {
        for (int row = 0; row < lhs_n_rows; ++row) {
          const auto row_index = static_cast<size_t>(row);
          if (marker[row_index] != col) {
            continue;
          }
          const auto value = Accumulator<T>::cast(accum[row_index]);
          if (nonzero(value)) {
            local.data.push_back(value);
            local.indices.push_back(static_cast<OutI>(row));
          }
        }
      } else {
        if (!rows_sorted) {
          sort_touched_indices(rows);
        }
        for (int row : rows) {
          const auto value =
              Accumulator<T>::cast(accum[static_cast<size_t>(row)]);
          if (nonzero(value)) {
            local.data.push_back(value);
            local.indices.push_back(static_cast<OutI>(row));
          }
        }
      }

      const auto col_nnz = local.data.size() - before;
      check_output_nnz<OutI>(col_nnz, "csc_matmat column");
      col_counts[static_cast<size_t>(col)] = static_cast<OutI>(col_nnz);
    }
  });

  std::vector<OutI> out_indptr(static_cast<size_t>(rhs_n_cols) + 1, OutI{0});
  size_t total_nnz = 0;
  for (int col = 0; col < rhs_n_cols; ++col) {
    total_nnz += static_cast<size_t>(col_counts[static_cast<size_t>(col)]);
    check_output_nnz<OutI>(total_nnz, "csc_matmat");
    out_indptr[static_cast<size_t>(col) + 1] = static_cast<OutI>(total_nnz);
  }

  std::vector<T> out_data(total_nnz);
  std::vector<OutI> out_indices(total_nnz);
  for (size_t worker = 0; worker < ranges.size(); ++worker) {
    const auto &range = ranges[worker];
    const auto &local = local_outputs[worker];
    size_t read = 0;
    for (int col = range.begin; col < range.end; ++col) {
      const auto count =
          static_cast<size_t>(col_counts[static_cast<size_t>(col)]);
      const auto write =
          static_cast<size_t>(out_indptr[static_cast<size_t>(col)]);
      std::copy(local.data.begin() + static_cast<std::ptrdiff_t>(read),
                local.data.begin() + static_cast<std::ptrdiff_t>(read + count),
                out_data.begin() + static_cast<std::ptrdiff_t>(write));
      std::copy(local.indices.begin() + static_cast<std::ptrdiff_t>(read),
                local.indices.begin() +
                    static_cast<std::ptrdiff_t>(read + count),
                out_indices.begin() + static_cast<std::ptrdiff_t>(write));
      read += count;
    }
    if (read != local.data.size() || read != local.indices.size()) {
      throw std::runtime_error("csc_matmat internal parallel count mismatch.");
    }
  }

  const int out_nnz = static_cast<int>(total_nnz);
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
             mx::Dtype out_index_dtype, int requested_workers) {
  if (out_index_dtype == mx::int32) {
    if (requested_workers > 1) {
      return csc_matmat_parallel_impl<T, LhsI, RhsI, int32_t>(
          std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
          std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
          lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
          requested_workers);
    }
    return csc_matmat_impl<T, LhsI, RhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  if (requested_workers > 1) {
    return csc_matmat_parallel_impl<T, LhsI, RhsI, int64_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
        requested_workers);
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
             mx::Dtype out_index_dtype, int requested_workers) {
  if (rhs_indices.dtype() == mx::int32) {
    return dispatch_out<T, LhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
        requested_workers);
  }
  return dispatch_out<T, LhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
      requested_workers);
}

template <typename T>
std::tuple<mx::array, mx::array, mx::array>
dispatch_lhs(mx::array lhs_data, mx::array lhs_indices, mx::array lhs_indptr,
             mx::array rhs_data, mx::array rhs_indices, mx::array rhs_indptr,
             int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
             mx::Dtype out_index_dtype, int requested_workers) {
  if (lhs_indices.dtype() == mx::int32) {
    return dispatch_rhs<T, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
        requested_workers);
  }
  return dispatch_rhs<T, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
      requested_workers);
}

mx::array symbolic_counts(const mx::array &lhs_indices,
                          const mx::array &lhs_indptr,
                          const mx::array &rhs_indices,
                          const mx::array &rhs_indptr, int lhs_n_rows,
                          int lhs_n_cols, int rhs_n_cols,
                          mx::Dtype out_index_dtype, mx::Stream stream) {
  auto primitive = std::make_shared<CSCMatmatSymbolic>(stream, lhs_n_rows,
                                                       lhs_n_cols, rhs_n_cols);
  return mx::array(mx::Shape{rhs_n_cols}, out_index_dtype, primitive,
                   {lhs_indices, lhs_indptr, rhs_indices, rhs_indptr});
}

std::tuple<mx::array, mx::array>
numeric_fill(const mx::array &lhs_data, const mx::array &lhs_indices,
             const mx::array &lhs_indptr, const mx::array &rhs_data,
             const mx::array &rhs_indices, const mx::array &rhs_indptr,
             const mx::array &out_indptr, int out_nnz, int lhs_n_rows,
             int lhs_n_cols, int rhs_n_cols, mx::Dtype out_index_dtype,
             mx::Stream stream) {
  auto primitive = std::make_shared<CSCMatmatNumeric>(stream, lhs_n_rows,
                                                      lhs_n_cols, rhs_n_cols);
  auto outputs =
      mx::array::make_arrays({mx::Shape{out_nnz}, mx::Shape{out_nnz}},
                             {lhs_data.dtype(), out_index_dtype}, primitive,
                             {lhs_data, lhs_indices, lhs_indptr, rhs_data,
                              rhs_indices, rhs_indptr, out_indptr});
  return {outputs[0], outputs[1]};
}

mx::array prune_counts(const mx::array &data, const mx::array &indptr,
                       int n_cols, mx::Stream stream) {
  auto primitive = std::make_shared<CSCMatmatPruneCounts>(stream);
  return mx::array(mx::Shape{n_cols}, indptr.dtype(), primitive,
                   {data, indptr});
}

std::tuple<mx::array, mx::array> prune_fill(const mx::array &data,
                                            const mx::array &indices,
                                            const mx::array &indptr,
                                            const mx::array &out_indptr,
                                            int out_nnz, mx::Stream stream) {
  auto primitive = std::make_shared<CSCMatmatPruneFill>(stream);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}}, {data.dtype(), indices.dtype()},
      primitive, {data, indices, indptr, out_indptr});
  return {outputs[0], outputs[1]};
}

template <typename LhsI, typename RhsI>
void dispatch_symbolic_out(const mx::array &lhs_indices,
                           const mx::array &lhs_indptr,
                           const mx::array &rhs_indices,
                           const mx::array &rhs_indptr, mx::array &counts,
                           int lhs_n_rows, int lhs_n_cols, int rhs_n_cols,
                           mx::Stream stream) {
  if (counts.dtype() == mx::int32) {
    symbolic_cpu_impl<LhsI, RhsI, int32_t>(lhs_indices, lhs_indptr, rhs_indices,
                                           rhs_indptr, counts, lhs_n_rows,
                                           lhs_n_cols, rhs_n_cols, stream);
    return;
  }
  symbolic_cpu_impl<LhsI, RhsI, int64_t>(lhs_indices, lhs_indptr, rhs_indices,
                                         rhs_indptr, counts, lhs_n_rows,
                                         lhs_n_cols, rhs_n_cols, stream);
}

template <typename LhsI>
void dispatch_symbolic_rhs(const mx::array &lhs_indices,
                           const mx::array &lhs_indptr,
                           const mx::array &rhs_indices,
                           const mx::array &rhs_indptr, mx::array &counts,
                           int lhs_n_rows, int lhs_n_cols, int rhs_n_cols,
                           mx::Stream stream) {
  if (rhs_indices.dtype() == mx::int32) {
    dispatch_symbolic_out<LhsI, int32_t>(lhs_indices, lhs_indptr, rhs_indices,
                                         rhs_indptr, counts, lhs_n_rows,
                                         lhs_n_cols, rhs_n_cols, stream);
    return;
  }
  dispatch_symbolic_out<LhsI, int64_t>(lhs_indices, lhs_indptr, rhs_indices,
                                       rhs_indptr, counts, lhs_n_rows,
                                       lhs_n_cols, rhs_n_cols, stream);
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
void numeric_cpu_dispatch_out(
    const mx::array &lhs_data, const mx::array &lhs_indices,
    const mx::array &lhs_indptr, const mx::array &rhs_data,
    const mx::array &rhs_indices, const mx::array &rhs_indptr,
    const mx::array &out_indptr, mx::array &out_data, mx::array &out_indices,
    int lhs_n_rows, int lhs_n_cols, int rhs_n_cols, mx::Stream stream) {
  numeric_cpu_impl<T, LhsI, RhsI, OutI>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_indices, lhs_n_rows, lhs_n_cols, rhs_n_cols,
      stream);
}

template <typename T, typename LhsI, typename RhsI>
void numeric_cpu_dispatch_index(
    const mx::array &lhs_data, const mx::array &lhs_indices,
    const mx::array &lhs_indptr, const mx::array &rhs_data,
    const mx::array &rhs_indices, const mx::array &rhs_indptr,
    const mx::array &out_indptr, mx::array &out_data, mx::array &out_indices,
    int lhs_n_rows, int lhs_n_cols, int rhs_n_cols, mx::Stream stream) {
  if (out_indices.dtype() == mx::int32) {
    numeric_cpu_dispatch_out<T, LhsI, RhsI, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        out_indptr, out_data, out_indices, lhs_n_rows, lhs_n_cols, rhs_n_cols,
        stream);
    return;
  }
  numeric_cpu_dispatch_out<T, LhsI, RhsI, int64_t>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_indices, lhs_n_rows, lhs_n_cols, rhs_n_cols,
      stream);
}

template <typename T, typename LhsI>
void numeric_cpu_dispatch_rhs(
    const mx::array &lhs_data, const mx::array &lhs_indices,
    const mx::array &lhs_indptr, const mx::array &rhs_data,
    const mx::array &rhs_indices, const mx::array &rhs_indptr,
    const mx::array &out_indptr, mx::array &out_data, mx::array &out_indices,
    int lhs_n_rows, int lhs_n_cols, int rhs_n_cols, mx::Stream stream) {
  if (rhs_indices.dtype() == mx::int32) {
    numeric_cpu_dispatch_index<T, LhsI, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        out_indptr, out_data, out_indices, lhs_n_rows, lhs_n_cols, rhs_n_cols,
        stream);
    return;
  }
  numeric_cpu_dispatch_index<T, LhsI, int64_t>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_indices, lhs_n_rows, lhs_n_cols, rhs_n_cols,
      stream);
}

template <typename T>
void numeric_cpu_dispatch_lhs(
    const mx::array &lhs_data, const mx::array &lhs_indices,
    const mx::array &lhs_indptr, const mx::array &rhs_data,
    const mx::array &rhs_indices, const mx::array &rhs_indptr,
    const mx::array &out_indptr, mx::array &out_data, mx::array &out_indices,
    int lhs_n_rows, int lhs_n_cols, int rhs_n_cols, mx::Stream stream) {
  if (lhs_indices.dtype() == mx::int32) {
    numeric_cpu_dispatch_rhs<T, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        out_indptr, out_data, out_indices, lhs_n_rows, lhs_n_cols, rhs_n_cols,
        stream);
    return;
  }
  numeric_cpu_dispatch_rhs<T, int64_t>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_indices, lhs_n_rows, lhs_n_cols, rhs_n_cols,
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
void dispatch_prune_fill(const mx::array &data, const mx::array &indices,
                         const mx::array &indptr, const mx::array &out_indptr,
                         mx::array &out_data, mx::array &out_indices,
                         mx::Stream stream) {
  if (indices.dtype() == mx::int32) {
    prune_fill_cpu_impl<T, int32_t>(data, indices, indptr, out_indptr, out_data,
                                    out_indices, stream);
    return;
  }
  prune_fill_cpu_impl<T, int64_t>(data, indices, indptr, out_indptr, out_data,
                                  out_indices, stream);
}

void CSCMatmatSymbolic::eval_cpu(const std::vector<mx::array> &inputs,
                                 std::vector<mx::array> &outputs) {
  const auto &lhs_indices = inputs[0];
  const auto &lhs_indptr = inputs[1];
  const auto &rhs_indices = inputs[2];
  const auto &rhs_indptr = inputs[3];
  auto &counts = outputs[0];

  if (lhs_indices.dtype() == mx::int32) {
    dispatch_symbolic_rhs<int32_t>(lhs_indices, lhs_indptr, rhs_indices,
                                   rhs_indptr, counts, lhs_n_rows_, lhs_n_cols_,
                                   rhs_n_cols_, stream());
    return;
  }
  dispatch_symbolic_rhs<int64_t>(lhs_indices, lhs_indptr, rhs_indices,
                                 rhs_indptr, counts, lhs_n_rows_, lhs_n_cols_,
                                 rhs_n_cols_, stream());
}

#ifdef _METAL_
void CSCMatmatSymbolic::eval_gpu(const std::vector<mx::array> &inputs,
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
      csc_matmat_index_kernel_name("csc_matmat_symbolic", lhs_indices.dtype(),
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
  encoder.set_bytes(lhs_n_cols_, 6);
  encoder.set_bytes(rhs_n_cols_, 7);

  auto threads = std::max<size_t>(static_cast<size_t>(rhs_n_cols_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCMatmatSymbolic::eval_gpu(const std::vector<mx::array> &,
                                 std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_matmat has no GPU implementation in this build.");
}
#endif

void CSCMatmatNumeric::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &lhs_indices = inputs[1];
  const auto &lhs_indptr = inputs[2];
  const auto &rhs_data = inputs[3];
  const auto &rhs_indices = inputs[4];
  const auto &rhs_indptr = inputs[5];
  const auto &out_indptr = inputs[6];

#define DISPATCH_CSC_MATMAT_NUMERIC_VALUE(DTYPE, TYPE)                         \
  if (lhs_data.dtype() == DTYPE) {                                             \
    numeric_cpu_dispatch_lhs<TYPE>(                                            \
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,  \
        out_indptr, outputs[0], outputs[1], lhs_n_rows_, lhs_n_cols_,          \
        rhs_n_cols_, stream());                                                \
    return;                                                                    \
  }

  DISPATCH_CSC_MATMAT_NUMERIC_VALUE(mx::float32, float)
  DISPATCH_CSC_MATMAT_NUMERIC_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSC_MATMAT_NUMERIC_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_MATMAT_NUMERIC_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_MATMAT_NUMERIC_VALUE

  throw std::runtime_error("csc_matmat unsupported value dtype.");
}

#ifdef _METAL_
void CSCMatmatNumeric::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &lhs_indices = inputs[1];
  const auto &lhs_indptr = inputs[2];
  const auto &rhs_data = inputs[3];
  const auto &rhs_indices = inputs[4];
  const auto &rhs_indptr = inputs[5];
  const auto &out_indptr = inputs[6];
  auto &out_data = outputs[0];
  auto &out_indices = outputs[1];

  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      csc_matmat_numeric_kernel_name(lhs_data.dtype(), lhs_indices.dtype(),
                                     rhs_indices.dtype(), out_indices.dtype()),
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
  encoder.set_output_array(out_indices, 8);
  encoder.set_bytes(lhs_n_rows_, 9);
  encoder.set_bytes(lhs_n_cols_, 10);
  encoder.set_bytes(rhs_n_cols_, 11);

  auto threads = std::max<size_t>(static_cast<size_t>(rhs_n_cols_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCMatmatNumeric::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_matmat has no GPU implementation in this build.");
}
#endif

void CSCMatmatPruneCounts::eval_cpu(const std::vector<mx::array> &inputs,
                                    std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indptr = inputs[1];

#define DISPATCH_CSC_MATMAT_PRUNE_COUNTS(DTYPE, TYPE)                          \
  if (data.dtype() == DTYPE) {                                                 \
    dispatch_prune_counts<TYPE>(data, indptr, outputs[0], stream());           \
    return;                                                                    \
  }

  DISPATCH_CSC_MATMAT_PRUNE_COUNTS(mx::float32, float)
  DISPATCH_CSC_MATMAT_PRUNE_COUNTS(mx::float16, mx::float16_t)
  DISPATCH_CSC_MATMAT_PRUNE_COUNTS(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_MATMAT_PRUNE_COUNTS(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_MATMAT_PRUNE_COUNTS

  throw std::runtime_error("csc_matmat unsupported value dtype.");
}

#ifdef _METAL_
void CSCMatmatPruneCounts::eval_gpu(const std::vector<mx::array> &inputs,
                                    std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indptr = inputs[1];
  auto &counts = outputs[0];

  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel =
      device.get_kernel(sparse_kernel_name("csc_matmat_prune_counts",
                                           data.dtype(), indptr.dtype()),
                        lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indptr, 1);
  encoder.set_output_array(counts, 2);
  const int n_cols = static_cast<int>(indptr.size()) - 1;
  encoder.set_bytes(n_cols, 3);

  auto threads = std::max<size_t>(static_cast<size_t>(n_cols), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCMatmatPruneCounts::eval_gpu(const std::vector<mx::array> &,
                                    std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_matmat has no GPU implementation in this build.");
}
#endif

void CSCMatmatPruneFill::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indices = inputs[1];
  const auto &indptr = inputs[2];
  const auto &out_indptr = inputs[3];

#define DISPATCH_CSC_MATMAT_PRUNE_FILL(DTYPE, TYPE)                            \
  if (data.dtype() == DTYPE) {                                                 \
    dispatch_prune_fill<TYPE>(data, indices, indptr, out_indptr, outputs[0],   \
                              outputs[1], stream());                           \
    return;                                                                    \
  }

  DISPATCH_CSC_MATMAT_PRUNE_FILL(mx::float32, float)
  DISPATCH_CSC_MATMAT_PRUNE_FILL(mx::float16, mx::float16_t)
  DISPATCH_CSC_MATMAT_PRUNE_FILL(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_MATMAT_PRUNE_FILL(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_MATMAT_PRUNE_FILL

  throw std::runtime_error("csc_matmat unsupported value dtype.");
}

#ifdef _METAL_
void CSCMatmatPruneFill::eval_gpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indices = inputs[1];
  const auto &indptr = inputs[2];
  const auto &out_indptr = inputs[3];
  auto &out_data = outputs[0];
  auto &out_indices = outputs[1];

  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel =
      device.get_kernel(sparse_kernel_name("csc_matmat_prune_fill",
                                           data.dtype(), indices.dtype()),
                        lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(out_indptr, 3);
  encoder.set_output_array(out_data, 4);
  encoder.set_output_array(out_indices, 5);
  const int n_cols = static_cast<int>(indptr.size()) - 1;
  encoder.set_bytes(n_cols, 6);

  auto threads = std::max<size_t>(static_cast<size_t>(n_cols), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCMatmatPruneFill::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_matmat has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
csc_matmat_staged(const mx::array &lhs_data, const mx::array &lhs_indices,
                  const mx::array &lhs_indptr, const mx::array &rhs_data,
                  const mx::array &rhs_indices, const mx::array &rhs_indptr,
                  int lhs_n_rows, int lhs_n_cols, int rhs_n_cols,
                  mx::Dtype out_index_dtype) {
  auto stream = mx::default_stream(mx::default_device());
  auto lhs_data_contig = mx::contiguous(lhs_data, false, stream);
  auto lhs_indices_contig = mx::contiguous(lhs_indices, false, stream);
  auto lhs_indptr_contig = mx::contiguous(lhs_indptr, false, stream);
  auto rhs_data_contig = mx::contiguous(rhs_data, false, stream);
  auto rhs_indices_contig = mx::contiguous(rhs_indices, false, stream);
  auto rhs_indptr_contig = mx::contiguous(rhs_indptr, false, stream);

  auto counts =
      symbolic_counts(lhs_indices_contig, lhs_indptr_contig, rhs_indices_contig,
                      rhs_indptr_contig, lhs_n_rows, lhs_n_cols, rhs_n_cols,
                      out_index_dtype, stream);
  mx::eval(counts);
  auto [candidate_indptr, candidate_nnz] =
      build_indptr_from_counts(counts, rhs_n_cols, out_index_dtype);

  auto [candidate_data, candidate_indices] = numeric_fill(
      lhs_data_contig, lhs_indices_contig, lhs_indptr_contig, rhs_data_contig,
      rhs_indices_contig, rhs_indptr_contig, candidate_indptr, candidate_nnz,
      lhs_n_rows, lhs_n_cols, rhs_n_cols, out_index_dtype, stream);

  auto nonzero_counts =
      prune_counts(candidate_data, candidate_indptr, rhs_n_cols, stream);
  mx::eval(nonzero_counts);
  auto [out_indptr, out_nnz] =
      build_indptr_from_counts(nonzero_counts, rhs_n_cols, out_index_dtype);
  auto [out_data, out_indices] =
      prune_fill(candidate_data, candidate_indices, candidate_indptr,
                 out_indptr, out_nnz, stream);
  return {out_data, out_indices, out_indptr};
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

  if (use_experimental_metal_spgemm()) {
    return csc_matmat_staged(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                             rhs_indices, rhs_indptr, lhs_n_rows, lhs_n_cols,
                             rhs_n_cols, out_index_dtype);
  }

  const int requested_workers = configured_spgemm_worker_count();
  if (lhs_data.dtype() == mx::float32) {
    return dispatch_lhs<float>(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                               rhs_indices, rhs_indptr, lhs_n_rows, lhs_n_cols,
                               rhs_n_rows, rhs_n_cols, out_index_dtype,
                               requested_workers);
  }
  if (lhs_data.dtype() == mx::float16) {
    return dispatch_lhs<mx::float16_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
        requested_workers);
  }
  if (lhs_data.dtype() == mx::bfloat16) {
    return dispatch_lhs<mx::bfloat16_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
        requested_workers);
  }
  if (lhs_data.dtype() == mx::complex64) {
    return dispatch_lhs<mx::complex64_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols, out_index_dtype,
        requested_workers);
  }
  throw std::runtime_error("csc_matmat unsupported value dtype.");
}

} // namespace mlx_sparse
