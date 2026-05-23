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

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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

namespace mlx_sparse {

namespace {

template <typename T>
typename Accumulator<T>::Type accumulator_value(T value) {
  using AccT = typename Accumulator<T>::Type;
  if constexpr (std::is_same_v<T, mx::float16_t> ||
                std::is_same_v<T, mx::bfloat16_t>) {
    return static_cast<float>(value);
  } else {
    return static_cast<AccT>(value);
  }
}

std::string matmat_index_kernel_name(const std::string &prefix,
                                     mx::Dtype lhs_index_dtype,
                                     mx::Dtype rhs_index_dtype,
                                     mx::Dtype out_index_dtype) {
  return prefix + "_" + index_kernel_suffix(lhs_index_dtype) + "_" +
         index_kernel_suffix(rhs_index_dtype) + "_" +
         index_kernel_suffix(out_index_dtype);
}

std::string matmat_numeric_kernel_name(mx::Dtype value_dtype,
                                       mx::Dtype lhs_index_dtype,
                                       mx::Dtype rhs_index_dtype,
                                       mx::Dtype out_index_dtype) {
  return "csr_matmat_numeric_" + value_kernel_suffix(value_dtype) + "_" +
         index_kernel_suffix(lhs_index_dtype) + "_" +
         index_kernel_suffix(rhs_index_dtype) + "_" +
         index_kernel_suffix(out_index_dtype);
}

bool use_experimental_metal_spgemm() {
  const char *flag = std::getenv("MLX_SPARSE_EXPERIMENTAL_METAL_SPGEMM");
  return flag != nullptr && std::string(flag) == "1" &&
         mx::default_device().type == mx::Device::gpu;
}

class CSRMatmatSymbolic : public mx::Primitive {
public:
  CSRMatmatSymbolic(mx::Stream stream, int lhs_n_rows, int rhs_n_rows,
                    int rhs_n_cols)
      : Primitive(stream), lhs_n_rows_(lhs_n_rows), rhs_n_rows_(rhs_n_rows),
        rhs_n_cols_(rhs_n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRMatmatSymbolic"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRMatmatSymbolic &>(other);
    return lhs_n_rows_ == rhs.lhs_n_rows_ && rhs_n_rows_ == rhs.rhs_n_rows_ &&
           rhs_n_cols_ == rhs.rhs_n_cols_;
  }

private:
  int lhs_n_rows_;
  int rhs_n_rows_;
  int rhs_n_cols_;
};

class CSRMatmatNumeric : public mx::Primitive {
public:
  CSRMatmatNumeric(mx::Stream stream, int lhs_n_rows, int rhs_n_rows,
                   int rhs_n_cols)
      : Primitive(stream), lhs_n_rows_(lhs_n_rows), rhs_n_rows_(rhs_n_rows),
        rhs_n_cols_(rhs_n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRMatmatNumeric"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRMatmatNumeric &>(other);
    return lhs_n_rows_ == rhs.lhs_n_rows_ && rhs_n_rows_ == rhs.rhs_n_rows_ &&
           rhs_n_cols_ == rhs.rhs_n_cols_;
  }

private:
  int lhs_n_rows_;
  int rhs_n_rows_;
  int rhs_n_cols_;
};

class CSRMatmatPruneCounts : public mx::Primitive {
public:
  explicit CSRMatmatPruneCounts(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRMatmatPruneCounts"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

class CSRMatmatPruneFill : public mx::Primitive {
public:
  explicit CSRMatmatPruneFill(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRMatmatPruneFill"; }

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
              "csr_matmat lhs indices contain an out-of-bounds column.");
        }
        for (RhsI rhs_pos = rhs_indptr_ptr[rhs_row];
             rhs_pos < rhs_indptr_ptr[rhs_row + 1]; ++rhs_pos) {
          const int col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
          if (col < 0 || col >= rhs_n_cols) {
            throw std::invalid_argument(
                "csr_matmat rhs indices contain an out-of-bounds column.");
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
                      const mx::array &rhs_indices,
                      const mx::array &rhs_indptr,
                      const mx::array &out_indptr, mx::array &out_data,
                      mx::array &out_indices, int lhs_n_rows, int rhs_n_rows,
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
                    lhs_n_rows, rhs_n_rows, rhs_n_cols]() mutable {
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
        out_indices_ptr[write] = static_cast<OutI>(col);
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
        if (data_ptr[p] != T{}) {
          count += I{1};
        }
      }
      counts_ptr[row] = count;
    }
  });
}

template <typename T, typename I>
void prune_fill_cpu_impl(const mx::array &data, const mx::array &indices,
                         const mx::array &indptr,
                         const mx::array &out_indptr, mx::array &out_data,
                         mx::array &out_indices, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_indices);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_indices =
                        mx::array::unsafe_weak_copy(out_indices)]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *out_indptr_ptr = out_indptr.data<I>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_indices_ptr = out_indices.data<I>();
    const int n_rows = static_cast<int>(indptr.size()) - 1;

    for (int row = 0; row < n_rows; ++row) {
      I write = out_indptr_ptr[row];
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const T value = data_ptr[p];
        if (value != T{}) {
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
                                                   int n_rows,
                                                   mx::Dtype index_dtype) {
  const auto *counts_ptr = counts.data<I>();
  std::vector<I> out_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  int64_t total = 0;
  for (int row = 0; row < n_rows; ++row) {
    const auto count = static_cast<int64_t>(counts_ptr[row]);
    if (count < 0) {
      throw std::runtime_error("csr_matmat produced a negative row count.");
    }
    total += count;
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error("csr_matmat output nnz exceeds MLX shape limits.");
    }
    if (total > static_cast<int64_t>(std::numeric_limits<I>::max())) {
      throw std::overflow_error(
          "csr_matmat output nnz exceeds index dtype capacity.");
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

template <typename I>
int prefix_counts(const std::vector<I> &counts, std::vector<I> &indptr) {
  indptr.resize(counts.size() + 1);
  indptr[0] = I{0};
  int64_t total = 0;
  for (size_t row = 0; row < counts.size(); ++row) {
    total += static_cast<int64_t>(counts[row]);
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error("csr_matmat output nnz exceeds MLX shape limits.");
    }
    if (total > static_cast<int64_t>(std::numeric_limits<I>::max())) {
      throw std::overflow_error(
          "csr_matmat output nnz exceeds index dtype capacity.");
    }
    indptr[row + 1] = static_cast<I>(total);
  }
  return static_cast<int>(total);
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
std::tuple<mx::array, mx::array, mx::array>
csr_matmat_host_impl(mx::array lhs_data, mx::array lhs_indices,
                     mx::array lhs_indptr, mx::array rhs_data,
                     mx::array rhs_indices, mx::array rhs_indptr,
                     int lhs_n_rows, int rhs_n_rows, int rhs_n_cols,
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

  std::vector<int> marker(static_cast<size_t>(rhs_n_cols), -1);
  std::vector<OutI> candidate_counts(static_cast<size_t>(lhs_n_rows), OutI{0});

  for (int row = 0; row < lhs_n_rows; ++row) {
    int row_count = 0;
    for (LhsI lhs_pos = lhs_indptr_ptr[row];
         lhs_pos < lhs_indptr_ptr[row + 1]; ++lhs_pos) {
      const int rhs_row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
      if (rhs_row < 0 || rhs_row >= rhs_n_rows) {
        throw std::invalid_argument(
            "csr_matmat lhs indices contain an out-of-bounds column.");
      }
      for (RhsI rhs_pos = rhs_indptr_ptr[rhs_row];
           rhs_pos < rhs_indptr_ptr[rhs_row + 1]; ++rhs_pos) {
        const int col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
        if (col < 0 || col >= rhs_n_cols) {
          throw std::invalid_argument(
              "csr_matmat rhs indices contain an out-of-bounds column.");
        }
        const auto col_index = static_cast<size_t>(col);
        if (marker[col_index] != row) {
          marker[col_index] = row;
          row_count += 1;
        }
      }
    }
    candidate_counts[static_cast<size_t>(row)] = static_cast<OutI>(row_count);
  }

  std::vector<OutI> candidate_indptr;
  const int candidate_nnz = prefix_counts(candidate_counts, candidate_indptr);
  std::vector<T> candidate_data(static_cast<size_t>(candidate_nnz));
  std::vector<OutI> candidate_indices(static_cast<size_t>(candidate_nnz));

  std::fill(marker.begin(), marker.end(), -1);
  std::vector<AccT> accum(static_cast<size_t>(rhs_n_cols),
                          Accumulator<T>::zero());
  std::vector<int> columns;
  for (int row = 0; row < lhs_n_rows; ++row) {
    columns.clear();
    for (LhsI lhs_pos = lhs_indptr_ptr[row];
         lhs_pos < lhs_indptr_ptr[row + 1]; ++lhs_pos) {
      const int rhs_row = static_cast<int>(lhs_indices_ptr[lhs_pos]);
      const auto lhs_value = lhs_data_ptr[lhs_pos];
      for (RhsI rhs_pos = rhs_indptr_ptr[rhs_row];
           rhs_pos < rhs_indptr_ptr[rhs_row + 1]; ++rhs_pos) {
        const int col = static_cast<int>(rhs_indices_ptr[rhs_pos]);
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
      const auto col_index = static_cast<size_t>(col);
      candidate_indices[static_cast<size_t>(write)] = static_cast<OutI>(col);
      candidate_data[static_cast<size_t>(write)] =
          Accumulator<T>::cast(accum[col_index]);
      ++write;
    }
  }

  std::vector<OutI> out_counts(static_cast<size_t>(lhs_n_rows), OutI{0});
  for (int row = 0; row < lhs_n_rows; ++row) {
    OutI count = OutI{0};
    for (OutI p = candidate_indptr[static_cast<size_t>(row)];
         p < candidate_indptr[static_cast<size_t>(row) + 1]; ++p) {
      if (candidate_data[static_cast<size_t>(p)] != T{}) {
        count += OutI{1};
      }
    }
    out_counts[static_cast<size_t>(row)] = count;
  }

  std::vector<OutI> out_indptr;
  const int out_nnz = prefix_counts(out_counts, out_indptr);
  std::vector<T> out_data;
  std::vector<OutI> out_indices;
  out_data.reserve(static_cast<size_t>(out_nnz));
  out_indices.reserve(static_cast<size_t>(out_nnz));
  for (int row = 0; row < lhs_n_rows; ++row) {
    for (OutI p = candidate_indptr[static_cast<size_t>(row)];
         p < candidate_indptr[static_cast<size_t>(row) + 1]; ++p) {
      const auto value = candidate_data[static_cast<size_t>(p)];
      if (value != T{}) {
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
dispatch_host_out(mx::array lhs_data, mx::array lhs_indices,
                  mx::array lhs_indptr, mx::array rhs_data,
                  mx::array rhs_indices, mx::array rhs_indptr,
                  int lhs_n_rows, int rhs_n_rows, int rhs_n_cols,
                  mx::Dtype out_index_dtype) {
  if (out_index_dtype == mx::int32) {
    return csr_matmat_host_impl<T, LhsI, RhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return csr_matmat_host_impl<T, LhsI, RhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
}

template <typename T, typename LhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_host_rhs(mx::array lhs_data, mx::array lhs_indices,
                  mx::array lhs_indptr, mx::array rhs_data,
                  mx::array rhs_indices, mx::array rhs_indptr,
                  int lhs_n_rows, int rhs_n_rows, int rhs_n_cols,
                  mx::Dtype out_index_dtype) {
  if (rhs_indices.dtype() == mx::int32) {
    return dispatch_host_out<T, LhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return dispatch_host_out<T, LhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
}

template <typename T>
std::tuple<mx::array, mx::array, mx::array>
dispatch_host_lhs(mx::array lhs_data, mx::array lhs_indices,
                  mx::array lhs_indptr, mx::array rhs_data,
                  mx::array rhs_indices, mx::array rhs_indptr,
                  int lhs_n_rows, int rhs_n_rows, int rhs_n_cols,
                  mx::Dtype out_index_dtype) {
  if (lhs_indices.dtype() == mx::int32) {
    return dispatch_host_rhs<T, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
  }
  return dispatch_host_rhs<T, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
}

mx::array symbolic_counts(const mx::array &lhs_indices,
                          const mx::array &lhs_indptr,
                          const mx::array &rhs_indices,
                          const mx::array &rhs_indptr, int lhs_n_rows,
                          int rhs_n_rows, int rhs_n_cols,
                          mx::Dtype out_index_dtype, mx::Stream stream) {
  auto primitive = std::make_shared<CSRMatmatSymbolic>(
      stream, lhs_n_rows, rhs_n_rows, rhs_n_cols);
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
  auto primitive = std::make_shared<CSRMatmatNumeric>(
      stream, lhs_n_rows, rhs_n_rows, rhs_n_cols);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}},
      {lhs_data.dtype(), out_index_dtype}, primitive,
      {lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
       out_indptr});
  return {outputs[0], outputs[1]};
}

mx::array prune_counts(const mx::array &data, const mx::array &indptr,
                       int n_rows, mx::Stream stream) {
  auto primitive = std::make_shared<CSRMatmatPruneCounts>(stream);
  return mx::array(mx::Shape{n_rows}, indptr.dtype(), primitive, {data, indptr});
}

std::tuple<mx::array, mx::array>
prune_fill(const mx::array &data, const mx::array &indices,
           const mx::array &indptr, const mx::array &out_indptr, int out_nnz,
           mx::Stream stream) {
  auto primitive = std::make_shared<CSRMatmatPruneFill>(stream);
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
                           int lhs_n_rows, int rhs_n_rows, int rhs_n_cols,
                           mx::Stream stream) {
  if (counts.dtype() == mx::int32) {
    symbolic_cpu_impl<LhsI, RhsI, int32_t>(
        lhs_indices, lhs_indptr, rhs_indices, rhs_indptr, counts, lhs_n_rows,
        rhs_n_rows, rhs_n_cols, stream);
    return;
  }
  symbolic_cpu_impl<LhsI, RhsI, int64_t>(
      lhs_indices, lhs_indptr, rhs_indices, rhs_indptr, counts, lhs_n_rows,
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
    dispatch_symbolic_out<LhsI, int32_t>(
        lhs_indices, lhs_indptr, rhs_indices, rhs_indptr, counts, lhs_n_rows,
        rhs_n_rows, rhs_n_cols, stream);
    return;
  }
  dispatch_symbolic_out<LhsI, int64_t>(
      lhs_indices, lhs_indptr, rhs_indices, rhs_indptr, counts, lhs_n_rows,
      rhs_n_rows, rhs_n_cols, stream);
}

template <typename T, typename LhsI, typename RhsI>
void dispatch_numeric_out(const mx::array &lhs_data,
                          const mx::array &lhs_indices,
                          const mx::array &lhs_indptr,
                          const mx::array &rhs_data,
                          const mx::array &rhs_indices,
                          const mx::array &rhs_indptr,
                          const mx::array &out_indptr, mx::array &out_data,
                          mx::array &out_indices, int lhs_n_rows,
                          int rhs_n_rows, int rhs_n_cols, mx::Stream stream) {
  if (out_indptr.dtype() == mx::int32) {
    numeric_cpu_impl<T, LhsI, RhsI, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        out_indptr, out_data, out_indices, lhs_n_rows, rhs_n_rows, rhs_n_cols,
        stream);
    return;
  }
  numeric_cpu_impl<T, LhsI, RhsI, int64_t>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_indices, lhs_n_rows, rhs_n_rows, rhs_n_cols,
      stream);
}

template <typename T, typename LhsI>
void dispatch_numeric_rhs(const mx::array &lhs_data,
                          const mx::array &lhs_indices,
                          const mx::array &lhs_indptr,
                          const mx::array &rhs_data,
                          const mx::array &rhs_indices,
                          const mx::array &rhs_indptr,
                          const mx::array &out_indptr, mx::array &out_data,
                          mx::array &out_indices, int lhs_n_rows,
                          int rhs_n_rows, int rhs_n_cols, mx::Stream stream) {
  if (rhs_indices.dtype() == mx::int32) {
    dispatch_numeric_out<T, LhsI, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        out_indptr, out_data, out_indices, lhs_n_rows, rhs_n_rows, rhs_n_cols,
        stream);
    return;
  }
  dispatch_numeric_out<T, LhsI, int64_t>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_indices, lhs_n_rows, rhs_n_rows, rhs_n_cols,
      stream);
}

template <typename T>
void dispatch_numeric_lhs(const mx::array &lhs_data,
                          const mx::array &lhs_indices,
                          const mx::array &lhs_indptr,
                          const mx::array &rhs_data,
                          const mx::array &rhs_indices,
                          const mx::array &rhs_indptr,
                          const mx::array &out_indptr, mx::array &out_data,
                          mx::array &out_indices, int lhs_n_rows,
                          int rhs_n_rows, int rhs_n_cols, mx::Stream stream) {
  if (lhs_indices.dtype() == mx::int32) {
    dispatch_numeric_rhs<T, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        out_indptr, out_data, out_indices, lhs_n_rows, rhs_n_rows, rhs_n_cols,
        stream);
    return;
  }
  dispatch_numeric_rhs<T, int64_t>(
      lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
      out_indptr, out_data, out_indices, lhs_n_rows, rhs_n_rows, rhs_n_cols,
      stream);
}

template <typename T>
void dispatch_prune_index(const mx::array &data, const mx::array &indices,
                          const mx::array &indptr,
                          const mx::array &out_indptr, mx::array &out_data,
                          mx::array &out_indices, mx::Stream stream) {
  if (indices.dtype() == mx::int32) {
    prune_fill_cpu_impl<T, int32_t>(data, indices, indptr, out_indptr, out_data,
                                    out_indices, stream);
    return;
  }
  prune_fill_cpu_impl<T, int64_t>(data, indices, indptr, out_indptr, out_data,
                                  out_indices, stream);
}

} // namespace

void CSRMatmatSymbolic::eval_cpu(const std::vector<mx::array> &inputs,
                                 std::vector<mx::array> &outputs) {
  const auto &lhs_indices = inputs[0];
  const auto &lhs_indptr = inputs[1];
  const auto &rhs_indices = inputs[2];
  const auto &rhs_indptr = inputs[3];
  auto &counts = outputs[0];

  if (lhs_indices.dtype() == mx::int32) {
    dispatch_symbolic_rhs<int32_t>(lhs_indices, lhs_indptr, rhs_indices,
                                   rhs_indptr, counts, lhs_n_rows_,
                                   rhs_n_rows_, rhs_n_cols_, stream());
    return;
  }
  dispatch_symbolic_rhs<int64_t>(lhs_indices, lhs_indptr, rhs_indices,
                                 rhs_indptr, counts, lhs_n_rows_, rhs_n_rows_,
                                 rhs_n_cols_, stream());
}

#ifdef _METAL_
void CSRMatmatSymbolic::eval_gpu(const std::vector<mx::array> &inputs,
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
      matmat_index_kernel_name("csr_matmat_symbolic", lhs_indices.dtype(),
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
void CSRMatmatSymbolic::eval_gpu(const std::vector<mx::array> &,
                                 std::vector<mx::array> &) {
  throw std::runtime_error("csr_matmat has no GPU implementation in this build.");
}
#endif

void CSRMatmatNumeric::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &lhs_indices = inputs[1];
  const auto &lhs_indptr = inputs[2];
  const auto &rhs_data = inputs[3];
  const auto &rhs_indices = inputs[4];
  const auto &rhs_indptr = inputs[5];
  const auto &out_indptr = inputs[6];

#define DISPATCH_CSR_MATMAT_NUMERIC_VALUE(DTYPE, TYPE)                         \
  if (lhs_data.dtype() == DTYPE) {                                             \
    dispatch_numeric_lhs<TYPE>(lhs_data, lhs_indices, lhs_indptr, rhs_data,    \
                               rhs_indices, rhs_indptr, out_indptr, outputs[0],\
                               outputs[1], lhs_n_rows_, rhs_n_rows_,           \
                               rhs_n_cols_, stream());                         \
    return;                                                                    \
  }

  DISPATCH_CSR_MATMAT_NUMERIC_VALUE(mx::float32, float)
  DISPATCH_CSR_MATMAT_NUMERIC_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_MATMAT_NUMERIC_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_MATMAT_NUMERIC_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_MATMAT_NUMERIC_VALUE

  throw std::runtime_error("csr_matmat unsupported value dtype.");
}

#ifdef _METAL_
void CSRMatmatNumeric::eval_gpu(const std::vector<mx::array> &inputs,
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
      matmat_numeric_kernel_name(lhs_data.dtype(), lhs_indices.dtype(),
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
  encoder.set_bytes(rhs_n_rows_, 10);
  encoder.set_bytes(rhs_n_cols_, 11);

  auto threads = std::max<size_t>(static_cast<size_t>(lhs_n_rows_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRMatmatNumeric::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error("csr_matmat has no GPU implementation in this build.");
}
#endif

void CSRMatmatPruneCounts::eval_cpu(const std::vector<mx::array> &inputs,
                                    std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indptr = inputs[1];

#define DISPATCH_CSR_MATMAT_PRUNE_COUNTS(DTYPE, TYPE)                          \
  if (data.dtype() == DTYPE) {                                                 \
    if (indptr.dtype() == mx::int32) {                                         \
      prune_counts_cpu_impl<TYPE, int32_t>(data, indptr, outputs[0], stream());\
    } else {                                                                   \
      prune_counts_cpu_impl<TYPE, int64_t>(data, indptr, outputs[0], stream());\
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_MATMAT_PRUNE_COUNTS(mx::float32, float)
  DISPATCH_CSR_MATMAT_PRUNE_COUNTS(mx::float16, mx::float16_t)
  DISPATCH_CSR_MATMAT_PRUNE_COUNTS(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_MATMAT_PRUNE_COUNTS(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_MATMAT_PRUNE_COUNTS

  throw std::runtime_error("csr_matmat unsupported value dtype.");
}

#ifdef _METAL_
void CSRMatmatPruneCounts::eval_gpu(const std::vector<mx::array> &inputs,
                                    std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indptr = inputs[1];
  auto &counts = outputs[0];

  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      sparse_kernel_name("csr_matmat_prune_counts", data.dtype(), indptr.dtype()),
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
void CSRMatmatPruneCounts::eval_gpu(const std::vector<mx::array> &,
                                    std::vector<mx::array> &) {
  throw std::runtime_error("csr_matmat has no GPU implementation in this build.");
}
#endif

void CSRMatmatPruneFill::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indices = inputs[1];
  const auto &indptr = inputs[2];
  const auto &out_indptr = inputs[3];

#define DISPATCH_CSR_MATMAT_PRUNE_FILL(DTYPE, TYPE)                            \
  if (data.dtype() == DTYPE) {                                                 \
    dispatch_prune_index<TYPE>(data, indices, indptr, out_indptr, outputs[0],  \
                               outputs[1], stream());                          \
    return;                                                                    \
  }

  DISPATCH_CSR_MATMAT_PRUNE_FILL(mx::float32, float)
  DISPATCH_CSR_MATMAT_PRUNE_FILL(mx::float16, mx::float16_t)
  DISPATCH_CSR_MATMAT_PRUNE_FILL(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_MATMAT_PRUNE_FILL(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_MATMAT_PRUNE_FILL

  throw std::runtime_error("csr_matmat unsupported value dtype.");
}

#ifdef _METAL_
void CSRMatmatPruneFill::eval_gpu(const std::vector<mx::array> &inputs,
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
  auto *kernel = device.get_kernel(
      sparse_kernel_name("csr_matmat_prune_fill", data.dtype(), indices.dtype()),
      lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(out_indptr, 3);
  encoder.set_output_array(out_data, 4);
  encoder.set_output_array(out_indices, 5);
  const int n_rows = static_cast<int>(indptr.size()) - 1;
  encoder.set_bytes(n_rows, 6);

  auto threads = std::max<size_t>(static_cast<size_t>(n_rows), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRMatmatPruneFill::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error("csr_matmat has no GPU implementation in this build.");
}
#endif

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
  if (rhs_n_cols > std::numeric_limits<int>::max()) {
    throw std::overflow_error("csr_matmat n_cols exceeds supported limits.");
  }

  const auto out_index_dtype =
      lhs_indices.dtype() == rhs_indices.dtype() ? lhs_indices.dtype() : mx::int64;
  if (out_index_dtype == mx::int32 &&
      rhs_n_cols > std::numeric_limits<int32_t>::max()) {
    throw std::overflow_error(
        "csr_matmat n_cols exceeds int32 output index capacity.");
  }

  if (!use_experimental_metal_spgemm()) {
    if (lhs_data.dtype() == mx::float32) {
      return dispatch_host_lhs<float>(
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
          lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
    }
    if (lhs_data.dtype() == mx::float16) {
      return dispatch_host_lhs<mx::float16_t>(
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
          lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
    }
    if (lhs_data.dtype() == mx::bfloat16) {
      return dispatch_host_lhs<mx::bfloat16_t>(
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
          lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
    }
    if (lhs_data.dtype() == mx::complex64) {
      return dispatch_host_lhs<mx::complex64_t>(
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
          lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype);
    }
    throw std::runtime_error("csr_matmat unsupported value dtype.");
  }

  auto stream = use_experimental_metal_spgemm()
                    ? mx::default_stream(mx::default_device())
                    : mx::default_stream(mx::Device(mx::Device::cpu, 0));
  auto lhs_data_contig = mx::contiguous(lhs_data, false, stream);
  auto lhs_indices_contig = mx::contiguous(lhs_indices, false, stream);
  auto lhs_indptr_contig = mx::contiguous(lhs_indptr, false, stream);
  auto rhs_data_contig = mx::contiguous(rhs_data, false, stream);
  auto rhs_indices_contig = mx::contiguous(rhs_indices, false, stream);
  auto rhs_indptr_contig = mx::contiguous(rhs_indptr, false, stream);

  auto counts =
      symbolic_counts(lhs_indices_contig, lhs_indptr_contig, rhs_indices_contig,
                      rhs_indptr_contig, lhs_n_rows, rhs_n_rows, rhs_n_cols,
                      out_index_dtype, stream);
  mx::eval(counts);
  auto [candidate_indptr, candidate_nnz] =
      build_indptr_from_counts(counts, lhs_n_rows, out_index_dtype);

  auto [candidate_data, candidate_indices] = numeric_fill(
      lhs_data_contig, lhs_indices_contig, lhs_indptr_contig, rhs_data_contig,
      rhs_indices_contig, rhs_indptr_contig, candidate_indptr, candidate_nnz,
      lhs_n_rows, rhs_n_rows, rhs_n_cols, out_index_dtype, stream);

  auto nonzero_counts =
      prune_counts(candidate_data, candidate_indptr, lhs_n_rows, stream);
  mx::eval(nonzero_counts);
  auto [out_indptr, out_nnz] =
      build_indptr_from_counts(nonzero_counts, lhs_n_rows, out_index_dtype);
  auto [out_data, out_indices] = prune_fill(
      candidate_data, candidate_indices, candidate_indptr, out_indptr, out_nnz,
      stream);
  return {out_data, out_indices, out_indptr};
}

} // namespace mlx_sparse
