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

#include "sparse/add/csr_add.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/common.h"
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

template <typename T> typename Accumulator<T>::Type accumulator_value(T value) {
  using AccT = typename Accumulator<T>::Type;
  if constexpr (std::is_same_v<T, mx::float16_t> ||
                std::is_same_v<T, mx::bfloat16_t>) {
    return static_cast<float>(value);
  } else {
    return static_cast<AccT>(value);
  }
}

template <typename T> bool is_nonzero(T value) { return !(value == T{}); }

std::string add_index_kernel_name(const std::string &prefix,
                                  mx::Dtype value_dtype,
                                  mx::Dtype lhs_index_dtype,
                                  mx::Dtype rhs_index_dtype,
                                  mx::Dtype out_index_dtype) {
  return prefix + "_" + value_kernel_suffix(value_dtype) + "_" +
         index_kernel_suffix(lhs_index_dtype) + "_" +
         index_kernel_suffix(rhs_index_dtype) + "_" +
         index_kernel_suffix(out_index_dtype);
}

bool should_use_metal_sparse_add() {
#ifdef _METAL_
  return mx::default_device().type == mx::Device::gpu;
#else
  return false;
#endif
}

template <typename I>
void check_csr_structure(const I *indices, const I *indptr, int n_rows,
                         int n_cols, size_t nnz, const char *name) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr_add shape dimensions must be non-negative.");
  }
  if (indptr[0] != I{0}) {
    throw std::invalid_argument(std::string(name) + " indptr must start at 0.");
  }
  if (static_cast<size_t>(indptr[n_rows]) != nnz) {
    throw std::invalid_argument(std::string(name) +
                                " indptr[-1] must equal data size.");
  }
  for (int row = 0; row < n_rows; ++row) {
    if (indptr[row] > indptr[row + 1]) {
      throw std::invalid_argument(std::string(name) +
                                  " indptr must be nondecreasing.");
    }
    I previous = I{};
    bool have_previous = false;
    for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
      const I col = indices[p];
      if (col < I{0} || col >= static_cast<I>(n_cols)) {
        throw std::invalid_argument(
            std::string(name) + " indices contain an out-of-bounds column.");
      }
      if (have_previous && col < previous) {
        throw std::invalid_argument(std::string(name) +
                                    " indices must be sorted within each row.");
      }
      previous = col;
      have_previous = true;
    }
  }
}

template <typename T, typename LhsI, typename RhsI>
size_t merged_row_count(const T *lhs_data, const LhsI *lhs_indices,
                        LhsI lhs_pos, LhsI lhs_end, const T *rhs_data,
                        const RhsI *rhs_indices, RhsI rhs_pos, RhsI rhs_end,
                        bool subtract) {
  using AccT = typename Accumulator<T>::Type;
  size_t count = 0;
  while (lhs_pos < lhs_end || rhs_pos < rhs_end) {
    int64_t col = 0;
    if (rhs_pos >= rhs_end ||
        (lhs_pos < lhs_end && lhs_indices[lhs_pos] < rhs_indices[rhs_pos])) {
      col = static_cast<int64_t>(lhs_indices[lhs_pos]);
    } else if (lhs_pos >= lhs_end ||
               rhs_indices[rhs_pos] < lhs_indices[lhs_pos]) {
      col = static_cast<int64_t>(rhs_indices[rhs_pos]);
    } else {
      col = static_cast<int64_t>(lhs_indices[lhs_pos]);
    }

    AccT acc = Accumulator<T>::zero();
    while (lhs_pos < lhs_end &&
           static_cast<int64_t>(lhs_indices[lhs_pos]) == col) {
      acc += accumulator_value<T>(lhs_data[lhs_pos]);
      ++lhs_pos;
    }
    while (rhs_pos < rhs_end &&
           static_cast<int64_t>(rhs_indices[rhs_pos]) == col) {
      const AccT value = accumulator_value<T>(rhs_data[rhs_pos]);
      acc = subtract ? acc - value : acc + value;
      ++rhs_pos;
    }
    if (is_nonzero(Accumulator<T>::cast(acc))) {
      ++count;
    }
  }
  return count;
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
OutI fill_merged_row(const T *lhs_data, const LhsI *lhs_indices, LhsI lhs_pos,
                     LhsI lhs_end, const T *rhs_data, const RhsI *rhs_indices,
                     RhsI rhs_pos, RhsI rhs_end, bool subtract, T *out_data,
                     OutI *out_indices, OutI write) {
  using AccT = typename Accumulator<T>::Type;
  while (lhs_pos < lhs_end || rhs_pos < rhs_end) {
    int64_t col = 0;
    if (rhs_pos >= rhs_end ||
        (lhs_pos < lhs_end && lhs_indices[lhs_pos] < rhs_indices[rhs_pos])) {
      col = static_cast<int64_t>(lhs_indices[lhs_pos]);
    } else if (lhs_pos >= lhs_end ||
               rhs_indices[rhs_pos] < lhs_indices[lhs_pos]) {
      col = static_cast<int64_t>(rhs_indices[rhs_pos]);
    } else {
      col = static_cast<int64_t>(lhs_indices[lhs_pos]);
    }

    AccT acc = Accumulator<T>::zero();
    while (lhs_pos < lhs_end &&
           static_cast<int64_t>(lhs_indices[lhs_pos]) == col) {
      acc += accumulator_value<T>(lhs_data[lhs_pos]);
      ++lhs_pos;
    }
    while (rhs_pos < rhs_end &&
           static_cast<int64_t>(rhs_indices[rhs_pos]) == col) {
      const AccT value = accumulator_value<T>(rhs_data[rhs_pos]);
      acc = subtract ? acc - value : acc + value;
      ++rhs_pos;
    }

    const T value = Accumulator<T>::cast(acc);
    if (is_nonzero(value)) {
      out_indices[write] = static_cast<OutI>(col);
      out_data[write] = value;
      ++write;
    }
  }
  return write;
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

template <typename T, typename LhsI, typename RhsI, typename OutI>
std::tuple<mx::array, mx::array, mx::array>
csr_add_host_impl(mx::array lhs_data, mx::array lhs_indices,
                  mx::array lhs_indptr, mx::array rhs_data,
                  mx::array rhs_indices, mx::array rhs_indptr, int n_rows,
                  int n_cols, bool subtract, mx::Dtype out_index_dtype,
                  int requested_workers) {
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

  check_csr_structure(lhs_indices_ptr, lhs_indptr_ptr, n_rows, n_cols,
                      lhs_data.size(), "csr_add lhs");
  check_csr_structure(rhs_indices_ptr, rhs_indptr_ptr, n_rows, n_cols,
                      rhs_data.size(), "csr_add rhs");

  std::vector<int64_t> row_work(static_cast<size_t>(n_rows), 0);
  for (int row = 0; row < n_rows; ++row) {
    const int64_t lhs_nnz =
        static_cast<int64_t>(lhs_indptr_ptr[row + 1] - lhs_indptr_ptr[row]);
    const int64_t rhs_nnz =
        static_cast<int64_t>(rhs_indptr_ptr[row + 1] - rhs_indptr_ptr[row]);
    row_work[static_cast<size_t>(row)] =
        std::max<int64_t>(0, lhs_nnz + rhs_nnz);
  }

  auto ranges = cpu_ranges_for_output_work(row_work, requested_workers);
  if (ranges.empty()) {
    ranges.push_back({0, n_rows});
  }

  std::vector<OutI> row_counts(static_cast<size_t>(n_rows), OutI{0});
  parallel_for_cpu_ranges(ranges, [&](CpuRange range) {
    for (int row = range.begin; row < range.end; ++row) {
      const auto count = merged_row_count<T, LhsI, RhsI>(
          lhs_data_ptr, lhs_indices_ptr, lhs_indptr_ptr[row],
          lhs_indptr_ptr[row + 1], rhs_data_ptr, rhs_indices_ptr,
          rhs_indptr_ptr[row], rhs_indptr_ptr[row + 1], subtract);
      check_output_nnz<OutI>(count, "csr_add row");
      row_counts[static_cast<size_t>(row)] = static_cast<OutI>(count);
    }
  });

  std::vector<OutI> out_indptr(static_cast<size_t>(n_rows) + 1, OutI{0});
  size_t total_nnz = 0;
  for (int row = 0; row < n_rows; ++row) {
    total_nnz += static_cast<size_t>(row_counts[static_cast<size_t>(row)]);
    check_output_nnz<OutI>(total_nnz, "csr_add");
    out_indptr[static_cast<size_t>(row) + 1] = static_cast<OutI>(total_nnz);
  }

  std::vector<T> out_data(total_nnz);
  std::vector<OutI> out_indices(total_nnz);
  parallel_for_cpu_ranges(ranges, [&](CpuRange range) {
    for (int row = range.begin; row < range.end; ++row) {
      auto write = out_indptr[static_cast<size_t>(row)];
      const auto end = fill_merged_row<T, LhsI, RhsI, OutI>(
          lhs_data_ptr, lhs_indices_ptr, lhs_indptr_ptr[row],
          lhs_indptr_ptr[row + 1], rhs_data_ptr, rhs_indices_ptr,
          rhs_indptr_ptr[row], rhs_indptr_ptr[row + 1], subtract,
          out_data.data(), out_indices.data(), write);
      if (end != out_indptr[static_cast<size_t>(row) + 1]) {
        throw std::runtime_error("csr_add internal count/fill mismatch.");
      }
    }
  });

  const int out_nnz = static_cast<int>(total_nnz);
  return {mx::array(out_data.begin(), mx::Shape{out_nnz}, lhs_data.dtype()),
          mx::array(out_indices.begin(), mx::Shape{out_nnz}, out_index_dtype),
          mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    out_index_dtype)};
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
      throw std::runtime_error("csr_add produced a negative per-row count.");
    }
    total += count;
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error("csr_add output nnz exceeds MLX shape limits.");
    }
    if (total > static_cast<int64_t>(std::numeric_limits<I>::max())) {
      throw std::overflow_error(
          "csr_add output nnz exceeds index dtype capacity.");
    }
    out_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(total);
  }

  return {mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    index_dtype),
          static_cast<int>(total)};
}

std::pair<mx::array, int> build_indptr_from_counts(const mx::array &counts,
                                                   int n_rows) {
  if (counts.dtype() == mx::int32) {
    return build_indptr_from_counts<int32_t>(counts, n_rows, mx::int32);
  }
  return build_indptr_from_counts<int64_t>(counts, n_rows, mx::int64);
}

class CSRAddCounts : public mx::Primitive {
public:
  CSRAddCounts(mx::Stream stream, int n_rows, bool subtract)
      : Primitive(stream), n_rows_(n_rows), subtract_(subtract) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRAddCounts"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRAddCounts &>(other);
    return n_rows_ == rhs.n_rows_ && subtract_ == rhs.subtract_;
  }

private:
  int n_rows_;
  bool subtract_;
};

class CSRAddFill : public mx::Primitive {
public:
  CSRAddFill(mx::Stream stream, int n_rows, bool subtract)
      : Primitive(stream), n_rows_(n_rows), subtract_(subtract) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRAddFill"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRAddFill &>(other);
    return n_rows_ == rhs.n_rows_ && subtract_ == rhs.subtract_;
  }

private:
  int n_rows_;
  bool subtract_;
};

template <typename T, typename LhsI, typename RhsI, typename OutI>
void add_counts_cpu_impl(const mx::array &lhs_data,
                         const mx::array &lhs_indices,
                         const mx::array &lhs_indptr, const mx::array &rhs_data,
                         const mx::array &rhs_indices,
                         const mx::array &rhs_indptr, mx::array &counts,
                         int n_rows, bool subtract, mx::Stream stream) {
  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(lhs_data);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(lhs_indptr);
  encoder.set_input_array(rhs_data);
  encoder.set_input_array(rhs_indices);
  encoder.set_input_array(rhs_indptr);
  encoder.set_output_array(counts);

  encoder.dispatch([lhs_data = mx::array::unsafe_weak_copy(lhs_data),
                    lhs_indices = mx::array::unsafe_weak_copy(lhs_indices),
                    lhs_indptr = mx::array::unsafe_weak_copy(lhs_indptr),
                    rhs_data = mx::array::unsafe_weak_copy(rhs_data),
                    rhs_indices = mx::array::unsafe_weak_copy(rhs_indices),
                    rhs_indptr = mx::array::unsafe_weak_copy(rhs_indptr),
                    counts = mx::array::unsafe_weak_copy(counts), n_rows,
                    subtract]() mutable {
    const auto *lhs_data_ptr = lhs_data.data<T>();
    const auto *lhs_indices_ptr = lhs_indices.data<LhsI>();
    const auto *lhs_indptr_ptr = lhs_indptr.data<LhsI>();
    const auto *rhs_data_ptr = rhs_data.data<T>();
    const auto *rhs_indices_ptr = rhs_indices.data<RhsI>();
    const auto *rhs_indptr_ptr = rhs_indptr.data<RhsI>();
    auto *counts_ptr = counts.data<OutI>();

    auto count_rows = [&](CpuRange range) {
      for (int row = range.begin; row < range.end; ++row) {
        const auto count = merged_row_count<T, LhsI, RhsI>(
            lhs_data_ptr, lhs_indices_ptr, lhs_indptr_ptr[row],
            lhs_indptr_ptr[row + 1], rhs_data_ptr, rhs_indices_ptr,
            rhs_indptr_ptr[row], rhs_indptr_ptr[row + 1], subtract);
        counts_ptr[row] = static_cast<OutI>(count);
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_rows <= 0) {
      count_rows({0, n_rows});
      return;
    }
    const auto ranges = equal_cpu_ranges(n_rows, workers);
    parallel_for_cpu_ranges(ranges, count_rows);
  });
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
void add_fill_cpu_impl(const mx::array &lhs_data, const mx::array &lhs_indices,
                       const mx::array &lhs_indptr, const mx::array &rhs_data,
                       const mx::array &rhs_indices,
                       const mx::array &rhs_indptr, const mx::array &out_indptr,
                       mx::array &out_data, mx::array &out_indices, int n_rows,
                       bool subtract, mx::Stream stream) {
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
                    n_rows, subtract]() mutable {
    const auto *lhs_data_ptr = lhs_data.data<T>();
    const auto *lhs_indices_ptr = lhs_indices.data<LhsI>();
    const auto *lhs_indptr_ptr = lhs_indptr.data<LhsI>();
    const auto *rhs_data_ptr = rhs_data.data<T>();
    const auto *rhs_indices_ptr = rhs_indices.data<RhsI>();
    const auto *rhs_indptr_ptr = rhs_indptr.data<RhsI>();
    const auto *out_indptr_ptr = out_indptr.data<OutI>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_indices_ptr = out_indices.data<OutI>();

    auto fill_rows = [&](CpuRange range) {
      for (int row = range.begin; row < range.end; ++row) {
        fill_merged_row<T, LhsI, RhsI, OutI>(
            lhs_data_ptr, lhs_indices_ptr, lhs_indptr_ptr[row],
            lhs_indptr_ptr[row + 1], rhs_data_ptr, rhs_indices_ptr,
            rhs_indptr_ptr[row], rhs_indptr_ptr[row + 1], subtract,
            out_data_ptr, out_indices_ptr, out_indptr_ptr[row]);
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_rows <= 0) {
      fill_rows({0, n_rows});
      return;
    }
    const auto ranges = equal_cpu_ranges(n_rows, workers);
    parallel_for_cpu_ranges(ranges, fill_rows);
  });
}

mx::array add_counts(const mx::array &lhs_data, const mx::array &lhs_indices,
                     const mx::array &lhs_indptr, const mx::array &rhs_data,
                     const mx::array &rhs_indices, const mx::array &rhs_indptr,
                     int n_rows, bool subtract, mx::Dtype out_index_dtype,
                     mx::Stream stream) {
  auto primitive = std::make_shared<CSRAddCounts>(stream, n_rows, subtract);
  return mx::array(
      mx::Shape{n_rows}, out_index_dtype, primitive,
      {lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr});
}

std::tuple<mx::array, mx::array>
add_fill(const mx::array &lhs_data, const mx::array &lhs_indices,
         const mx::array &lhs_indptr, const mx::array &rhs_data,
         const mx::array &rhs_indices, const mx::array &rhs_indptr,
         const mx::array &out_indptr, int out_nnz, bool subtract,
         mx::Stream stream) {
  auto primitive = std::make_shared<CSRAddFill>(
      stream, static_cast<int>(out_indptr.size()) - 1, subtract);
  auto outputs =
      mx::array::make_arrays({mx::Shape{out_nnz}, mx::Shape{out_nnz}},
                             {lhs_data.dtype(), out_indptr.dtype()}, primitive,
                             {lhs_data, lhs_indices, lhs_indptr, rhs_data,
                              rhs_indices, rhs_indptr, out_indptr});
  return {outputs[0], outputs[1]};
}

template <typename T, typename LhsI, typename RhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_host_out(mx::array lhs_data, mx::array lhs_indices,
                  mx::array lhs_indptr, mx::array rhs_data,
                  mx::array rhs_indices, mx::array rhs_indptr, int n_rows,
                  int n_cols, bool subtract, mx::Dtype out_index_dtype,
                  int requested_workers) {
  if (out_index_dtype == mx::int32) {
    return csr_add_host_impl<T, LhsI, RhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        n_rows, n_cols, subtract, out_index_dtype, requested_workers);
  }
  return csr_add_host_impl<T, LhsI, RhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      n_rows, n_cols, subtract, out_index_dtype, requested_workers);
}

template <typename T, typename LhsI>
std::tuple<mx::array, mx::array, mx::array>
dispatch_host_rhs(mx::array lhs_data, mx::array lhs_indices,
                  mx::array lhs_indptr, mx::array rhs_data,
                  mx::array rhs_indices, mx::array rhs_indptr, int n_rows,
                  int n_cols, bool subtract, mx::Dtype out_index_dtype,
                  int requested_workers) {
  if (rhs_indices.dtype() == mx::int32) {
    return dispatch_host_out<T, LhsI, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        n_rows, n_cols, subtract, out_index_dtype, requested_workers);
  }
  return dispatch_host_out<T, LhsI, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      n_rows, n_cols, subtract, out_index_dtype, requested_workers);
}

template <typename T>
std::tuple<mx::array, mx::array, mx::array>
dispatch_host_lhs(mx::array lhs_data, mx::array lhs_indices,
                  mx::array lhs_indptr, mx::array rhs_data,
                  mx::array rhs_indices, mx::array rhs_indptr, int n_rows,
                  int n_cols, bool subtract, mx::Dtype out_index_dtype,
                  int requested_workers) {
  if (lhs_indices.dtype() == mx::int32) {
    return dispatch_host_rhs<T, int32_t>(
        std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
        std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
        n_rows, n_cols, subtract, out_index_dtype, requested_workers);
  }
  return dispatch_host_rhs<T, int64_t>(
      std::move(lhs_data), std::move(lhs_indices), std::move(lhs_indptr),
      std::move(rhs_data), std::move(rhs_indices), std::move(rhs_indptr),
      n_rows, n_cols, subtract, out_index_dtype, requested_workers);
}

} // namespace

void CSRAddCounts::eval_cpu(const std::vector<mx::array> &inputs,
                            std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &lhs_indices = inputs[1];
  const auto &lhs_indptr = inputs[2];
  const auto &rhs_data = inputs[3];
  const auto &rhs_indices = inputs[4];
  const auto &rhs_indptr = inputs[5];
  auto &counts = outputs[0];

#define DISPATCH_CSR_ADD_COUNTS_VALUE(DTYPE, TYPE)                             \
  if (lhs_data.dtype() == DTYPE) {                                             \
    if (lhs_indices.dtype() == mx::int32 &&                                    \
        rhs_indices.dtype() == mx::int32 && counts.dtype() == mx::int32) {     \
      add_counts_cpu_impl<TYPE, int32_t, int32_t, int32_t>(                    \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, counts, n_rows_, subtract_, stream());                   \
    } else if (lhs_indices.dtype() == mx::int32 &&                             \
               rhs_indices.dtype() == mx::int32) {                             \
      add_counts_cpu_impl<TYPE, int32_t, int32_t, int64_t>(                    \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, counts, n_rows_, subtract_, stream());                   \
    } else if (lhs_indices.dtype() == mx::int32 &&                             \
               rhs_indices.dtype() == mx::int64) {                             \
      add_counts_cpu_impl<TYPE, int32_t, int64_t, int64_t>(                    \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, counts, n_rows_, subtract_, stream());                   \
    } else if (lhs_indices.dtype() == mx::int64 &&                             \
               rhs_indices.dtype() == mx::int32) {                             \
      add_counts_cpu_impl<TYPE, int64_t, int32_t, int64_t>(                    \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, counts, n_rows_, subtract_, stream());                   \
    } else {                                                                   \
      add_counts_cpu_impl<TYPE, int64_t, int64_t, int64_t>(                    \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, counts, n_rows_, subtract_, stream());                   \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_ADD_COUNTS_VALUE(mx::float32, float)
  DISPATCH_CSR_ADD_COUNTS_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_ADD_COUNTS_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_ADD_COUNTS_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_ADD_COUNTS_VALUE

  throw std::runtime_error("csr_add unsupported value dtype.");
}

#ifdef _METAL_
void CSRAddCounts::eval_gpu(const std::vector<mx::array> &inputs,
                            std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &lhs_indices = inputs[1];
  const auto &lhs_indptr = inputs[2];
  const auto &rhs_data = inputs[3];
  const auto &rhs_indices = inputs[4];
  const auto &rhs_indptr = inputs[5];
  auto &counts = outputs[0];

  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      add_index_kernel_name("csr_add_counts", lhs_data.dtype(),
                            lhs_indices.dtype(), rhs_indices.dtype(),
                            counts.dtype()),
      lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(lhs_data, 0);
  encoder.set_input_array(lhs_indices, 1);
  encoder.set_input_array(lhs_indptr, 2);
  encoder.set_input_array(rhs_data, 3);
  encoder.set_input_array(rhs_indices, 4);
  encoder.set_input_array(rhs_indptr, 5);
  encoder.set_output_array(counts, 6);
  encoder.set_bytes(n_rows_, 7);
  const int subtract = subtract_ ? 1 : 0;
  encoder.set_bytes(subtract, 8);

  auto threads = std::max<size_t>(static_cast<size_t>(n_rows_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRAddCounts::eval_gpu(const std::vector<mx::array> &,
                            std::vector<mx::array> &) {
  throw std::runtime_error("csr_add has no GPU implementation in this build.");
}
#endif

void CSRAddFill::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &lhs_indices = inputs[1];
  const auto &lhs_indptr = inputs[2];
  const auto &rhs_data = inputs[3];
  const auto &rhs_indices = inputs[4];
  const auto &rhs_indptr = inputs[5];
  const auto &out_indptr = inputs[6];

#define DISPATCH_CSR_ADD_FILL_VALUE(DTYPE, TYPE)                               \
  if (lhs_data.dtype() == DTYPE) {                                             \
    if (lhs_indices.dtype() == mx::int32 &&                                    \
        rhs_indices.dtype() == mx::int32 && out_indptr.dtype() == mx::int32) { \
      add_fill_cpu_impl<TYPE, int32_t, int32_t, int32_t>(                      \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, out_indptr, outputs[0], outputs[1], n_rows_, subtract_,  \
          stream());                                                           \
    } else if (lhs_indices.dtype() == mx::int32 &&                             \
               rhs_indices.dtype() == mx::int32) {                             \
      add_fill_cpu_impl<TYPE, int32_t, int32_t, int64_t>(                      \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, out_indptr, outputs[0], outputs[1], n_rows_, subtract_,  \
          stream());                                                           \
    } else if (lhs_indices.dtype() == mx::int32 &&                             \
               rhs_indices.dtype() == mx::int64) {                             \
      add_fill_cpu_impl<TYPE, int32_t, int64_t, int64_t>(                      \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, out_indptr, outputs[0], outputs[1], n_rows_, subtract_,  \
          stream());                                                           \
    } else if (lhs_indices.dtype() == mx::int64 &&                             \
               rhs_indices.dtype() == mx::int32) {                             \
      add_fill_cpu_impl<TYPE, int64_t, int32_t, int64_t>(                      \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, out_indptr, outputs[0], outputs[1], n_rows_, subtract_,  \
          stream());                                                           \
    } else {                                                                   \
      add_fill_cpu_impl<TYPE, int64_t, int64_t, int64_t>(                      \
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,            \
          rhs_indptr, out_indptr, outputs[0], outputs[1], n_rows_, subtract_,  \
          stream());                                                           \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_ADD_FILL_VALUE(mx::float32, float)
  DISPATCH_CSR_ADD_FILL_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_ADD_FILL_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_ADD_FILL_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_ADD_FILL_VALUE

  throw std::runtime_error("csr_add unsupported value dtype.");
}

#ifdef _METAL_
void CSRAddFill::eval_gpu(const std::vector<mx::array> &inputs,
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
      add_index_kernel_name("csr_add_fill", lhs_data.dtype(),
                            lhs_indices.dtype(), rhs_indices.dtype(),
                            out_indices.dtype()),
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
  encoder.set_bytes(n_rows_, 9);
  const int subtract = subtract_ ? 1 : 0;
  encoder.set_bytes(subtract, 10);

  auto threads = std::max<size_t>(static_cast<size_t>(n_rows_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRAddFill::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error("csr_add has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
csr_add(const mx::array &lhs_data, const mx::array &lhs_indices,
        const mx::array &lhs_indptr, const mx::array &rhs_data,
        const mx::array &rhs_indices, const mx::array &rhs_indptr, int n_rows,
        int n_cols, bool subtract) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr_add shape dimensions must be non-negative.");
  }
  require_rank(lhs_data, 1, "csr_add lhs_data");
  require_rank(lhs_indices, 1, "csr_add lhs_indices");
  require_rank(lhs_indptr, 1, "csr_add lhs_indptr");
  require_rank(rhs_data, 1, "csr_add rhs_data");
  require_rank(rhs_indices, 1, "csr_add rhs_indices");
  require_rank(rhs_indptr, 1, "csr_add rhs_indptr");
  require_same_value_dtype(lhs_data, rhs_data, "csr_add lhs_data",
                           "csr_add rhs_data");
  require_same_index_dtype(lhs_indices, lhs_indptr, "csr_add lhs_indices",
                           "csr_add lhs_indptr");
  require_same_index_dtype(rhs_indices, rhs_indptr, "csr_add rhs_indices",
                           "csr_add rhs_indptr");
  require_size(lhs_indptr, n_rows + 1, "csr_add lhs_indptr");
  require_size(rhs_indptr, n_rows + 1, "csr_add rhs_indptr");
  if (lhs_indices.size() != lhs_data.size() ||
      rhs_indices.size() != rhs_data.size()) {
    throw std::invalid_argument(
        "csr_add data and indices must have equal lengths.");
  }
  if (n_cols > std::numeric_limits<int>::max()) {
    throw std::overflow_error("csr_add n_cols exceeds supported limits.");
  }

  const auto out_index_dtype = lhs_indices.dtype() == rhs_indices.dtype()
                                   ? lhs_indices.dtype()
                                   : mx::int64;
  if (out_index_dtype == mx::int32 &&
      n_cols > std::numeric_limits<int32_t>::max()) {
    throw std::overflow_error(
        "csr_add n_cols exceeds int32 output index capacity.");
  }

  if (!should_use_metal_sparse_add()) {
    const int requested_workers = configured_spgemm_worker_count();
    if (lhs_data.dtype() == mx::float32) {
      return dispatch_host_lhs<float>(
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
          n_rows, n_cols, subtract, out_index_dtype, requested_workers);
    }
    if (lhs_data.dtype() == mx::float16) {
      return dispatch_host_lhs<mx::float16_t>(
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
          n_rows, n_cols, subtract, out_index_dtype, requested_workers);
    }
    if (lhs_data.dtype() == mx::bfloat16) {
      return dispatch_host_lhs<mx::bfloat16_t>(
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
          n_rows, n_cols, subtract, out_index_dtype, requested_workers);
    }
    if (lhs_data.dtype() == mx::complex64) {
      return dispatch_host_lhs<mx::complex64_t>(
          lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
          n_rows, n_cols, subtract, out_index_dtype, requested_workers);
    }
    throw std::runtime_error("csr_add unsupported value dtype.");
  }

  auto stream = mx::default_stream(mx::default_device());
  auto lhs_data_contig = mx::contiguous(lhs_data, false, stream);
  auto lhs_indices_contig = mx::contiguous(lhs_indices, false, stream);
  auto lhs_indptr_contig = mx::contiguous(lhs_indptr, false, stream);
  auto rhs_data_contig = mx::contiguous(rhs_data, false, stream);
  auto rhs_indices_contig = mx::contiguous(rhs_indices, false, stream);
  auto rhs_indptr_contig = mx::contiguous(rhs_indptr, false, stream);

  auto counts =
      add_counts(lhs_data_contig, lhs_indices_contig, lhs_indptr_contig,
                 rhs_data_contig, rhs_indices_contig, rhs_indptr_contig, n_rows,
                 subtract, out_index_dtype, stream);
  mx::eval(counts);
  auto [out_indptr, out_nnz] = build_indptr_from_counts(counts, n_rows);
  auto [out_data, out_indices] =
      add_fill(lhs_data_contig, lhs_indices_contig, lhs_indptr_contig,
               rhs_data_contig, rhs_indices_contig, rhs_indptr_contig,
               out_indptr, out_nnz, subtract, stream);
  return {out_data, out_indices, out_indptr};
}

} // namespace mlx_sparse
