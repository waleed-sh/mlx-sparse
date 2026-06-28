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

#include "sparse/fromdense/fromdense.h"

#include <algorithm>
#include <cmath>
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

template <typename T> bool keep_dense_value(T value, float threshold) {
  if (threshold == 0.0f) {
    return value != T{};
  }
  return std::fabs(static_cast<float>(value)) > threshold;
}

template <>
bool keep_dense_value<mx::complex64_t>(mx::complex64_t value, float threshold) {
  if (threshold == 0.0f) {
    return value != mx::complex64_t{};
  }
  return std::abs(value) > threshold;
}

mx::Dtype index_dtype_from_bits(int index_dtype_bits) {
  if (index_dtype_bits == 32) {
    return mx::int32;
  }
  if (index_dtype_bits == 64) {
    return mx::int64;
  }
  throw std::invalid_argument(
      "csr_fromdense index dtype must be encoded as 32 or 64.");
}

template <typename I> void check_output_nnz(size_t nnz, const char *op_name) {
  if (nnz > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(op_name) +
                              " nnz exceeds MLX shape limits.");
  }
  if (nnz > static_cast<size_t>(std::numeric_limits<I>::max())) {
    throw std::overflow_error(std::string(op_name) +
                              " nnz exceeds index dtype capacity.");
  }
}

class FromDenseCounts : public mx::Primitive {
public:
  FromDenseCounts(mx::Stream stream, int n_cols, float threshold)
      : Primitive(stream), n_cols_(n_cols), threshold_(threshold) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array> &,
                             const std::vector<mx::array> &,
                             const std::vector<int> &) override;

  std::vector<mx::array> vjp(const std::vector<mx::array> &,
                             const std::vector<mx::array> &,
                             const std::vector<int> &,
                             const std::vector<mx::array> &) override;

  const char *name() const override { return "FromDenseCounts"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const FromDenseCounts &>(other);
    return n_cols_ == rhs.n_cols_ && threshold_ == rhs.threshold_;
  }

private:
  int n_cols_;
  float threshold_;
};

class FromDenseFill : public mx::Primitive {
public:
  FromDenseFill(mx::Stream stream, int n_cols, float threshold)
      : Primitive(stream), n_cols_(n_cols), threshold_(threshold) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array> &,
                             const std::vector<mx::array> &,
                             const std::vector<int> &) override;

  std::vector<mx::array> vjp(const std::vector<mx::array> &,
                             const std::vector<mx::array> &,
                             const std::vector<int> &,
                             const std::vector<mx::array> &) override;

  const char *name() const override { return "FromDenseFill"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const FromDenseFill &>(other);
    return n_cols_ == rhs.n_cols_ && threshold_ == rhs.threshold_;
  }

private:
  int n_cols_;
  float threshold_;
};

template <typename T, typename I>
void fromdense_counts_cpu_impl(const mx::array &dense, mx::array &counts,
                               int n_cols, float threshold, mx::Stream stream) {
  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(dense);
  encoder.set_output_array(counts);

  encoder.dispatch([dense = mx::array::unsafe_weak_copy(dense),
                    counts = mx::array::unsafe_weak_copy(counts), n_cols,
                    threshold]() mutable {
    const auto *dense_ptr = dense.data<T>();
    auto *counts_ptr = counts.data<I>();
    const int n_rows = static_cast<int>(dense.shape(0));

    auto count_rows = [&](CpuRange range) {
      for (int row = range.begin; row < range.end; ++row) {
        I count = I{0};
        const size_t base = static_cast<size_t>(row) * n_cols;
        for (int col = 0; col < n_cols; ++col) {
          if (keep_dense_value<T>(dense_ptr[base + col], threshold)) {
            count += I{1};
          }
        }
        counts_ptr[row] = count;
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

template <typename T, typename I>
void fromdense_fill_cpu_impl(const mx::array &dense,
                             const mx::array &out_indptr, mx::array &out_data,
                             mx::array &out_indices, int n_cols,
                             float threshold, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(dense);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_indices);

  encoder.dispatch([dense = mx::array::unsafe_weak_copy(dense),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_indices = mx::array::unsafe_weak_copy(out_indices),
                    n_cols, threshold]() mutable {
    const auto *dense_ptr = dense.data<T>();
    const auto *out_indptr_ptr = out_indptr.data<I>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_indices_ptr = out_indices.data<I>();
    const int n_rows = static_cast<int>(dense.shape(0));

    auto fill_rows = [&](CpuRange range) {
      for (int row = range.begin; row < range.end; ++row) {
        I write = out_indptr_ptr[row];
        const size_t base = static_cast<size_t>(row) * n_cols;
        for (int col = 0; col < n_cols; ++col) {
          const T value = dense_ptr[base + col];
          if (keep_dense_value<T>(value, threshold)) {
            out_data_ptr[write] = value;
            out_indices_ptr[write] = static_cast<I>(col);
            ++write;
          }
        }
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

[[noreturn]] void throw_fromdense_autodiff_error(const char *primitive,
                                                 const char *transform) {
  throw std::runtime_error(
      std::string("csr_fromdense ") + transform + " is not supported for " +
      primitive +
      " because fromdense has value-dependent sparse topology. Use a fixed-"
      "topology sparse constructor when differentiating sparse values.");
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
      throw std::runtime_error("csr_fromdense produced a negative row count.");
    }
    total += count;
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error("csr_fromdense nnz exceeds MLX shape limits.");
    }
    if (total > static_cast<int64_t>(std::numeric_limits<I>::max())) {
      throw std::overflow_error(
          "csr_fromdense nnz exceeds index dtype capacity.");
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

template <typename T, typename I> struct LocalDenseCsr {
  CpuRange range{0, 0};
  std::vector<T> data;
  std::vector<I> indices;
  std::vector<I> row_counts;
};

template <typename T, typename I>
std::tuple<mx::array, mx::array, mx::array>
csr_fromdense_host_serial(const mx::array &dense, int n_rows, int n_cols,
                          mx::Dtype index_dtype, float threshold) {
  const auto *dense_ptr = dense.data<T>();
  std::vector<I> out_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  std::vector<T> out_data;
  std::vector<I> out_indices;

  constexpr size_t kMaxInitialReserve = 1 << 20;
  const auto dense_size = static_cast<size_t>(dense.size());
  out_data.reserve(std::min(dense_size, kMaxInitialReserve));
  out_indices.reserve(std::min(dense_size, kMaxInitialReserve));

  for (int row = 0; row < n_rows; ++row) {
    const size_t base = static_cast<size_t>(row) * n_cols;
    for (int col = 0; col < n_cols; ++col) {
      const T value = dense_ptr[base + col];
      if (keep_dense_value<T>(value, threshold)) {
        out_data.push_back(value);
        out_indices.push_back(static_cast<I>(col));
      }
    }
    check_output_nnz<I>(out_data.size(), "csr_fromdense");
    out_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(out_data.size());
  }

  const int out_nnz = static_cast<int>(out_data.size());
  return {mx::array(out_data.begin(), mx::Shape{out_nnz}, dense.dtype()),
          mx::array(out_indices.begin(), mx::Shape{out_nnz}, index_dtype),
          mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    index_dtype)};
}

template <typename T, typename I>
std::tuple<mx::array, mx::array, mx::array>
csr_fromdense_host_parallel(const mx::array &dense, int n_rows, int n_cols,
                            mx::Dtype index_dtype, float threshold,
                            int requested_workers) {
  const auto ranges = equal_cpu_ranges(n_rows, requested_workers);
  if (ranges.size() <= 1) {
    return csr_fromdense_host_serial<T, I>(dense, n_rows, n_cols, index_dtype,
                                           threshold);
  }

  const auto *dense_ptr = dense.data<T>();
  std::vector<LocalDenseCsr<T, I>> local_outputs(ranges.size());

  parallel_for_cpu_ranges_indexed(ranges, [&](size_t worker, CpuRange range) {
    auto &local = local_outputs[worker];
    local.range = range;
    local.row_counts.resize(static_cast<size_t>(range.end - range.begin), I{0});

    const auto local_rows = static_cast<size_t>(range.end - range.begin);
    const auto dense_size = static_cast<size_t>(dense.size());
    const auto reserve = std::min(
        dense_size, std::min<size_t>(local_rows * static_cast<size_t>(n_cols),
                                     size_t{1} << 20));
    local.data.reserve(reserve);
    local.indices.reserve(reserve);

    for (int row = range.begin; row < range.end; ++row) {
      const auto before = local.data.size();
      const size_t base = static_cast<size_t>(row) * n_cols;
      for (int col = 0; col < n_cols; ++col) {
        const T value = dense_ptr[base + col];
        if (keep_dense_value<T>(value, threshold)) {
          local.data.push_back(value);
          local.indices.push_back(static_cast<I>(col));
        }
      }
      const auto row_nnz = local.data.size() - before;
      check_output_nnz<I>(row_nnz, "csr_fromdense row");
      local.row_counts[static_cast<size_t>(row - range.begin)] =
          static_cast<I>(row_nnz);
    }
  });

  std::vector<I> out_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  size_t total_nnz = 0;
  for (const auto &local : local_outputs) {
    for (int row = local.range.begin; row < local.range.end; ++row) {
      const auto count = static_cast<size_t>(
          local.row_counts[static_cast<size_t>(row - local.range.begin)]);
      total_nnz += count;
      check_output_nnz<I>(total_nnz, "csr_fromdense");
      out_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(total_nnz);
    }
  }

  std::vector<T> out_data(total_nnz);
  std::vector<I> out_indices(total_nnz);
  for (const auto &local : local_outputs) {
    size_t read = 0;
    for (int row = local.range.begin; row < local.range.end; ++row) {
      const auto count = static_cast<size_t>(
          local.row_counts[static_cast<size_t>(row - local.range.begin)]);
      const auto write = static_cast<size_t>(out_indptr[row]);
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
      throw std::runtime_error(
          "csr_fromdense internal parallel count mismatch.");
    }
  }

  const int out_nnz = static_cast<int>(total_nnz);
  return {mx::array(out_data.begin(), mx::Shape{out_nnz}, dense.dtype()),
          mx::array(out_indices.begin(), mx::Shape{out_nnz}, index_dtype),
          mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    index_dtype)};
}

template <typename T, typename I>
std::tuple<mx::array, mx::array, mx::array>
csr_fromdense_host_typed(mx::array dense, int n_rows, int n_cols,
                         mx::Dtype index_dtype, float threshold) {
  dense.eval();
  const int workers = configured_cpu_worker_count();
  if (workers > 1 && n_rows > 0) {
    return csr_fromdense_host_parallel<T, I>(dense, n_rows, n_cols, index_dtype,
                                             threshold, workers);
  }
  return csr_fromdense_host_serial<T, I>(dense, n_rows, n_cols, index_dtype,
                                         threshold);
}

template <typename T>
std::tuple<mx::array, mx::array, mx::array>
csr_fromdense_host_value(mx::array dense, int n_rows, int n_cols,
                         mx::Dtype index_dtype, float threshold) {
  if (index_dtype == mx::int32) {
    return csr_fromdense_host_typed<T, int32_t>(std::move(dense), n_rows,
                                                n_cols, index_dtype, threshold);
  }
  return csr_fromdense_host_typed<T, int64_t>(std::move(dense), n_rows, n_cols,
                                              index_dtype, threshold);
}

std::tuple<mx::array, mx::array, mx::array>
csr_fromdense_host(mx::array dense, int n_rows, int n_cols,
                   mx::Dtype index_dtype, float threshold) {
  if (dense.dtype() == mx::float32) {
    return csr_fromdense_host_value<float>(std::move(dense), n_rows, n_cols,
                                           index_dtype, threshold);
  }
  if (dense.dtype() == mx::float16) {
    return csr_fromdense_host_value<mx::float16_t>(
        std::move(dense), n_rows, n_cols, index_dtype, threshold);
  }
  if (dense.dtype() == mx::bfloat16) {
    return csr_fromdense_host_value<mx::bfloat16_t>(
        std::move(dense), n_rows, n_cols, index_dtype, threshold);
  }
  if (dense.dtype() == mx::complex64) {
    return csr_fromdense_host_value<mx::complex64_t>(
        std::move(dense), n_rows, n_cols, index_dtype, threshold);
  }
  throw std::runtime_error("csr_fromdense unsupported value dtype.");
}

mx::array fromdense_counts(const mx::array &dense, int n_rows,
                           mx::Dtype index_dtype, int n_cols, float threshold,
                           mx::Stream stream) {
  auto primitive = std::make_shared<FromDenseCounts>(stream, n_cols, threshold);
  return mx::array(mx::Shape{n_rows}, index_dtype, primitive, {dense});
}

std::tuple<mx::array, mx::array>
fromdense_fill(const mx::array &dense, const mx::array &out_indptr, int out_nnz,
               int n_cols, float threshold, mx::Dtype index_dtype,
               mx::Stream stream) {
  auto primitive = std::make_shared<FromDenseFill>(stream, n_cols, threshold);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}}, {dense.dtype(), index_dtype},
      primitive, {dense, out_indptr});
  return {outputs[0], outputs[1]};
}

} // namespace

void FromDenseCounts::eval_cpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  const auto &dense = inputs[0];
  auto &counts = outputs[0];

#define DISPATCH_FROMDENSE_COUNTS_VALUE(DTYPE, TYPE)                           \
  if (dense.dtype() == DTYPE) {                                                \
    if (counts.dtype() == mx::int32) {                                         \
      fromdense_counts_cpu_impl<TYPE, int32_t>(dense, counts, n_cols_,         \
                                               threshold_, stream());          \
    } else {                                                                   \
      fromdense_counts_cpu_impl<TYPE, int64_t>(dense, counts, n_cols_,         \
                                               threshold_, stream());          \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_FROMDENSE_COUNTS_VALUE(mx::float32, float)
  DISPATCH_FROMDENSE_COUNTS_VALUE(mx::float16, mx::float16_t)
  DISPATCH_FROMDENSE_COUNTS_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_FROMDENSE_COUNTS_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_FROMDENSE_COUNTS_VALUE

  throw std::runtime_error("csr_fromdense unsupported value dtype.");
}

std::vector<mx::array> FromDenseCounts::jvp(const std::vector<mx::array> &,
                                            const std::vector<mx::array> &,
                                            const std::vector<int> &) {
  throw_fromdense_autodiff_error("FromDenseCounts", "JVP");
}

std::vector<mx::array> FromDenseCounts::vjp(const std::vector<mx::array> &,
                                            const std::vector<mx::array> &,
                                            const std::vector<int> &,
                                            const std::vector<mx::array> &) {
  throw_fromdense_autodiff_error("FromDenseCounts", "VJP");
}

#ifdef _METAL_
void FromDenseCounts::eval_gpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  const auto &dense = inputs[0];
  auto &counts = outputs[0];

  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name = std::string("fromdense_counts_") +
                     value_kernel_suffix(dense.dtype()) + "_" +
                     index_kernel_suffix(counts.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(dense, 0);
  encoder.set_output_array(counts, 1);
  const int n_rows = static_cast<int>(dense.shape(0));
  encoder.set_bytes(n_rows, 2);
  encoder.set_bytes(n_cols_, 3);
  encoder.set_bytes(threshold_, 4);

  auto threads = std::max<size_t>(static_cast<size_t>(n_rows), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void FromDenseCounts::eval_gpu(const std::vector<mx::array> &,
                               std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_fromdense has no GPU implementation in this build.");
}
#endif

void FromDenseFill::eval_cpu(const std::vector<mx::array> &inputs,
                             std::vector<mx::array> &outputs) {
  const auto &dense = inputs[0];
  const auto &out_indptr = inputs[1];

#define DISPATCH_FROMDENSE_FILL_VALUE(DTYPE, TYPE)                             \
  if (dense.dtype() == DTYPE) {                                                \
    if (out_indptr.dtype() == mx::int32) {                                     \
      fromdense_fill_cpu_impl<TYPE, int32_t>(dense, out_indptr, outputs[0],    \
                                             outputs[1], n_cols_, threshold_,  \
                                             stream());                        \
    } else {                                                                   \
      fromdense_fill_cpu_impl<TYPE, int64_t>(dense, out_indptr, outputs[0],    \
                                             outputs[1], n_cols_, threshold_,  \
                                             stream());                        \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_FROMDENSE_FILL_VALUE(mx::float32, float)
  DISPATCH_FROMDENSE_FILL_VALUE(mx::float16, mx::float16_t)
  DISPATCH_FROMDENSE_FILL_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_FROMDENSE_FILL_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_FROMDENSE_FILL_VALUE

  throw std::runtime_error("csr_fromdense unsupported value dtype.");
}

std::vector<mx::array> FromDenseFill::jvp(const std::vector<mx::array> &,
                                          const std::vector<mx::array> &,
                                          const std::vector<int> &) {
  throw_fromdense_autodiff_error("FromDenseFill", "JVP");
}

std::vector<mx::array> FromDenseFill::vjp(const std::vector<mx::array> &,
                                          const std::vector<mx::array> &,
                                          const std::vector<int> &,
                                          const std::vector<mx::array> &) {
  throw_fromdense_autodiff_error("FromDenseFill", "VJP");
}

#ifdef _METAL_
void FromDenseFill::eval_gpu(const std::vector<mx::array> &inputs,
                             std::vector<mx::array> &outputs) {
  const auto &dense = inputs[0];
  const auto &out_indptr = inputs[1];
  auto &out_data = outputs[0];
  auto &out_indices = outputs[1];

  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name = std::string("fromdense_fill_") +
                     value_kernel_suffix(dense.dtype()) + "_" +
                     index_kernel_suffix(out_indptr.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(dense, 0);
  encoder.set_input_array(out_indptr, 1);
  encoder.set_output_array(out_data, 2);
  encoder.set_output_array(out_indices, 3);
  const int n_rows = static_cast<int>(dense.shape(0));
  encoder.set_bytes(n_rows, 4);
  encoder.set_bytes(n_cols_, 5);
  encoder.set_bytes(threshold_, 6);

  auto threads = std::max<size_t>(static_cast<size_t>(n_rows), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void FromDenseFill::eval_gpu(const std::vector<mx::array> &,
                             std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_fromdense has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
csr_fromdense(const mx::array &dense, int index_dtype_bits, float threshold,
              mx::StreamOrDevice s) {
  require_rank(dense, 2, "csr_fromdense dense");
  require_supported_value_dtype(dense, "csr_fromdense dense");
  if (threshold < 0.0f) {
    throw std::invalid_argument(
        "csr_fromdense threshold must be non-negative.");
  }

  const auto index_dtype = index_dtype_from_bits(index_dtype_bits);
  const int n_rows = dense.shape(0);
  const int n_cols = dense.shape(1);
  if (index_dtype == mx::int32 &&
      n_cols > std::numeric_limits<int32_t>::max()) {
    throw std::overflow_error(
        "csr_fromdense n_cols exceeds int32 index capacity.");
  }

  auto stream = mx::to_stream(s);
  auto dense_contig = mx::contiguous(dense, false, stream);
  auto counts = fromdense_counts(dense_contig, n_rows, index_dtype, n_cols,
                                 threshold, stream);
  mx::eval(counts);
  auto [out_indptr, out_nnz] =
      build_indptr_from_counts(counts, n_rows, index_dtype);
  auto [out_data, out_indices] =
      fromdense_fill(dense_contig, out_indptr, out_nnz, n_cols, threshold,
                     index_dtype, stream);
  return {out_data, out_indices, out_indptr};
}

} // namespace mlx_sparse
