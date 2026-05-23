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

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "common/common.h"
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

class FromDenseCounts : public mx::Primitive {
public:
  FromDenseCounts(mx::Stream stream, int n_cols, float threshold)
      : Primitive(stream), n_cols_(n_cols), threshold_(threshold) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

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

    for (int row = 0; row < n_rows; ++row) {
      I count = I{0};
      const size_t base = static_cast<size_t>(row) * n_cols;
      for (int col = 0; col < n_cols; ++col) {
        if (keep_dense_value<T>(dense_ptr[base + col], threshold)) {
          count += I{1};
        }
      }
      counts_ptr[row] = count;
    }
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

    for (int row = 0; row < n_rows; ++row) {
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
