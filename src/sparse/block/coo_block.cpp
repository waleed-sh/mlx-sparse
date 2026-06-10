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

#include "sparse/block/coo_block.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/common.h"
#include "common/cpu_parallel.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

std::vector<int>
checked_offsets_from_sizes(const std::vector<mx::array> &arrays,
                           const char *op_name) {
  std::vector<int> offsets(arrays.size() + 1, 0);
  int64_t total = 0;
  for (size_t i = 0; i < arrays.size(); ++i) {
    const auto size = static_cast<int64_t>(arrays[i].size());
    if (size < 0) {
      throw std::invalid_argument(std::string(op_name) +
                                  " input sizes must be non-negative.");
    }
    total += size;
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error(std::string(op_name) +
                                " output nnz exceeds MLX shape limits.");
    }
    offsets[i + 1] = static_cast<int>(total);
  }
  return offsets;
}

void require_non_empty_inputs(const std::vector<mx::array> &arrays,
                              const char *op_name) {
  if (arrays.empty()) {
    throw std::invalid_argument(std::string(op_name) +
                                " requires at least one input array.");
  }
}

void require_same_value_dtype_all(const std::vector<mx::array> &arrays,
                                  const char *op_name) {
  require_non_empty_inputs(arrays, op_name);
  require_supported_value_dtype(arrays[0], op_name);
  for (size_t i = 0; i < arrays.size(); ++i) {
    require_rank(arrays[i], 1, op_name);
    require_supported_value_dtype(arrays[i], op_name);
    if (arrays[i].dtype() != arrays[0].dtype()) {
      throw std::invalid_argument(std::string(op_name) +
                                  " input value dtypes must match.");
    }
  }
}

void require_same_index_dtype_all(const std::vector<mx::array> &arrays,
                                  const char *op_name) {
  require_non_empty_inputs(arrays, op_name);
  require_index_dtype(arrays[0], op_name);
  for (size_t i = 0; i < arrays.size(); ++i) {
    require_rank(arrays[i], 1, op_name);
    require_index_dtype(arrays[i], op_name);
    if (arrays[i].dtype() != arrays[0].dtype()) {
      throw std::invalid_argument(std::string(op_name) +
                                  " input index dtypes must match.");
    }
  }
}

void check_index_capacity(int n_rows, int n_cols, mx::Dtype index_dtype) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "coo_block output shape dimensions must be non-negative.");
  }
  if (index_dtype == mx::int32 &&
      (static_cast<int64_t>(n_rows) > std::numeric_limits<int32_t>::max() ||
       static_cast<int64_t>(n_cols) > std::numeric_limits<int32_t>::max())) {
    throw std::overflow_error(
        "coo_block output shape exceeds int32 index capacity.");
  }
}

std::string block_data_kernel_name(mx::Dtype value_dtype) {
  return std::string("coo_block_data_copy_") + value_kernel_suffix(value_dtype);
}

std::string block_indices_kernel_name(mx::Dtype index_dtype) {
  return std::string("coo_block_indices_offset_") +
         index_kernel_suffix(index_dtype);
}

class COOBlockData : public mx::Primitive {
public:
  COOBlockData(mx::Stream stream, std::vector<int> offsets)
      : Primitive(stream), offsets_(std::move(offsets)) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array> &primals,
                             const std::vector<mx::array> &tangents,
                             const std::vector<int> &argnums) override;

  std::vector<mx::array> vjp(const std::vector<mx::array> &,
                             const std::vector<mx::array> &cotangents,
                             const std::vector<int> &argnums,
                             const std::vector<mx::array> &) override;

  const char *name() const override { return "COOBlockData"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOBlockData &>(other);
    return offsets_ == rhs.offsets_;
  }

private:
  std::vector<int> offsets_;
};

class COOBlockIndices : public mx::Primitive {
public:
  COOBlockIndices(mx::Stream stream, std::vector<int> nnz_offsets,
                  std::vector<int> row_offsets, std::vector<int> col_offsets)
      : Primitive(stream), nnz_offsets_(std::move(nnz_offsets)),
        row_offsets_(std::move(row_offsets)),
        col_offsets_(std::move(col_offsets)) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOBlockIndices"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOBlockIndices &>(other);
    return nnz_offsets_ == rhs.nnz_offsets_ &&
           row_offsets_ == rhs.row_offsets_ && col_offsets_ == rhs.col_offsets_;
  }

private:
  std::vector<int> nnz_offsets_;
  std::vector<int> row_offsets_;
  std::vector<int> col_offsets_;
};

template <typename T>
void block_data_cpu_impl(const std::vector<mx::array> &inputs, mx::array &out,
                         const std::vector<int> &offsets, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  for (const auto &input : inputs) {
    encoder.set_input_array(input);
  }
  encoder.set_output_array(out);

  encoder.dispatch([inputs, out = mx::array::unsafe_weak_copy(out),
                    offsets]() mutable {
    std::vector<mx::array> weak_inputs;
    weak_inputs.reserve(inputs.size());
    for (const auto &input : inputs) {
      weak_inputs.push_back(mx::array::unsafe_weak_copy(input));
    }

    auto *dst = out.data<T>();
    const int total = offsets.empty() ? 0 : offsets.back();

    auto fill_range = [&](CpuRange range) {
      for (int k = range.begin; k < range.end; ++k) {
        const auto block_it =
            std::upper_bound(offsets.begin(), offsets.end(), k);
        const auto block =
            static_cast<size_t>(std::distance(offsets.begin(), block_it) - 1);
        const int local = k - offsets[block];
        dst[k] = weak_inputs[block].data<T>()[local];
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || total <= 0) {
      fill_range({0, total});
      return;
    }
    parallel_for_cpu_ranges(equal_cpu_ranges(total, workers), fill_range);
  });
}

template <typename I>
void block_indices_cpu_impl(const std::vector<mx::array> &inputs,
                            mx::array &out_row, mx::array &out_col,
                            const std::vector<int> &nnz_offsets,
                            const std::vector<int> &row_offsets,
                            const std::vector<int> &col_offsets,
                            mx::Stream stream) {
  out_row.set_data(mx::allocator::malloc(out_row.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  for (const auto &input : inputs) {
    encoder.set_input_array(input);
  }
  encoder.set_output_array(out_row);
  encoder.set_output_array(out_col);

  encoder.dispatch([inputs, out_row = mx::array::unsafe_weak_copy(out_row),
                    out_col = mx::array::unsafe_weak_copy(out_col), nnz_offsets,
                    row_offsets, col_offsets]() mutable {
    std::vector<mx::array> weak_inputs;
    weak_inputs.reserve(inputs.size());
    for (const auto &input : inputs) {
      weak_inputs.push_back(mx::array::unsafe_weak_copy(input));
    }

    auto *row_dst = out_row.data<I>();
    auto *col_dst = out_col.data<I>();
    const int total = nnz_offsets.empty() ? 0 : nnz_offsets.back();

    auto fill_range = [&](CpuRange range) {
      for (int k = range.begin; k < range.end; ++k) {
        const auto block_it =
            std::upper_bound(nnz_offsets.begin(), nnz_offsets.end(), k);
        const auto block = static_cast<size_t>(
            std::distance(nnz_offsets.begin(), block_it) - 1);
        const int local = k - nnz_offsets[block];
        const auto *row_src = weak_inputs[2 * block].data<I>();
        const auto *col_src = weak_inputs[2 * block + 1].data<I>();
        row_dst[k] = static_cast<I>(static_cast<int64_t>(row_src[local]) +
                                    static_cast<int64_t>(row_offsets[block]));
        col_dst[k] = static_cast<I>(static_cast<int64_t>(col_src[local]) +
                                    static_cast<int64_t>(col_offsets[block]));
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || total <= 0) {
      fill_range({0, total});
      return;
    }
    parallel_for_cpu_ranges(equal_cpu_ranges(total, workers), fill_range);
  });
}

} // namespace

void COOBlockData::eval_cpu(const std::vector<mx::array> &inputs,
                            std::vector<mx::array> &outputs) {
  const auto dtype = inputs[0].dtype();
  if (dtype == mx::float32) {
    block_data_cpu_impl<float>(inputs, outputs[0], offsets_, stream());
    return;
  }
  if (dtype == mx::float16) {
    block_data_cpu_impl<mx::float16_t>(inputs, outputs[0], offsets_, stream());
    return;
  }
  if (dtype == mx::bfloat16) {
    block_data_cpu_impl<mx::bfloat16_t>(inputs, outputs[0], offsets_, stream());
    return;
  }
  if (dtype == mx::complex64) {
    block_data_cpu_impl<mx::complex64_t>(inputs, outputs[0], offsets_,
                                         stream());
    return;
  }
  throw std::runtime_error("coo_block_data unsupported value dtype.");
}

#ifdef _METAL_
void COOBlockData::eval_gpu(const std::vector<mx::array> &inputs,
                            std::vector<mx::array> &outputs) {
  auto &out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(block_data_kernel_name(out.dtype()), lib);
  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);

  for (size_t block = 0; block < inputs.size(); ++block) {
    const int nnz = offsets_[block + 1] - offsets_[block];
    if (nnz == 0) {
      continue;
    }
    encoder.set_input_array(inputs[block], 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(nnz, 2);
    encoder.set_bytes(offsets_[block], 3);
    const auto threads = std::max<size_t>(static_cast<size_t>(nnz), 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
  }
}
#else
void COOBlockData::eval_gpu(const std::vector<mx::array> &,
                            std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_block_data has no GPU implementation in this build.");
}
#endif

std::vector<mx::array> COOBlockData::jvp(const std::vector<mx::array> &primals,
                                         const std::vector<mx::array> &tangents,
                                         const std::vector<int> &argnums) {
  std::vector<mx::array> tangent_blocks;
  tangent_blocks.reserve(primals.size());
  for (size_t arg = 0; arg < primals.size(); ++arg) {
    bool found = false;
    for (size_t tangent_index = 0; tangent_index < argnums.size();
         ++tangent_index) {
      if (argnums[tangent_index] == static_cast<int>(arg)) {
        tangent_blocks.push_back(tangents[tangent_index]);
        found = true;
        break;
      }
    }
    if (!found) {
      tangent_blocks.push_back(mx::zeros_like(primals[arg], stream()));
    }
  }
  return {coo_block_data(tangent_blocks, stream())};
}

std::vector<mx::array> COOBlockData::vjp(
    const std::vector<mx::array> &, const std::vector<mx::array> &cotangents,
    const std::vector<int> &argnums, const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  for (const int arg : argnums) {
    if (arg < 0 || static_cast<size_t>(arg + 1) >= offsets_.size()) {
      throw std::runtime_error(
          "COOBlockData VJP received an out-of-range argument index.");
    }
    vjps.push_back(mx::slice(cotangents[0], mx::Shape{offsets_[arg]},
                             mx::Shape{offsets_[arg + 1]}, stream()));
  }
  return vjps;
}

void COOBlockIndices::eval_cpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  const auto dtype = inputs[0].dtype();
  if (dtype == mx::int32) {
    block_indices_cpu_impl<int32_t>(inputs, outputs[0], outputs[1],
                                    nnz_offsets_, row_offsets_, col_offsets_,
                                    stream());
    return;
  }
  if (dtype == mx::int64) {
    block_indices_cpu_impl<int64_t>(inputs, outputs[0], outputs[1],
                                    nnz_offsets_, row_offsets_, col_offsets_,
                                    stream());
    return;
  }
  throw std::runtime_error("coo_block_indices unsupported index dtype.");
}

#ifdef _METAL_
void COOBlockIndices::eval_gpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &out_row = outputs[0];
  auto &out_col = outputs[1];
  out_row.set_data(mx::allocator::malloc(out_row.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel =
      device.get_kernel(block_indices_kernel_name(out_row.dtype()), lib);
  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);

  for (size_t block = 0; block < row_offsets_.size(); ++block) {
    const int nnz = nnz_offsets_[block + 1] - nnz_offsets_[block];
    if (nnz == 0) {
      continue;
    }
    encoder.set_input_array(inputs[2 * block], 0);
    encoder.set_input_array(inputs[2 * block + 1], 1);
    encoder.set_output_array(out_row, 2);
    encoder.set_output_array(out_col, 3);
    encoder.set_bytes(nnz, 4);
    encoder.set_bytes(nnz_offsets_[block], 5);
    encoder.set_bytes(row_offsets_[block], 6);
    encoder.set_bytes(col_offsets_[block], 7);
    const auto threads = std::max<size_t>(static_cast<size_t>(nnz), 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
  }
}
#else
void COOBlockIndices::eval_gpu(const std::vector<mx::array> &,
                               std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_block_indices has no GPU implementation in this build.");
}
#endif

mx::array coo_block_data(const std::vector<mx::array> &data_arrays,
                         mx::StreamOrDevice s) {
  require_same_value_dtype_all(data_arrays, "coo_block_data");
  const auto offsets =
      checked_offsets_from_sizes(data_arrays, "coo_block_data");
  const int out_nnz = offsets.back();

  auto stream = mx::to_stream(s);
  std::vector<mx::array> inputs;
  inputs.reserve(data_arrays.size());
  for (const auto &data : data_arrays) {
    inputs.push_back(mx::contiguous(data, false, stream));
  }
  auto primitive = std::make_shared<COOBlockData>(stream, offsets);
  return mx::array(mx::Shape{out_nnz}, data_arrays[0].dtype(), primitive,
                   inputs);
}

std::tuple<mx::array, mx::array>
coo_block_indices(const std::vector<mx::array> &row_arrays,
                  const std::vector<mx::array> &col_arrays,
                  const std::vector<int> &row_offsets,
                  const std::vector<int> &col_offsets, int n_rows, int n_cols,
                  mx::StreamOrDevice s) {
  if (row_arrays.size() != col_arrays.size() ||
      row_arrays.size() != row_offsets.size() ||
      row_arrays.size() != col_offsets.size()) {
    throw std::invalid_argument(
        "coo_block_indices requires matching row, column, and offset counts.");
  }
  require_same_index_dtype_all(row_arrays, "coo_block_indices row");
  require_same_index_dtype_all(col_arrays, "coo_block_indices col");
  if (row_arrays[0].dtype() != col_arrays[0].dtype()) {
    throw std::invalid_argument(
        "coo_block_indices row and column dtypes must match.");
  }
  for (size_t i = 0; i < row_arrays.size(); ++i) {
    if (row_arrays[i].size() != col_arrays[i].size()) {
      throw std::invalid_argument(
          "coo_block_indices row and column arrays must have equal lengths.");
    }
    if (row_offsets[i] < 0 || col_offsets[i] < 0) {
      throw std::invalid_argument(
          "coo_block_indices offsets must be non-negative.");
    }
  }
  check_index_capacity(n_rows, n_cols, row_arrays[0].dtype());
  const auto nnz_offsets =
      checked_offsets_from_sizes(row_arrays, "coo_block_indices");
  const int out_nnz = nnz_offsets.back();

  auto stream = mx::to_stream(s);
  std::vector<mx::array> inputs;
  inputs.reserve(row_arrays.size() * 2);
  for (size_t i = 0; i < row_arrays.size(); ++i) {
    inputs.push_back(mx::contiguous(row_arrays[i], false, stream));
    inputs.push_back(mx::contiguous(col_arrays[i], false, stream));
  }
  auto primitive = std::make_shared<COOBlockIndices>(stream, nnz_offsets,
                                                     row_offsets, col_offsets);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}},
      {row_arrays[0].dtype(), row_arrays[0].dtype()}, primitive, inputs);
  return {outputs[0], outputs[1]};
}

std::tuple<mx::array, mx::array, mx::array>
coo_block(const std::vector<mx::array> &data_arrays,
          const std::vector<mx::array> &row_arrays,
          const std::vector<mx::array> &col_arrays,
          const std::vector<int> &row_offsets,
          const std::vector<int> &col_offsets, int n_rows, int n_cols,
          mx::StreamOrDevice s) {
  if (data_arrays.size() != row_arrays.size()) {
    throw std::invalid_argument(
        "coo_block requires matching data and coordinate block counts.");
  }
  for (size_t i = 0; i < data_arrays.size(); ++i) {
    if (data_arrays[i].size() != row_arrays[i].size()) {
      throw std::invalid_argument(
          "coo_block data and coordinate arrays must have equal lengths.");
    }
  }
  auto stream = mx::to_stream(s);
  auto data = coo_block_data(data_arrays, stream);
  auto [row, col] = coo_block_indices(row_arrays, col_arrays, row_offsets,
                                      col_offsets, n_rows, n_cols, stream);
  return {data, row, col};
}

} // namespace mlx_sparse
