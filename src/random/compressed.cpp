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

#include "random/common.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/common.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

class RandomCompressedCounts : public mx::Primitive {
public:
  RandomCompressedCounts(mx::Stream stream, int64_t n_rows, int64_t n_cols,
                         int64_t nnz, bool csc)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), nnz_(nnz),
        csc_(csc) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "RandomCompressedCounts"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const RandomCompressedCounts &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           nnz_ == rhs.nnz_ && csc_ == rhs.csc_;
  }

private:
  int64_t n_rows_;
  int64_t n_cols_;
  int64_t nnz_;
  bool csc_;
};

class RandomCompressedUnpackSortedKeys : public mx::Primitive {
public:
  RandomCompressedUnpackSortedKeys(mx::Stream stream, int64_t minor_extent,
                                   bool csc)
      : Primitive(stream), minor_extent_(minor_extent), csc_(csc) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override {
    return "RandomCompressedUnpackSortedKeys";
  }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs =
        static_cast<const RandomCompressedUnpackSortedKeys &>(other);
    return minor_extent_ == rhs.minor_extent_ && csc_ == rhs.csc_;
  }

private:
  int64_t minor_extent_;
  bool csc_;
};

} // namespace

void RandomCompressedCounts::eval_cpu(const std::vector<mx::array> &inputs,
                                      std::vector<mx::array> &outputs) {
  const auto &key = inputs[0];
  auto &counts = outputs[0];
  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(key);
  encoder.set_output_array(counts);
  encoder.dispatch([key = mx::array::unsafe_weak_copy(key),
                    counts = mx::array::unsafe_weak_copy(counts),
                    n_rows = n_rows_, n_cols = n_cols_, nnz = nnz_,
                    csc = csc_]() mutable {
    auto *counts_ptr = counts.data<int32_t>();
    const int64_t segments = csc ? n_cols : n_rows;
    std::fill(counts_ptr, counts_ptr + segments, int32_t{0});
    if (nnz == 0) {
      return;
    }
    const auto *key_ptr = key.data<uint32_t>();
    const uint64_t seed = keyed_seed(key_ptr, n_rows, n_cols, nnz);
    const uint64_t total =
        static_cast<uint64_t>(n_rows) * static_cast<uint64_t>(n_cols);
    const uint64_t n_cols_u = static_cast<uint64_t>(n_cols);
    for (int64_t i = 0; i < nnz; ++i) {
      const uint64_t linear =
          random_linear_index(static_cast<uint64_t>(i), total, seed);
      const uint64_t major = csc ? linear % n_cols_u : linear / n_cols_u;
      counts_ptr[major] += int32_t{1};
    }
  });
}

void RandomCompressedUnpackSortedKeys::eval_cpu(
    const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
  const auto &keys = inputs[0];
  auto &indices = outputs[0];
  indices.set_data(mx::allocator::malloc(indices.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(keys);
  encoder.set_output_array(indices);
  encoder.dispatch([keys = mx::array::unsafe_weak_copy(keys),
                    indices = mx::array::unsafe_weak_copy(indices),
                    minor_extent = minor_extent_]() mutable {
    const auto *keys_ptr = keys.data<int64_t>();
    const auto nnz = keys.size();
    if (indices.dtype() == mx::int32) {
      auto *out = indices.data<int32_t>();
      for (size_t i = 0; i < nnz; ++i) {
        out[i] = static_cast<int32_t>(static_cast<uint64_t>(keys_ptr[i]) %
                                      static_cast<uint64_t>(minor_extent));
      }
    } else {
      auto *out = indices.data<int64_t>();
      for (size_t i = 0; i < nnz; ++i) {
        out[i] = static_cast<int64_t>(static_cast<uint64_t>(keys_ptr[i]) %
                                      static_cast<uint64_t>(minor_extent));
      }
    }
  });
}

#ifdef _METAL_
void RandomCompressedCounts::eval_gpu(const std::vector<mx::array> &inputs,
                                      std::vector<mx::array> &outputs) {
  const auto &key = inputs[0];
  auto &counts = outputs[0];
  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  auto *zero_kernel = device.get_kernel("random_compressed_zero_counts", lib);
  encoder.set_compute_pipeline_state(zero_kernel);
  encoder.set_output_array(counts, 0);
  const int64_t segments = csc_ ? n_cols_ : n_rows_;
  encoder.set_bytes(segments, 1);
  auto zero_threads = std::max<size_t>(static_cast<size_t>(segments), 1);
  auto zero_group =
      std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                           MTL::Size(zero_group, 1, 1));

  auto *count_kernel = device.get_kernel("random_compressed_counts", lib);
  encoder.set_compute_pipeline_state(count_kernel);
  encoder.set_input_array(key, 0);
  encoder.set_output_array(counts, 1);
  encoder.set_bytes(n_rows_, 2);
  encoder.set_bytes(n_cols_, 3);
  encoder.set_bytes(nnz_, 4);
  const int csc_int = csc_ ? 1 : 0;
  encoder.set_bytes(csc_int, 5);
  auto count_threads = std::max<size_t>(static_cast<size_t>(nnz_), 1);
  auto count_group =
      std::min(count_threads, count_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(count_threads, 1, 1),
                           MTL::Size(count_group, 1, 1));
}

void RandomCompressedUnpackSortedKeys::eval_gpu(
    const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
  const auto &keys = inputs[0];
  auto &indices = outputs[0];
  indices.set_data(mx::allocator::malloc(indices.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name = std::string("random_compressed_unpack_sorted_keys_") +
                     index_kernel_suffix(indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(keys, 0);
  encoder.set_output_array(indices, 1);
  encoder.set_bytes(minor_extent_, 2);
  const int64_t nnz = static_cast<int64_t>(keys.size());
  encoder.set_bytes(nnz, 3);

  auto threads = std::max<size_t>(keys.size(), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void RandomCompressedCounts::eval_gpu(const std::vector<mx::array> &,
                                      std::vector<mx::array> &) {
  throw std::runtime_error(
      "random compressed counts has no GPU implementation in this build.");
}

void RandomCompressedUnpackSortedKeys::eval_gpu(const std::vector<mx::array> &,
                                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "random compressed key unpack has no GPU implementation in this build.");
}
#endif

mx::array random_compressed_counts(const mx::array &key, int64_t n_rows,
                                   int64_t n_cols, int64_t nnz, bool csc,
                                   mx::Stream stream) {
  const int64_t segments = csc ? n_cols : n_rows;
  auto primitive = std::make_shared<RandomCompressedCounts>(stream, n_rows,
                                                            n_cols, nnz, csc);
  return mx::array(mx::Shape{static_cast<int>(segments)}, mx::int32, primitive,
                   {key});
}

mx::array random_compressed_unpack_sorted_keys(const mx::array &keys,
                                               int64_t minor_extent,
                                               mx::Dtype index_dtype, bool csc,
                                               mx::Stream stream) {
  auto primitive = std::make_shared<RandomCompressedUnpackSortedKeys>(
      stream, minor_extent, csc);
  return mx::array(mx::Shape{static_cast<int>(keys.size())}, index_dtype,
                   primitive, {keys});
}

mx::array random_compressed_indptr(const mx::array &counts,
                                   mx::Dtype index_dtype, mx::Stream stream) {
  auto counts_typed = mx::astype(counts, index_dtype, stream);
  auto prefix = mx::cumsum(counts_typed, 0, false, true, stream);
  auto zero = mx::zeros(mx::Shape{1}, index_dtype, stream);
  return mx::concatenate(std::vector<mx::array>{zero, prefix}, 0, stream);
}

} // namespace mlx_sparse
