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

#include "random/random.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "common/common.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "random/common.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

class RandomCooUnpackSortedKeys : public mx::Primitive {
public:
  RandomCooUnpackSortedKeys(mx::Stream stream, int64_t n_cols)
      : Primitive(stream), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "RandomCooUnpackSortedKeys"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const RandomCooUnpackSortedKeys &>(other);
    return n_cols_ == rhs.n_cols_;
  }

private:
  int64_t n_cols_;
};

std::tuple<mx::array, mx::array>
random_coo_unpack_sorted_keys(const mx::array &keys, int64_t n_cols,
                              mx::Dtype index_dtype, mx::Stream stream) {
  auto primitive = std::make_shared<RandomCooUnpackSortedKeys>(stream, n_cols);
  auto outputs =
      mx::array::make_arrays({mx::Shape{static_cast<int>(keys.size())},
                              mx::Shape{static_cast<int>(keys.size())}},
                             {index_dtype, index_dtype}, primitive, {keys});
  return {outputs[0], outputs[1]};
}

} // namespace

void RandomCooUnpackSortedKeys::eval_cpu(const std::vector<mx::array> &inputs,
                                         std::vector<mx::array> &outputs) {
  const auto &keys = inputs[0];
  auto &row = outputs[0];
  auto &col = outputs[1];
  row.set_data(mx::allocator::malloc(row.nbytes()));
  col.set_data(mx::allocator::malloc(col.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(keys);
  encoder.set_output_array(row);
  encoder.set_output_array(col);
  encoder.dispatch([keys = mx::array::unsafe_weak_copy(keys),
                    row = mx::array::unsafe_weak_copy(row),
                    col = mx::array::unsafe_weak_copy(col),
                    n_cols = n_cols_]() mutable {
    const auto *keys_ptr = keys.data<int64_t>();
    const auto nnz = keys.size();
    if (row.dtype() == mx::int32) {
      auto *row_ptr = row.data<int32_t>();
      auto *col_ptr = col.data<int32_t>();
      for (size_t i = 0; i < nnz; ++i) {
        const auto linear = static_cast<uint64_t>(keys_ptr[i]);
        row_ptr[i] =
            static_cast<int32_t>(linear / static_cast<uint64_t>(n_cols));
        col_ptr[i] =
            static_cast<int32_t>(linear % static_cast<uint64_t>(n_cols));
      }
    } else {
      auto *row_ptr = row.data<int64_t>();
      auto *col_ptr = col.data<int64_t>();
      for (size_t i = 0; i < nnz; ++i) {
        const auto linear = static_cast<uint64_t>(keys_ptr[i]);
        row_ptr[i] =
            static_cast<int64_t>(linear / static_cast<uint64_t>(n_cols));
        col_ptr[i] =
            static_cast<int64_t>(linear % static_cast<uint64_t>(n_cols));
      }
    }
  });
}

#ifdef _METAL_
void RandomCooUnpackSortedKeys::eval_gpu(const std::vector<mx::array> &inputs,
                                         std::vector<mx::array> &outputs) {
  const auto &keys = inputs[0];
  auto &row = outputs[0];
  auto &col = outputs[1];
  row.set_data(mx::allocator::malloc(row.nbytes()));
  col.set_data(mx::allocator::malloc(col.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name = std::string("random_coo_unpack_sorted_keys_") +
                     index_kernel_suffix(row.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(keys, 0);
  encoder.set_output_array(row, 1);
  encoder.set_output_array(col, 2);
  encoder.set_bytes(n_cols_, 3);
  const int64_t nnz = static_cast<int64_t>(keys.size());
  encoder.set_bytes(nnz, 4);

  auto threads = std::max<size_t>(keys.size(), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void RandomCooUnpackSortedKeys::eval_gpu(const std::vector<mx::array> &,
                                         std::vector<mx::array> &) {
  throw std::runtime_error(
      "random COO key unpack has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array>
random_coo_indices(const mx::array &key, int64_t n_rows, int64_t n_cols,
                   int64_t nnz, int index_dtype_bits, mx::StreamOrDevice s) {
  check_random_key(key);
  const auto index_dtype = random_index_dtype_from_bits(index_dtype_bits);
  check_random_shape(n_rows, n_cols, nnz, index_dtype);

  auto stream = mx::to_stream(s);
  auto key_contig = mx::contiguous(key, false, stream);
  if (nnz == 0) {
    return {mx::zeros(mx::Shape{0}, index_dtype, stream),
            mx::zeros(mx::Shape{0}, index_dtype, stream)};
  }
  auto keys = random_structural_keys(key_contig, n_rows, n_cols, nnz,
                                     /*csc=*/false, stream);
  auto sorted_keys = mx::sort(keys, stream);
  return random_coo_unpack_sorted_keys(sorted_keys, n_cols, index_dtype,
                                       stream);
}

} // namespace mlx_sparse
