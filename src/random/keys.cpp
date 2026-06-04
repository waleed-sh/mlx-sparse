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
#include "common/cpu_parallel.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

class RandomStructuralKeys : public mx::Primitive {
public:
  RandomStructuralKeys(mx::Stream stream, int64_t n_rows, int64_t n_cols,
                       int64_t nnz, bool csc)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), nnz_(nnz),
        csc_(csc) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "RandomStructuralKeys"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const RandomStructuralKeys &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           nnz_ == rhs.nnz_ && csc_ == rhs.csc_;
  }

private:
  int64_t n_rows_;
  int64_t n_cols_;
  int64_t nnz_;
  bool csc_;
};

} // namespace

void RandomStructuralKeys::eval_cpu(const std::vector<mx::array> &inputs,
                                    std::vector<mx::array> &outputs) {
  const auto &key = inputs[0];
  auto &keys = outputs[0];
  keys.set_data(mx::allocator::malloc(keys.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(key);
  encoder.set_output_array(keys);
  encoder.dispatch([key = mx::array::unsafe_weak_copy(key),
                    keys = mx::array::unsafe_weak_copy(keys), n_rows = n_rows_,
                    n_cols = n_cols_, nnz = nnz_, csc = csc_]() mutable {
    auto *keys_ptr = keys.data<int64_t>();
    if (nnz == 0) {
      return;
    }
    const auto *key_ptr = key.data<uint32_t>();
    const uint64_t seed = keyed_seed(key_ptr, n_rows, n_cols, nnz);
    const uint64_t total =
        static_cast<uint64_t>(n_rows) * static_cast<uint64_t>(n_cols);
    const uint64_t n_cols_u = static_cast<uint64_t>(n_cols);
    const uint64_t n_rows_u = static_cast<uint64_t>(n_rows);

    auto fill_range = [&](CpuRange range) {
      for (int64_t i = range.begin; i < range.end; ++i) {
        const uint64_t linear =
            random_linear_index(static_cast<uint64_t>(i), total, seed);
        if (csc) {
          const uint64_t row = linear / n_cols_u;
          const uint64_t col = linear % n_cols_u;
          keys_ptr[i] = static_cast<int64_t>(col * n_rows_u + row);
        } else {
          keys_ptr[i] = static_cast<int64_t>(linear);
        }
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || nnz <= 0) {
      fill_range({0, static_cast<int>(nnz)});
      return;
    }
    parallel_for_cpu_ranges(equal_cpu_ranges(static_cast<int>(nnz), workers),
                            fill_range);
  });
}

#ifdef _METAL_
void RandomStructuralKeys::eval_gpu(const std::vector<mx::array> &inputs,
                                    std::vector<mx::array> &outputs) {
  const auto &key = inputs[0];
  auto &keys = outputs[0];
  keys.set_data(mx::allocator::malloc(keys.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel("random_structural_keys", lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(key, 0);
  encoder.set_output_array(keys, 1);
  encoder.set_bytes(n_rows_, 2);
  encoder.set_bytes(n_cols_, 3);
  encoder.set_bytes(nnz_, 4);
  const int csc_int = csc_ ? 1 : 0;
  encoder.set_bytes(csc_int, 5);

  auto threads = std::max<size_t>(static_cast<size_t>(nnz_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void RandomStructuralKeys::eval_gpu(const std::vector<mx::array> &,
                                    std::vector<mx::array> &) {
  throw std::runtime_error(
      "random structural keys has no GPU implementation in this build.");
}
#endif

mx::array random_structural_keys(const mx::array &key, int64_t n_rows,
                                 int64_t n_cols, int64_t nnz, bool csc,
                                 mx::Stream stream) {
  auto primitive =
      std::make_shared<RandomStructuralKeys>(stream, n_rows, n_cols, nnz, csc);
  return mx::array(mx::Shape{static_cast<int>(nnz)}, mx::int64, primitive,
                   {key});
}

} // namespace mlx_sparse
