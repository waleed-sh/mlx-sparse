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

#include "sparse/csc_tocoo/csc_tocoo.h"

#include <algorithm>
#include <stdexcept>
#include <string>

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

class CSCToCOOCol : public mx::Primitive {
public:
  CSCToCOOCol(mx::Stream stream, int n_cols)
      : Primitive(stream), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCToCOOCol"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCToCOOCol &>(other);
    return n_cols_ == rhs.n_cols_;
  }

private:
  int n_cols_;
};

template <typename I>
void csc_tocoo_col_cpu_impl(const mx::array &indptr, mx::array &col, int n_cols,
                            mx::Stream stream) {
  col.set_data(mx::allocator::malloc(col.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(indptr);
  encoder.set_output_array(col);
  encoder.dispatch([indptr = mx::array::unsafe_weak_copy(indptr),
                    col = mx::array::unsafe_weak_copy(col), n_cols]() mutable {
    const auto *indptr_ptr = indptr.data<I>();
    auto *col_ptr = col.data<I>();

    auto fill_cols = [&](CpuRange range) {
      for (int c = range.begin; c < range.end; ++c) {
        for (I p = indptr_ptr[c]; p < indptr_ptr[c + 1]; ++p) {
          col_ptr[p] = static_cast<I>(c);
        }
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_cols <= 0) {
      fill_cols({0, n_cols});
      return;
    }
    parallel_for_cpu_ranges(
        cpu_ranges_for_compressed_segments(indptr_ptr, n_cols, workers),
        fill_cols);
  });
}

} // namespace

void CSCToCOOCol::eval_cpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  const auto &indptr = inputs[0];
  if (indptr.dtype() == mx::int32) {
    csc_tocoo_col_cpu_impl<int32_t>(indptr, outputs[0], n_cols_, stream());
    return;
  }
  if (indptr.dtype() == mx::int64) {
    csc_tocoo_col_cpu_impl<int64_t>(indptr, outputs[0], n_cols_, stream());
    return;
  }
  throw std::runtime_error("csc_tocoo requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSCToCOOCol::eval_gpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  const auto &indptr = inputs[0];
  auto &col = outputs[0];
  col.set_data(mx::allocator::malloc(col.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      std::string("csc_tocoo_col_") + index_kernel_suffix(indptr.dtype()), lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(indptr, 0);
  encoder.set_output_array(col, 1);
  encoder.set_bytes(n_cols_, 2);
  const auto threads = std::max<size_t>(static_cast<size_t>(n_cols_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCToCOOCol::eval_gpu(const std::vector<mx::array> &,
                           std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_tocoo has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array> csc_tocoo(const mx::array &data,
                                                      const mx::array &indices,
                                                      const mx::array &indptr,
                                                      int n_rows, int n_cols,
                                                      mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csc_tocoo shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csc_tocoo data");
  require_rank(indices, 1, "csc_tocoo indices");
  require_rank(indptr, 1, "csc_tocoo indptr");
  require_supported_value_dtype(data, "csc_tocoo data");
  require_same_index_dtype(indices, indptr, "csc_tocoo indices",
                           "csc_tocoo indptr");
  require_size(indptr, n_cols + 1, "csc_tocoo indptr");
  if (data.size() != indices.size()) {
    throw std::invalid_argument(
        "csc_tocoo data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto primitive = std::make_shared<CSCToCOOCol>(stream, n_cols);
  auto col = mx::array(mx::Shape{static_cast<int>(data.size())},
                       indices.dtype(), primitive, {indptr_contig});
  return {data, indices, col};
}

} // namespace mlx_sparse
