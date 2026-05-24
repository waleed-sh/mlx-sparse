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

#include "sparse/csc_todense/csc_todense.h"

#include <algorithm>
#include <stdexcept>
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

class CSCToDense : public mx::Primitive {
public:
  CSCToDense(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCToDense"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCToDense &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csc_todense_cpu_impl(const mx::array &data, const mx::array &indices,
                          const mx::array &indptr, mx::array &out, int n_rows,
                          int n_cols, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    n_cols]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *out_ptr = out.data<T>();
    std::fill(out_ptr, out_ptr + out.size(), T{0});

    for (int col = 0; col < n_cols; ++col) {
      for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
        const auto row = static_cast<int>(indices_ptr[p]);
        out_ptr[static_cast<size_t>(row) * n_cols + col] += data_ptr[p];
      }
    }
  });
}

} // namespace

void CSCToDense::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error("csc_todense requires int32 or int64 indices.");
  }

#define DISPATCH_CSC_TODENSE_VALUE(DTYPE, TYPE)                                \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_todense_cpu_impl<TYPE, int32_t>(data, indices, indptr, out, n_rows_, \
                                          n_cols_, stream());                  \
    } else {                                                                   \
      csc_todense_cpu_impl<TYPE, int64_t>(data, indices, indptr, out, n_rows_, \
                                          n_cols_, stream());                  \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_TODENSE_VALUE(mx::float32, float)
  DISPATCH_CSC_TODENSE_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSC_TODENSE_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_TODENSE_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_TODENSE_VALUE

  throw std::runtime_error("csc_todense unsupported value dtype.");
}

#ifdef _METAL_
void CSCToDense::eval_gpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  auto zero_kernel_name =
      std::string("csc_todense_zero_") + value_kernel_suffix(data.dtype());
  auto *zero_kernel = device.get_kernel(zero_kernel_name, lib);
  encoder.set_compute_pipeline_state(zero_kernel);
  encoder.set_output_array(out, 0);
  encoder.set_bytes(static_cast<int>(out.size()), 1);
  auto zero_threads = std::max<size_t>(out.size(), 1);
  auto zero_group =
      std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                           MTL::Size(zero_group, 1, 1));

  auto fill_kernel_name =
      sparse_kernel_name("csc_todense_fill", data.dtype(), indices.dtype());
  auto *fill_kernel = device.get_kernel(fill_kernel_name, lib);
  encoder.set_compute_pipeline_state(fill_kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(n_rows_, 4);
  encoder.set_bytes(n_cols_, 5);
  auto fill_threads = static_cast<size_t>(std::max(n_cols_, 1));
  auto fill_group =
      std::min(fill_threads, fill_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(fill_threads, 1, 1),
                           MTL::Size(fill_group, 1, 1));
}
#else
void CSCToDense::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_todense has no GPU implementation in this build.");
}
#endif

mx::array csc_todense(const mx::array &data, const mx::array &indices,
                      const mx::array &indptr, int n_rows, int n_cols,
                      mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csc_todense shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csc_todense data");
  require_rank(indices, 1, "csc_todense indices");
  require_rank(indptr, 1, "csc_todense indptr");
  require_supported_value_dtype(data, "csc_todense data");
  require_same_index_dtype(indices, indptr, "csc_todense indices",
                           "csc_todense indptr");
  require_size(indptr, n_cols + 1, "csc_todense indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csc_todense data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);

  return mx::array(mx::Shape{n_rows, n_cols}, data.dtype(),
                   std::make_shared<CSCToDense>(stream, n_rows, n_cols),
                   {data_contig, indices_contig, indptr_contig});
}

} // namespace mlx_sparse
