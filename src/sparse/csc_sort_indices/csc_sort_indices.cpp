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

#include "sparse/csc_sort_indices/csc_sort_indices.h"

#include <algorithm>
#include <numeric>
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

class CSCSortIndices : public mx::Primitive {
public:
  explicit CSCSortIndices(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCSortIndices"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

template <typename T, typename I>
void csc_sort_indices_cpu_impl(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, mx::array &out_data,
                               mx::array &out_indices, mx::array &out_indptr,
                               mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));
  out_indptr.set_data(mx::allocator::malloc(out_indptr.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_indices);
  encoder.set_output_array(out_indptr);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_indices = mx::array::unsafe_weak_copy(out_indices),
                    out_indptr =
                        mx::array::unsafe_weak_copy(out_indptr)]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_indices_ptr = out_indices.data<I>();
    auto *out_indptr_ptr = out_indptr.data<I>();

    std::copy(data_ptr, data_ptr + data.size(), out_data_ptr);
    std::copy(indices_ptr, indices_ptr + indices.size(), out_indices_ptr);
    std::copy(indptr_ptr, indptr_ptr + indptr.size(), out_indptr_ptr);

    const auto n_cols = static_cast<int>(indptr.size()) - 1;
    for (int col = 0; col < n_cols; ++col) {
      const auto start = static_cast<size_t>(indptr_ptr[col]);
      const auto end = static_cast<size_t>(indptr_ptr[col + 1]);
      std::vector<size_t> order(end - start);
      std::iota(order.begin(), order.end(), start);
      std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        return indices_ptr[lhs] < indices_ptr[rhs];
      });
      for (size_t offset = 0; offset < order.size(); ++offset) {
        out_data_ptr[start + offset] = data_ptr[order[offset]];
        out_indices_ptr[start + offset] = indices_ptr[order[offset]];
      }
    }
  });
}

} // namespace

void CSCSortIndices::eval_cpu(const std::vector<mx::array> &inputs,
                              std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error(
        "csc_sort_indices requires int32 or int64 indices.");
  }

#define DISPATCH_CSC_SORT_VALUE(DTYPE, TYPE)                                   \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_sort_indices_cpu_impl<TYPE, int32_t>(data, indices, indptr,          \
                                               outputs[0], outputs[1],         \
                                               outputs[2], stream());          \
    } else {                                                                   \
      csc_sort_indices_cpu_impl<TYPE, int64_t>(data, indices, indptr,          \
                                               outputs[0], outputs[1],         \
                                               outputs[2], stream());          \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_SORT_VALUE(mx::float32, float)
  DISPATCH_CSC_SORT_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSC_SORT_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_SORT_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_SORT_VALUE

  throw std::runtime_error("csc_sort_indices unsupported value dtype.");
}

#ifdef _METAL_
void CSCSortIndices::eval_gpu(const std::vector<mx::array> &inputs,
                              std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out_data = outputs[0];
  auto &out_indices = outputs[1];
  auto &out_indptr = outputs[2];

  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));
  out_indptr.set_data(mx::allocator::malloc(out_indptr.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csc_sort_indices", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_output_array(out_data, 3);
  encoder.set_output_array(out_indices, 4);
  encoder.set_output_array(out_indptr, 5);
  encoder.set_bytes(static_cast<uint32_t>(data.size()), 6);
  encoder.set_bytes(static_cast<uint32_t>(indptr.size()), 7);

  auto threads = std::max<size_t>(std::max(data.size(), indptr.size()), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCSortIndices::eval_gpu(const std::vector<mx::array> &,
                              std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_sort_indices has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
csc_sort_indices(const mx::array &data, const mx::array &indices,
                 const mx::array &indptr, mx::StreamOrDevice s) {
  require_rank(data, 1, "csc_sort_indices data");
  require_rank(indices, 1, "csc_sort_indices indices");
  require_rank(indptr, 1, "csc_sort_indices indptr");
  require_supported_value_dtype(data, "csc_sort_indices data");
  require_same_index_dtype(indices, indptr, "csc_sort_indices indices",
                           "csc_sort_indices indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csc_sort_indices data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);

  auto primitive = std::make_shared<CSCSortIndices>(stream);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{static_cast<int>(data.size())},
       mx::Shape{static_cast<int>(indices.size())},
       mx::Shape{static_cast<int>(indptr.size())}},
      {data.dtype(), indices.dtype(), indptr.dtype()}, primitive,
      {data_contig, indices_contig, indptr_contig});
  return {outputs[0], outputs[1], outputs[2]};
}

} // namespace mlx_sparse
