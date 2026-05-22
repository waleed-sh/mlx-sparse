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

#include "sparse/csr_todense/csr_todense.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "common/common.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

class CSRToDense : public mx::Primitive {
public:
  CSRToDense(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRToDense"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRToDense &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csr_todense_cpu_impl(const mx::array &data, const mx::array &indices,
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

    for (int row = 0; row < n_rows; ++row) {
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const auto col = static_cast<int>(indices_ptr[p]);
        out_ptr[static_cast<size_t>(row) * n_cols + col] += data_ptr[p];
      }
    }
  });
}

} // namespace

void CSRToDense::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error("csr_todense requires int32 or int64 indices.");
  }

#define DISPATCH_CSR_TODENSE_VALUE(DTYPE, TYPE)                                \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csr_todense_cpu_impl<TYPE, int32_t>(data, indices, indptr, out, n_rows_, \
                                          n_cols_, stream());                  \
    } else {                                                                   \
      csr_todense_cpu_impl<TYPE, int64_t>(data, indices, indptr, out, n_rows_, \
                                          n_cols_, stream());                  \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_TODENSE_VALUE(mx::float32, float)
  DISPATCH_CSR_TODENSE_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_TODENSE_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_TODENSE_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_TODENSE_VALUE

  throw std::runtime_error("csr_todense unsupported value dtype.");
}

#ifdef _METAL_
void CSRToDense::eval_gpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_todense", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(static_cast<uint32_t>(n_rows_), 4);
  encoder.set_bytes(static_cast<uint32_t>(n_cols_), 5);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
#else
void CSRToDense::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_todense has no GPU implementation in this build.");
}
#endif

mx::array csr_todense(const mx::array &data, const mx::array &indices,
                      const mx::array &indptr, int n_rows, int n_cols,
                      mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr_todense shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csr_todense data");
  require_rank(indices, 1, "csr_todense indices");
  require_rank(indptr, 1, "csr_todense indptr");
  require_supported_value_dtype(data, "csr_todense data");
  require_same_index_dtype(indices, indptr, "csr_todense indices",
                           "csr_todense indptr");
  require_size(indptr, n_rows + 1, "csr_todense indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_todense data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);

  return mx::array(mx::Shape{n_rows, n_cols}, data.dtype(),
                   std::make_shared<CSRToDense>(stream, n_rows, n_cols),
                   {data_contig, indices_contig, indptr_contig});
}

} // namespace mlx_sparse
