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

#include "sparse/csc_trace/csc_trace.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

constexpr size_t kTraceThreads = 128;
constexpr int kTraceColsPerBlock = 512;

mx::Dtype trace_accumulator_dtype(mx::Dtype dtype) {
  if (dtype == mx::float16 || dtype == mx::bfloat16) {
    return mx::float32;
  }
  return dtype;
}

class CSCTrace : public mx::Primitive {
public:
  CSCTrace(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCTrace"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCTrace &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csc_trace_cpu_impl(const mx::array &data, const mx::array &indices,
                        const mx::array &indptr, mx::array &out, int diag_size,
                        mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out = mx::array::unsafe_weak_copy(out),
                    diag_size]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto acc = Accumulator<T>::zero();

    for (int col = 0; col < diag_size; ++col) {
      for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
        if (indices_ptr[p] == static_cast<I>(col)) {
          acc += static_cast<typename Accumulator<T>::Type>(data_ptr[p]);
        }
      }
    }
    *out.data<T>() = Accumulator<T>::cast(acc);
  });
}

void validate_csc_reduction_inputs(const mx::array &data,
                                   const mx::array &indices,
                                   const mx::array &indptr, int n_rows,
                                   int n_cols, const char *op) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(std::string(op) +
                                " shape dimensions must be non-negative.");
  }
  require_rank(data, 1, op);
  require_rank(indices, 1, op);
  require_rank(indptr, 1, op);
  require_supported_value_dtype(data, op);
  require_same_index_dtype(indices, indptr, op, op);
  require_size(indptr, n_cols + 1, op);
  if (indices.size() != data.size()) {
    throw std::invalid_argument(std::string(op) +
                                " data and indices must have equal length.");
  }
}

} // namespace

void CSCTrace::eval_cpu(const std::vector<mx::array> &inputs,
                        std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  const int diag_size = std::min(n_rows_, n_cols_);

#define DISPATCH_CSC_TRACE(DTYPE, TYPE)                                        \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_trace_cpu_impl<TYPE, int32_t>(data, indices, indptr, outputs[0],     \
                                        diag_size, stream());                  \
    } else {                                                                   \
      csc_trace_cpu_impl<TYPE, int64_t>(data, indices, indptr, outputs[0],     \
                                        diag_size, stream());                  \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_TRACE(mx::float32, float)
  DISPATCH_CSC_TRACE(mx::float16, mx::float16_t)
  DISPATCH_CSC_TRACE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_TRACE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_TRACE

  throw std::runtime_error("csc_trace unsupported value dtype.");
}

#ifdef _METAL_
void CSCTrace::eval_gpu(const std::vector<mx::array> &inputs,
                        std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out = outputs[0];
  const int diag_size = std::min(n_rows_, n_cols_);

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  const int num_blocks =
      (diag_size + kTraceColsPerBlock - 1) / kTraceColsPerBlock;
  if (num_blocks > 1) {
    const auto partial_dtype = trace_accumulator_dtype(data.dtype());
    mx::array partials(mx::allocator::malloc(static_cast<size_t>(num_blocks) *
                                             mx::size_of(partial_dtype)),
                       mx::Shape{num_blocks}, partial_dtype);

    auto blocks_kernel_name =
        sparse_kernel_name("csc_trace_blocks", data.dtype(), indices.dtype());
    auto *blocks_kernel = device.get_kernel(blocks_kernel_name, lib);
    encoder.set_compute_pipeline_state(blocks_kernel);
    encoder.set_input_array(data, 0);
    encoder.set_input_array(indices, 1);
    encoder.set_input_array(indptr, 2);
    encoder.set_output_array(partials, 3);
    encoder.set_bytes(diag_size, 4);
    encoder.set_bytes(kTraceColsPerBlock, 5);
    encoder.dispatch_threads(
        MTL::Size(static_cast<size_t>(num_blocks) * kTraceThreads, 1, 1),
        MTL::Size(kTraceThreads, 1, 1));

    auto finalize_kernel_name =
        std::string("csc_trace_finalize_") + value_kernel_suffix(data.dtype());
    auto *finalize_kernel = device.get_kernel(finalize_kernel_name, lib);
    encoder.set_compute_pipeline_state(finalize_kernel);
    encoder.set_input_array(partials, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(num_blocks, 2);
    encoder.dispatch_threads(MTL::Size(kTraceThreads, 1, 1),
                             MTL::Size(kTraceThreads, 1, 1));
    encoder.add_temporary(std::move(partials));
    return;
  }

  auto kernel_name =
      sparse_kernel_name("csc_trace", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(diag_size, 4);
  encoder.dispatch_threads(MTL::Size(128, 1, 1), MTL::Size(128, 1, 1));
}
#else
void CSCTrace::eval_gpu(const std::vector<mx::array> &,
                        std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_trace has no GPU implementation in this build.");
}
#endif

mx::array csc_trace(const mx::array &data, const mx::array &indices,
                    const mx::array &indptr, int n_rows, int n_cols,
                    mx::StreamOrDevice s) {
  validate_csc_reduction_inputs(data, indices, indptr, n_rows, n_cols,
                                "csc_trace");

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);

  return mx::array(mx::Shape{}, data.dtype(),
                   std::make_shared<CSCTrace>(stream, n_rows, n_cols),
                   {data_contig, indices_contig, indptr_contig});
}

} // namespace mlx_sparse
