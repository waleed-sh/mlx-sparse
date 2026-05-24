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

#include "sparse/csr_tocsc/csr_tocsc.h"

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

class CSRToCSC : public mx::Primitive {
public:
  CSRToCSC(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRToCSC"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRToCSC &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csr_tocsc_cpu_impl(const mx::array &data, const mx::array &indices,
                        const mx::array &indptr, mx::array &out_data,
                        mx::array &out_indices, mx::array &out_indptr,
                        int n_rows, int n_cols, mx::Stream stream) {
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
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    n_rows, n_cols]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_indices_ptr = out_indices.data<I>();
    auto *out_indptr_ptr = out_indptr.data<I>();
    const auto nnz = data.size();

    std::fill(out_indptr_ptr, out_indptr_ptr + n_cols + 1, I{0});
    for (size_t p = 0; p < nnz; ++p) {
      out_indptr_ptr[static_cast<size_t>(indices_ptr[p]) + 1] += I{1};
    }
    for (int col = 0; col < n_cols; ++col) {
      out_indptr_ptr[col + 1] += out_indptr_ptr[col];
    }

    std::vector<I> next(out_indptr_ptr, out_indptr_ptr + n_cols);
    for (int row = 0; row < n_rows; ++row) {
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const auto col = static_cast<size_t>(indices_ptr[p]);
        const auto dst = static_cast<size_t>(next[col]++);
        out_data_ptr[dst] = data_ptr[p];
        out_indices_ptr[dst] = static_cast<I>(row);
      }
    }
  });
}

} // namespace

void CSRToCSC::eval_cpu(const std::vector<mx::array> &inputs,
                        std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error("csr_tocsc requires int32 or int64 indices.");
  }

#define DISPATCH_CSR_TO_CSC_VALUE(DTYPE, TYPE)                                 \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csr_tocsc_cpu_impl<TYPE, int32_t>(data, indices, indptr, outputs[0],     \
                                        outputs[1], outputs[2], n_rows_,       \
                                        n_cols_, stream());                    \
    } else {                                                                   \
      csr_tocsc_cpu_impl<TYPE, int64_t>(data, indices, indptr, outputs[0],     \
                                        outputs[1], outputs[2], n_rows_,       \
                                        n_cols_, stream());                    \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_TO_CSC_VALUE(mx::float32, float)
  DISPATCH_CSR_TO_CSC_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_TO_CSC_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_TO_CSC_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_TO_CSC_VALUE

  throw std::runtime_error("csr_tocsc unsupported value dtype.");
}

#ifdef _METAL_
void CSRToCSC::eval_gpu(const std::vector<mx::array> &inputs,
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
  auto &encoder = mx::metal::get_command_encoder(s);

  mx::array counts(
      mx::allocator::malloc(static_cast<size_t>(n_cols_ + 1) * sizeof(int32_t)),
      mx::Shape{n_cols_ + 1}, mx::int32);
  mx::array next(
      mx::allocator::malloc(static_cast<size_t>(n_cols_ + 1) * sizeof(int32_t)),
      mx::Shape{n_cols_ + 1}, mx::int32);

  auto *zero_kernel = device.get_kernel("csr_tocsc_zero_offsets", lib);
  encoder.set_compute_pipeline_state(zero_kernel);
  encoder.set_output_array(counts, 0);
  encoder.set_bytes(n_cols_, 1);
  auto zero_threads = static_cast<size_t>(std::max(n_cols_ + 1, 1));
  auto zero_group =
      std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                           MTL::Size(zero_group, 1, 1));

  auto count_kernel_name =
      std::string("csr_tocsc_count_") + index_kernel_suffix(indices.dtype());
  auto *count_kernel = device.get_kernel(count_kernel_name, lib);
  encoder.set_compute_pipeline_state(count_kernel);
  encoder.set_input_array(indices, 0);
  encoder.set_output_array(counts, 1);
  encoder.set_bytes(static_cast<int>(data.size()), 2);
  auto count_threads = std::max<size_t>(data.size(), 1);
  auto count_group =
      std::min(count_threads, count_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(count_threads, 1, 1),
                           MTL::Size(count_group, 1, 1));

  auto prefix_kernel_name =
      std::string("csr_tocsc_prefix_") + index_kernel_suffix(indices.dtype());
  auto *prefix_kernel = device.get_kernel(prefix_kernel_name, lib);
  encoder.set_compute_pipeline_state(prefix_kernel);
  encoder.set_input_array(counts, 0);
  encoder.set_output_array(next, 1);
  encoder.set_output_array(out_indptr, 2);
  encoder.set_bytes(n_cols_, 3);
  encoder.set_bytes(static_cast<int>(data.size()), 4);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

  auto fill_kernel_name =
      sparse_kernel_name("csr_tocsc_fill", data.dtype(), indices.dtype());
  auto *fill_kernel = device.get_kernel(fill_kernel_name, lib);
  encoder.set_compute_pipeline_state(fill_kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(next, 3);
  encoder.set_output_array(out_data, 4);
  encoder.set_output_array(out_indices, 5);
  encoder.set_bytes(n_rows_, 6);
  auto fill_threads = static_cast<size_t>(std::max(n_rows_, 1));
  auto fill_group =
      std::min(fill_threads, fill_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(fill_threads, 1, 1),
                           MTL::Size(fill_group, 1, 1));

  encoder.add_temporary(std::move(counts));
  encoder.add_temporary(std::move(next));
}
#else
void CSRToCSC::eval_gpu(const std::vector<mx::array> &,
                        std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_tocsc has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array> csr_tocsc(const mx::array &data,
                                                      const mx::array &indices,
                                                      const mx::array &indptr,
                                                      int n_rows, int n_cols,
                                                      mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr_tocsc shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csr_tocsc data");
  require_rank(indices, 1, "csr_tocsc indices");
  require_rank(indptr, 1, "csr_tocsc indptr");
  require_supported_value_dtype(data, "csr_tocsc data");
  require_same_index_dtype(indices, indptr, "csr_tocsc indices",
                           "csr_tocsc indptr");
  require_size(indptr, n_rows + 1, "csr_tocsc indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_tocsc data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);

  auto primitive = std::make_shared<CSRToCSC>(stream, n_rows, n_cols);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{static_cast<int>(data.size())},
       mx::Shape{static_cast<int>(indices.size())}, mx::Shape{n_cols + 1}},
      {data.dtype(), indices.dtype(), indptr.dtype()}, primitive,
      {data_contig, indices_contig, indptr_contig});
  return {outputs[0], outputs[1], outputs[2]};
}

} // namespace mlx_sparse
