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

#include "sparse/csc_row_sums/csc_row_sums.h"

#include "sparse/csc_tocsr/csc_tocsr.h"
#include "sparse/csr_row_sums/csr_row_sums.h"

#include <algorithm>
#include <stdexcept>
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

class CSCRowSums : public mx::Primitive {
public:
  CSCRowSums(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCRowSums"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCRowSums &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csc_row_sums_cpu_impl(const mx::array &data, const mx::array &indices,
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
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *out_ptr = out.data<T>();

    if constexpr (std::is_same_v<AccT, T>) {
      std::fill(out_ptr, out_ptr + n_rows, T{});
      for (int col = 0; col < n_cols; ++col) {
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          out_ptr[indices_ptr[p]] += data_ptr[p];
        }
      }
    } else {
      std::vector<AccT> accum(static_cast<size_t>(n_rows),
                              Accumulator<T>::zero());
      for (int col = 0; col < n_cols; ++col) {
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          accum[static_cast<size_t>(indices_ptr[p])] +=
              static_cast<AccT>(data_ptr[p]);
        }
      }
      for (int row = 0; row < n_rows; ++row) {
        out_ptr[row] = Accumulator<T>::cast(accum[static_cast<size_t>(row)]);
      }
    }
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

void CSCRowSums::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];

#define DISPATCH_CSC_ROW_SUMS(DTYPE, TYPE)                                     \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_row_sums_cpu_impl<TYPE, int32_t>(data, indices, indptr, outputs[0],  \
                                           n_rows_, n_cols_, stream());        \
    } else {                                                                   \
      csc_row_sums_cpu_impl<TYPE, int64_t>(data, indices, indptr, outputs[0],  \
                                           n_rows_, n_cols_, stream());        \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_ROW_SUMS(mx::float32, float)
  DISPATCH_CSC_ROW_SUMS(mx::float16, mx::float16_t)
  DISPATCH_CSC_ROW_SUMS(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_ROW_SUMS(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_ROW_SUMS

  throw std::runtime_error("csc_row_sums unsupported value dtype.");
}

#ifdef _METAL_
void CSCRowSums::eval_gpu(const std::vector<mx::array> &inputs,
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

  auto *zero_kernel = device.get_kernel("csc_row_sums_zero_float32", lib);
  encoder.set_compute_pipeline_state(zero_kernel);
  encoder.set_output_array(out, 0);
  encoder.set_bytes(n_rows_, 1);
  auto zero_threads = static_cast<size_t>(std::max(n_rows_, 1));
  auto zero_group =
      std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                           MTL::Size(zero_group, 1, 1));

  auto atomic_kernel_name = std::string("csc_row_sums_atomic_") +
                            index_kernel_suffix(indices.dtype());
  auto *kernel = device.get_kernel(atomic_kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(n_rows_, 4);
  encoder.set_bytes(n_cols_, 5);
  auto threads = static_cast<size_t>(std::max(n_cols_, 1));
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCRowSums::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_row_sums has no GPU implementation in this build.");
}
#endif

mx::array csc_row_sums(const mx::array &data, const mx::array &indices,
                       const mx::array &indptr, int n_rows, int n_cols,
                       mx::StreamOrDevice s) {
  validate_csc_reduction_inputs(data, indices, indptr, n_rows, n_cols,
                                "csc_row_sums");

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);

  if (stream.device == mx::Device::gpu && data.dtype() != mx::float32) {
    auto [csr_data, csr_indices, csr_indptr] = csc_tocsr(
        data_contig, indices_contig, indptr_contig, n_rows, n_cols, stream);
    return csr_row_sums(csr_data, csr_indices, csr_indptr, n_rows, n_cols,
                        stream);
  }

  return mx::array(mx::Shape{n_rows}, data.dtype(),
                   std::make_shared<CSCRowSums>(stream, n_rows, n_cols),
                   {data_contig, indices_contig, indptr_contig});
}

} // namespace mlx_sparse
