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

#include "sparse/csr_matvec_transpose/csr_matvec_transpose.h"

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

class CSRMatVecTranspose : public mx::Primitive {
public:
  CSRMatVecTranspose(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRMatVecTranspose"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRMatVecTranspose &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csr_matvec_transpose_cpu_impl(const mx::array &data,
                                   const mx::array &indices,
                                   const mx::array &indptr, const mx::array &x,
                                   mx::array &out, int n_rows, int n_cols,
                                   mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(x);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    x = mx::array::unsafe_weak_copy(x),
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    n_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *x_ptr = x.data<T>();
    auto *out_ptr = out.data<T>();

    if constexpr (std::is_same_v<AccT, T>) {
      std::fill(out_ptr, out_ptr + n_cols, T{});
      for (int row = 0; row < n_rows; ++row) {
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          out_ptr[indices_ptr[p]] += data_ptr[p] * x_ptr[row];
        }
      }
    } else {
      std::vector<AccT> accum(static_cast<size_t>(n_cols),
                              Accumulator<T>::zero());
      for (int row = 0; row < n_rows; ++row) {
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          accum[static_cast<size_t>(indices_ptr[p])] +=
              multiply_accumulate<T>(data_ptr[p], x_ptr[row]);
        }
      }
      for (int col = 0; col < n_cols; ++col) {
        out_ptr[col] = Accumulator<T>::cast(accum[static_cast<size_t>(col)]);
      }
    }
  });
}

} // namespace

void CSRMatVecTranspose::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &x = inputs[3];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error(
        "csr_matvec_transpose requires int32 or int64 indices.");
  }

#define DISPATCH_CSR_MATVEC_T_VALUE(DTYPE, TYPE)                               \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csr_matvec_transpose_cpu_impl<TYPE, int32_t>(                            \
          data, indices, indptr, x, out, n_rows_, n_cols_, stream());          \
    } else {                                                                   \
      csr_matvec_transpose_cpu_impl<TYPE, int64_t>(                            \
          data, indices, indptr, x, out, n_rows_, n_cols_, stream());          \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_MATVEC_T_VALUE(mx::float32, float)
  DISPATCH_CSR_MATVEC_T_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_MATVEC_T_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_MATVEC_T_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_MATVEC_T_VALUE

  throw std::runtime_error("csr_matvec_transpose unsupported value dtype.");
}

#ifdef _METAL_
void CSRMatVecTranspose::eval_gpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &x = inputs[3];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_matvec_transpose", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(x, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(n_cols_, 6);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
#else
void CSRMatVecTranspose::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_matvec_transpose has no GPU implementation in this build.");
}
#endif

mx::array csr_matvec_transpose(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, const mx::array &x,
                               int n_rows, int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr_matvec_transpose shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csr_matvec_transpose data");
  require_rank(indices, 1, "csr_matvec_transpose indices");
  require_rank(indptr, 1, "csr_matvec_transpose indptr");
  require_rank(x, 1, "csr_matvec_transpose x");
  require_same_value_dtype(data, x, "csr_matvec_transpose data",
                           "csr_matvec_transpose x");
  require_same_index_dtype(indices, indptr, "csr_matvec_transpose indices",
                           "csr_matvec_transpose indptr");
  require_size(indptr, n_rows + 1, "csr_matvec_transpose indptr");
  require_size(x, n_rows, "csr_matvec_transpose x");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_matvec_transpose data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto x_contig = mx::contiguous(x, false, stream);

  return mx::array(mx::Shape{n_cols}, data.dtype(),
                   std::make_shared<CSRMatVecTranspose>(stream, n_rows, n_cols),
                   {data_contig, indices_contig, indptr_contig, x_contig});
}

} // namespace mlx_sparse
