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

#include "sparse/coo_col_norms/coo_col_norms.h"

#include <algorithm>
#include <cmath>
#include <complex>
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

class COOColNorms : public mx::Primitive {
public:
  COOColNorms(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOColNorms"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOColNorms &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T> double norm_square(T value) {
  if constexpr (std::is_same_v<T, mx::complex64_t>) {
    const std::complex<float> z(value);
    return static_cast<double>(std::norm(z));
  } else {
    const double x = static_cast<double>(value);
    return x * x;
  }
}

template <typename T, typename I>
void coo_col_norms_cpu_impl(const mx::array &data, const mx::array &col,
                            mx::array &out, int n_cols, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(col);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    col = mx::array::unsafe_weak_copy(col),
                    out = mx::array::unsafe_weak_copy(out), n_cols]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *col_ptr = col.data<I>();
    std::vector<double> accum(static_cast<size_t>(n_cols), 0.0);

    for (size_t p = 0; p < data.size(); ++p) {
      accum[static_cast<size_t>(col_ptr[p])] += norm_square<T>(data_ptr[p]);
    }

    auto *out_ptr = out.data<float>();
    for (int c = 0; c < n_cols; ++c) {
      out_ptr[c] = static_cast<float>(std::sqrt(accum[static_cast<size_t>(c)]));
    }
  });
}

void validate_coo_reduction_inputs(const mx::array &data, const mx::array &row,
                                   const mx::array &col, int n_rows, int n_cols,
                                   const char *op) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(std::string(op) +
                                " shape dimensions must be non-negative.");
  }
  require_rank(data, 1, op);
  require_rank(row, 1, op);
  require_rank(col, 1, op);
  require_supported_value_dtype(data, op);
  require_same_index_dtype(row, col, op, op);
  if (row.size() != data.size() || col.size() != data.size()) {
    throw std::invalid_argument(std::string(op) +
                                " data, row, and col must have equal length.");
  }
}

} // namespace

void COOColNorms::eval_cpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &col = inputs[2];

#define DISPATCH_COO_COL_NORMS(DTYPE, TYPE)                                    \
  if (data.dtype() == DTYPE) {                                                 \
    if (col.dtype() == mx::int32) {                                            \
      coo_col_norms_cpu_impl<TYPE, int32_t>(data, col, outputs[0], n_cols_,    \
                                            stream());                         \
    } else {                                                                   \
      coo_col_norms_cpu_impl<TYPE, int64_t>(data, col, outputs[0], n_cols_,    \
                                            stream());                         \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_COL_NORMS(mx::float32, float)
  DISPATCH_COO_COL_NORMS(mx::float16, mx::float16_t)
  DISPATCH_COO_COL_NORMS(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_COL_NORMS(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_COL_NORMS

  throw std::runtime_error("coo_col_norms unsupported value dtype.");
}

#ifdef _METAL_
void COOColNorms::eval_gpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &col = inputs[2];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  auto *zero_kernel = device.get_kernel("coo_col_norms_zero_float32", lib);
  encoder.set_compute_pipeline_state(zero_kernel);
  encoder.set_output_array(out, 0);
  encoder.set_bytes(n_cols_, 1);
  auto zero_threads = static_cast<size_t>(std::max(n_cols_, 1));
  auto zero_group =
      std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                           MTL::Size(zero_group, 1, 1));

  auto kernel_name =
      sparse_kernel_name("coo_col_norms_atomic", data.dtype(), col.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(col, 1);
  encoder.set_output_array(out, 2);
  auto nnz = static_cast<int>(data.size());
  encoder.set_bytes(nnz, 3);
  encoder.set_bytes(n_cols_, 4);
  auto threads = std::max<size_t>(data.size(), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));

  auto *sqrt_kernel = device.get_kernel("coo_col_norms_sqrt_float32", lib);
  encoder.set_compute_pipeline_state(sqrt_kernel);
  encoder.set_output_array(out, 0);
  encoder.set_bytes(n_cols_, 1);
  encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                           MTL::Size(zero_group, 1, 1));
}
#else
void COOColNorms::eval_gpu(const std::vector<mx::array> &,
                           std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_col_norms has no GPU implementation in this build.");
}
#endif

mx::array coo_col_norms(const mx::array &data, const mx::array &row,
                        const mx::array &col, int n_rows, int n_cols,
                        mx::StreamOrDevice s) {
  validate_coo_reduction_inputs(data, row, col, n_rows, n_cols,
                                "coo_col_norms");

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto row_contig = mx::contiguous(row, false, stream);
  auto col_contig = mx::contiguous(col, false, stream);

  return mx::array(mx::Shape{n_cols}, mx::float32,
                   std::make_shared<COOColNorms>(stream, n_rows, n_cols),
                   {data_contig, row_contig, col_contig});
}

} // namespace mlx_sparse
