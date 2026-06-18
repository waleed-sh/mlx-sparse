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

#include "sparse/coo_matmul/coo_matmul.h"

#include "sparse/coo_batched_matmul/coo_batched_matmul.h"
#include "sparse/coo_matmul_data_vjp/coo_matmul_data_vjp.h"
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "common/common.h"
#include "common/vmap.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

class COOMatMul : public mx::Primitive {
public:
  COOMatMul(mx::Stream stream, int n_rows, int n_cols, int rhs_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols),
        rhs_cols_(rhs_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array> &primals,
                             const std::vector<mx::array> &tangents,
                             const std::vector<int> &argnums) override;

  std::vector<mx::array> vjp(const std::vector<mx::array> &primals,
                             const std::vector<mx::array> &cotangents,
                             const std::vector<int> &argnums,
                             const std::vector<mx::array> &) override;

  std::pair<std::vector<mx::array>, std::vector<int>>
  vmap(const std::vector<mx::array> &inputs,
       const std::vector<int> &axes) override;

  const char *name() const override { return "COOMatMul"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOMatMul &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           rhs_cols_ == rhs.rhs_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
  int rhs_cols_;
};

template <typename T, typename I>
void coo_matmul_cpu_impl(const mx::array &data, const mx::array &row,
                         const mx::array &col, const mx::array &rhs,
                         mx::array &out, int n_rows, int rhs_cols,
                         mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(row);
  encoder.set_input_array(col);
  encoder.set_input_array(rhs);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    row = mx::array::unsafe_weak_copy(row),
                    col = mx::array::unsafe_weak_copy(col),
                    rhs = mx::array::unsafe_weak_copy(rhs),
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    rhs_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *row_ptr = row.data<I>();
    const auto *col_ptr = col.data<I>();
    const auto *rhs_ptr = rhs.data<T>();
    auto *out_ptr = out.data<T>();
    const size_t out_size = static_cast<size_t>(n_rows) * rhs_cols;

    if constexpr (std::is_same_v<AccT, T>) {
      std::fill(out_ptr, out_ptr + out_size, T{});
      for (size_t p = 0; p < data.size(); ++p) {
        const auto out_offset = static_cast<size_t>(row_ptr[p]) * rhs_cols;
        const auto rhs_offset = static_cast<size_t>(col_ptr[p]) * rhs_cols;
        const T value = data_ptr[p];
        for (int k = 0; k < rhs_cols; ++k) {
          out_ptr[out_offset + k] += value * rhs_ptr[rhs_offset + k];
        }
      }
    } else {
      std::vector<AccT> accum(out_size, Accumulator<T>::zero());
      for (size_t p = 0; p < data.size(); ++p) {
        const auto out_offset = static_cast<size_t>(row_ptr[p]) * rhs_cols;
        const auto rhs_offset = static_cast<size_t>(col_ptr[p]) * rhs_cols;
        const T value = data_ptr[p];
        for (int k = 0; k < rhs_cols; ++k) {
          accum[out_offset + k] +=
              multiply_accumulate<T>(value, rhs_ptr[rhs_offset + k]);
        }
      }
      for (size_t i = 0; i < out_size; ++i) {
        out_ptr[i] = Accumulator<T>::cast(accum[i]);
      }
    }
  });
}

} // namespace

void COOMatMul::eval_cpu(const std::vector<mx::array> &inputs,
                         std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  auto &rhs = inputs[3];
  auto &out = outputs[0];

  if (row.dtype() != mx::int32 && row.dtype() != mx::int64) {
    throw std::runtime_error("coo_matmul requires int32 or int64 indices.");
  }

#define DISPATCH_COO_MATMUL_VALUE(DTYPE, TYPE)                                 \
  if (data.dtype() == DTYPE) {                                                 \
    if (row.dtype() == mx::int32) {                                            \
      coo_matmul_cpu_impl<TYPE, int32_t>(data, row, col, rhs, out, n_rows_,    \
                                         rhs_cols_, stream());                 \
    } else {                                                                   \
      coo_matmul_cpu_impl<TYPE, int64_t>(data, row, col, rhs, out, n_rows_,    \
                                         rhs_cols_, stream());                 \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_MATMUL_VALUE(mx::float32, float)
  DISPATCH_COO_MATMUL_VALUE(mx::float16, mx::float16_t)
  DISPATCH_COO_MATMUL_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_MATMUL_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_MATMUL_VALUE

  throw std::runtime_error("coo_matmul unsupported value dtype.");
}

#ifdef _METAL_
void COOMatMul::eval_gpu(const std::vector<mx::array> &inputs,
                         std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  auto &rhs = inputs[3];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  if (data.dtype() == mx::float32) {
    auto *zero_kernel = device.get_kernel("coo_matmul_zero_float32", lib);
    encoder.set_compute_pipeline_state(zero_kernel);
    encoder.set_output_array(out, 0);
    auto out_size = static_cast<int>(out.size());
    encoder.set_bytes(out_size, 1);
    auto zero_threads = std::max<size_t>(out.size(), 1);
    auto zero_group =
        std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                             MTL::Size(zero_group, 1, 1));

    auto kernel_name =
        std::string("coo_matmul_atomic_") + index_kernel_suffix(row.dtype());
    auto *kernel = device.get_kernel(kernel_name, lib);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(data, 0);
    encoder.set_input_array(row, 1);
    encoder.set_input_array(col, 2);
    encoder.set_input_array(rhs, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(rhs_cols_, 5);
    auto total = static_cast<int>(data.size() * rhs_cols_);
    encoder.set_bytes(total, 6);
    auto threads = std::max<size_t>(data.size() * rhs_cols_, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
    return;
  }

  auto kernel_name =
      sparse_kernel_name("coo_matmul_serial", data.dtype(), row.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(row, 1);
  encoder.set_input_array(col, 2);
  encoder.set_input_array(rhs, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(rhs_cols_, 6);
  auto nnz = static_cast<int>(data.size());
  encoder.set_bytes(nnz, 7);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
#else
void COOMatMul::eval_gpu(const std::vector<mx::array> &,
                         std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_matmul has no GPU implementation in this build.");
}
#endif

std::vector<mx::array> COOMatMul::jvp(const std::vector<mx::array> &primals,
                                      const std::vector<mx::array> &tangents,
                                      const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    if (argnums[i] == 0) {
      terms.push_back(coo_matmul(tangents[i], primals[1], primals[2],
                                 primals[3], n_rows_, n_cols_, stream()));
    } else if (argnums[i] == 3) {
      terms.push_back(coo_matmul(primals[0], primals[1], primals[2],
                                 tangents[i], n_rows_, n_cols_, stream()));
    } else {
      throw std::runtime_error(
          "COOMatMul JVP is implemented only for data and dense RHS.");
    }
  }
  if (terms.empty()) {
    throw std::runtime_error("COOMatMul JVP requires at least one tangent.");
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array> COOMatMul::vjp(const std::vector<mx::array> &primals,
                                      const std::vector<mx::array> &cotangents,
                                      const std::vector<int> &argnums,
                                      const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  for (int argnum : argnums) {
    if (argnum == 0) {
      auto rhs = primals[3].dtype() == mx::complex64
                     ? mx::conjugate(primals[3], stream())
                     : primals[3];
      vjps.push_back(coo_matmul_data_vjp(primals[1], primals[2], rhs,
                                         cotangents[0], n_rows_, n_cols_,
                                         stream()));
    } else if (argnum == 3) {
      auto data = primals[0].dtype() == mx::complex64
                      ? mx::conjugate(primals[0], stream())
                      : primals[0];
      vjps.push_back(coo_matmul(data, primals[2], primals[1], cotangents[0],
                                n_cols_, n_rows_, stream()));
    } else {
      throw std::runtime_error(
          "COOMatMul VJP is implemented only for data and dense RHS.");
    }
  }
  return vjps;
}

std::pair<std::vector<mx::array>, std::vector<int>>
COOMatMul::vmap(const std::vector<mx::array> &inputs,
                const std::vector<int> &axes) {
  require_vmap_arity(inputs, axes, 4, "COOMatMul");
  require_fixed_sparse_vmap_axes(axes, 3, "COOMatMul");

  auto rhs =
      dense_rhs_with_vmap_axis_front(inputs[3], axes[3], stream(), "COOMatMul");
  require_vmap_rhs_rank(rhs, 3, "COOMatMul");
  require_vmap_rhs_sparse_dim(rhs, 1, n_cols_, "COOMatMul");
  require_vmap_rhs_dim(rhs, 2, rhs_cols_, "dense RHS column dimension",
                       "COOMatMul");

  return {{coo_batched_matmul(inputs[0], inputs[1], inputs[2], rhs, n_rows_,
                              n_cols_, stream())},
          {0}};
}

mx::array coo_matmul(const mx::array &data, const mx::array &row,
                     const mx::array &col, const mx::array &rhs, int n_rows,
                     int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "coo_matmul shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "coo_matmul data");
  require_rank(row, 1, "coo_matmul row");
  require_rank(col, 1, "coo_matmul col");
  require_rank(rhs, 2, "coo_matmul rhs");
  require_same_value_dtype(data, rhs, "coo_matmul data", "coo_matmul rhs");
  require_same_index_dtype(row, col, "coo_matmul row", "coo_matmul col");
  if (row.size() != data.size() || col.size() != data.size()) {
    throw std::invalid_argument(
        "coo_matmul data, row, and col must have equal length.");
  }
  if (rhs.shape(0) != n_cols) {
    throw std::invalid_argument("coo_matmul rhs first dimension must equal "
                                "the sparse matrix column count.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto row_contig = mx::contiguous(row, false, stream);
  auto col_contig = mx::contiguous(col, false, stream);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  const int rhs_cols = rhs.shape(1);

  return mx::array(
      mx::Shape{n_rows, rhs_cols}, data.dtype(),
      std::make_shared<COOMatMul>(stream, n_rows, n_cols, rhs_cols),
      {data_contig, row_contig, col_contig, rhs_contig});
}

mx::array coo_matvec(const mx::array &data, const mx::array &row,
                     const mx::array &col, const mx::array &x, int n_rows,
                     int n_cols, mx::StreamOrDevice s) {
  require_rank(x, 1, "coo_matvec x");
  require_size(x, n_cols, "coo_matvec x");
  auto stream = mx::to_stream(s);
  auto x_matrix = mx::reshape(x, mx::Shape{n_cols, 1}, stream);
  auto out = coo_matmul(data, row, col, x_matrix, n_rows, n_cols, stream);
  return mx::reshape(out, mx::Shape{n_rows}, stream);
}

} // namespace mlx_sparse
