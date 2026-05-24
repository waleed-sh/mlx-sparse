// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "sparse/coo_batched_matmul/coo_batched_matmul.h"

#include "sparse/coo_matmul/coo_matmul.h"
#include "sparse/coo_matmul_data_vjp/coo_matmul_data_vjp.h"
#include <algorithm>
#include <stdexcept>
#include <type_traits>
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

class COOBatchedMatMul : public mx::Primitive {
public:
  COOBatchedMatMul(mx::Stream stream, int n_rows, int n_cols, int batch_size,
                   int rhs_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols),
        batch_size_(batch_size), rhs_cols_(rhs_cols) {}

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

  const char *name() const override { return "COOBatchedMatMul"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOBatchedMatMul &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           batch_size_ == rhs.batch_size_ && rhs_cols_ == rhs.rhs_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
  int batch_size_;
  int rhs_cols_;
};

template <typename T, typename I>
void coo_batched_matmul_cpu_impl(const mx::array &data, const mx::array &row,
                                 const mx::array &col, const mx::array &rhs,
                                 mx::array &out, int n_rows, int n_cols,
                                 int batch_size, int rhs_cols,
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
                    out = mx::array::unsafe_weak_copy(out), n_rows, n_cols,
                    batch_size, rhs_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *row_ptr = row.data<I>();
    const auto *col_ptr = col.data<I>();
    const auto *rhs_ptr = rhs.data<T>();
    auto *out_ptr = out.data<T>();
    const size_t per_batch_out = static_cast<size_t>(n_rows) * rhs_cols;
    const size_t per_batch_rhs = static_cast<size_t>(n_cols) * rhs_cols;

    if constexpr (std::is_same_v<AccT, T>) {
      std::fill(out_ptr, out_ptr + out.size(), T{});
      for (int batch = 0; batch < batch_size; ++batch) {
        const auto out_batch = static_cast<size_t>(batch) * per_batch_out;
        const auto rhs_batch = static_cast<size_t>(batch) * per_batch_rhs;
        for (size_t p = 0; p < data.size(); ++p) {
          const auto out_offset =
              out_batch + static_cast<size_t>(row_ptr[p]) * rhs_cols;
          const auto rhs_offset =
              rhs_batch + static_cast<size_t>(col_ptr[p]) * rhs_cols;
          const T value = data_ptr[p];
          for (int k = 0; k < rhs_cols; ++k) {
            out_ptr[out_offset + k] += value * rhs_ptr[rhs_offset + k];
          }
        }
      }
    } else {
      std::vector<AccT> accum(out.size(), Accumulator<T>::zero());
      for (int batch = 0; batch < batch_size; ++batch) {
        const auto out_batch = static_cast<size_t>(batch) * per_batch_out;
        const auto rhs_batch = static_cast<size_t>(batch) * per_batch_rhs;
        for (size_t p = 0; p < data.size(); ++p) {
          const auto out_offset =
              out_batch + static_cast<size_t>(row_ptr[p]) * rhs_cols;
          const auto rhs_offset =
              rhs_batch + static_cast<size_t>(col_ptr[p]) * rhs_cols;
          const T value = data_ptr[p];
          for (int k = 0; k < rhs_cols; ++k) {
            accum[out_offset + k] +=
                multiply_accumulate<T>(value, rhs_ptr[rhs_offset + k]);
          }
        }
      }
      for (size_t i = 0; i < out.size(); ++i) {
        out_ptr[i] = Accumulator<T>::cast(accum[i]);
      }
    }
  });
}

} // namespace

void COOBatchedMatMul::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  auto &rhs = inputs[3];
  auto &out = outputs[0];

  if (row.dtype() != mx::int32 && row.dtype() != mx::int64) {
    throw std::runtime_error(
        "coo_batched_matmul requires int32 or int64 indices.");
  }

#define DISPATCH_COO_BATCHED_MATMUL_VALUE(DTYPE, TYPE)                         \
  if (data.dtype() == DTYPE) {                                                 \
    if (row.dtype() == mx::int32) {                                            \
      coo_batched_matmul_cpu_impl<TYPE, int32_t>(                              \
          data, row, col, rhs, out, n_rows_, n_cols_, batch_size_, rhs_cols_,  \
          stream());                                                           \
    } else {                                                                   \
      coo_batched_matmul_cpu_impl<TYPE, int64_t>(                              \
          data, row, col, rhs, out, n_rows_, n_cols_, batch_size_, rhs_cols_,  \
          stream());                                                           \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_BATCHED_MATMUL_VALUE(mx::float32, float)
  DISPATCH_COO_BATCHED_MATMUL_VALUE(mx::float16, mx::float16_t)
  DISPATCH_COO_BATCHED_MATMUL_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_BATCHED_MATMUL_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_BATCHED_MATMUL_VALUE

  throw std::runtime_error("coo_batched_matmul unsupported value dtype.");
}

#ifdef _METAL_
void COOBatchedMatMul::eval_gpu(const std::vector<mx::array> &inputs,
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
    auto *zero_kernel =
        device.get_kernel("coo_batched_matmul_zero_float32", lib);
    encoder.set_compute_pipeline_state(zero_kernel);
    encoder.set_output_array(out, 0);
    auto out_size = static_cast<int>(out.size());
    encoder.set_bytes(out_size, 1);
    auto zero_threads = std::max<size_t>(out.size(), 1);
    auto zero_group =
        std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                             MTL::Size(zero_group, 1, 1));

    auto kernel_name = std::string("coo_batched_matmul_atomic_") +
                       index_kernel_suffix(row.dtype());
    auto *kernel = device.get_kernel(kernel_name, lib);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(data, 0);
    encoder.set_input_array(row, 1);
    encoder.set_input_array(col, 2);
    encoder.set_input_array(rhs, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(n_rows_, 5);
    encoder.set_bytes(n_cols_, 6);
    encoder.set_bytes(batch_size_, 7);
    encoder.set_bytes(rhs_cols_, 8);
    auto total = static_cast<int>(data.size() * batch_size_ * rhs_cols_);
    encoder.set_bytes(total, 9);
    auto threads = std::max<size_t>(total, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
    return;
  }

  auto kernel_name = sparse_kernel_name("coo_batched_matmul_serial",
                                        data.dtype(), row.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(row, 1);
  encoder.set_input_array(col, 2);
  encoder.set_input_array(rhs, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(n_cols_, 6);
  encoder.set_bytes(batch_size_, 7);
  encoder.set_bytes(rhs_cols_, 8);
  auto nnz = static_cast<int>(data.size());
  encoder.set_bytes(nnz, 9);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
#else
void COOBatchedMatMul::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_batched_matmul has no GPU implementation in this build.");
}
#endif

std::vector<mx::array>
COOBatchedMatMul::jvp(const std::vector<mx::array> &primals,
                      const std::vector<mx::array> &tangents,
                      const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    if (argnums[i] == 0) {
      terms.push_back(coo_batched_matmul(tangents[i], primals[1], primals[2],
                                         primals[3], n_rows_, n_cols_,
                                         stream()));
    } else if (argnums[i] == 3) {
      terms.push_back(coo_batched_matmul(primals[0], primals[1], primals[2],
                                         tangents[i], n_rows_, n_cols_,
                                         stream()));
    } else {
      throw std::runtime_error(
          "COOBatchedMatMul JVP is implemented only for data and dense RHS.");
    }
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array>
COOBatchedMatMul::vjp(const std::vector<mx::array> &primals,
                      const std::vector<mx::array> &cotangents,
                      const std::vector<int> &argnums,
                      const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());

  auto rhs_transposed = mx::transpose(primals[3], {1, 0, 2}, stream());
  auto rhs_flat = mx::reshape(
      rhs_transposed, mx::Shape{n_cols_, batch_size_ * rhs_cols_}, stream());
  auto cotangent_transposed = mx::transpose(cotangents[0], {1, 0, 2}, stream());
  auto cotangent_flat =
      mx::reshape(cotangent_transposed,
                  mx::Shape{n_rows_, batch_size_ * rhs_cols_}, stream());

  for (int argnum : argnums) {
    if (argnum == 0) {
      auto rhs = primals[3].dtype() == mx::complex64
                     ? mx::conjugate(rhs_flat, stream())
                     : rhs_flat;
      vjps.push_back(coo_matmul_data_vjp(primals[1], primals[2], rhs,
                                         cotangent_flat, n_rows_, n_cols_,
                                         stream()));
    } else if (argnum == 3) {
      auto data = primals[0].dtype() == mx::complex64
                      ? mx::conjugate(primals[0], stream())
                      : primals[0];
      auto rhs_vjp_flat =
          coo_matmul(data, primals[2], primals[1], cotangent_flat, n_cols_,
                     n_rows_, stream());
      auto rhs_vjp_transposed = mx::reshape(
          rhs_vjp_flat, mx::Shape{n_cols_, batch_size_, rhs_cols_}, stream());
      vjps.push_back(mx::transpose(rhs_vjp_transposed, {1, 0, 2}, stream()));
    } else {
      throw std::runtime_error(
          "COOBatchedMatMul VJP is implemented only for data and dense RHS.");
    }
  }
  return vjps;
}

mx::array coo_batched_matmul(const mx::array &data, const mx::array &row,
                             const mx::array &col, const mx::array &rhs,
                             int n_rows, int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "coo_batched_matmul shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "coo_batched_matmul data");
  require_rank(row, 1, "coo_batched_matmul row");
  require_rank(col, 1, "coo_batched_matmul col");
  require_rank(rhs, 3, "coo_batched_matmul rhs");
  require_same_value_dtype(data, rhs, "coo_batched_matmul data",
                           "coo_batched_matmul rhs");
  require_same_index_dtype(row, col, "coo_batched_matmul row",
                           "coo_batched_matmul col");
  if (row.size() != data.size() || col.size() != data.size()) {
    throw std::invalid_argument(
        "coo_batched_matmul data, row, and col must have equal length.");
  }
  if (rhs.shape(1) != n_cols) {
    throw std::invalid_argument("coo_batched_matmul rhs sparse dimension must "
                                "equal the sparse matrix column count.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto row_contig = mx::contiguous(row, false, stream);
  auto col_contig = mx::contiguous(col, false, stream);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  const int batch_size = rhs.shape(0);
  const int rhs_cols = rhs.shape(2);

  return mx::array(mx::Shape{batch_size, n_rows, rhs_cols}, data.dtype(),
                   std::make_shared<COOBatchedMatMul>(stream, n_rows, n_cols,
                                                      batch_size, rhs_cols),
                   {data_contig, row_contig, col_contig, rhs_contig});
}

mx::array coo_batched_matvec(const mx::array &data, const mx::array &row,
                             const mx::array &col, const mx::array &rhs,
                             int n_rows, int n_cols, mx::StreamOrDevice s) {
  require_rank(rhs, 2, "coo_batched_matvec rhs");
  if (rhs.shape(1) != n_cols) {
    throw std::invalid_argument("coo_batched_matvec rhs last dimension must "
                                "equal the sparse matrix column count.");
  }
  auto stream = mx::to_stream(s);
  auto rhs_matrix =
      mx::reshape(rhs, mx::Shape{rhs.shape(0), n_cols, 1}, stream);
  auto out =
      coo_batched_matmul(data, row, col, rhs_matrix, n_rows, n_cols, stream);
  return mx::reshape(out, mx::Shape{rhs.shape(0), n_rows}, stream);
}

} // namespace mlx_sparse
