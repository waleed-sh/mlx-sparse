// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "sparse/coo_matmul_data_vjp/coo_matmul_data_vjp.h"

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

class COOMatMulDataVJP : public mx::Primitive {
public:
  COOMatMulDataVJP(mx::Stream stream, int n_rows, int n_cols, int rhs_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols),
        rhs_cols_(rhs_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOMatMulDataVJP"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOMatMulDataVJP &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           rhs_cols_ == rhs.rhs_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
  int rhs_cols_;
};

template <typename T, typename I>
void coo_matmul_data_vjp_cpu_impl(const mx::array &row, const mx::array &col,
                                  const mx::array &rhs,
                                  const mx::array &cotangent, mx::array &out,
                                  int rhs_cols, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(row);
  encoder.set_input_array(col);
  encoder.set_input_array(rhs);
  encoder.set_input_array(cotangent);
  encoder.set_output_array(out);

  encoder.dispatch([row = mx::array::unsafe_weak_copy(row),
                    col = mx::array::unsafe_weak_copy(col),
                    rhs = mx::array::unsafe_weak_copy(rhs),
                    cotangent = mx::array::unsafe_weak_copy(cotangent),
                    out = mx::array::unsafe_weak_copy(out),
                    rhs_cols]() mutable {
    const auto *row_ptr = row.data<I>();
    const auto *col_ptr = col.data<I>();
    const auto *rhs_ptr = rhs.data<T>();
    const auto *cotangent_ptr = cotangent.data<T>();
    auto *out_ptr = out.data<T>();

    for (size_t p = 0; p < row.size(); ++p) {
      const auto rhs_offset = static_cast<size_t>(col_ptr[p]) * rhs_cols;
      const auto cot_offset = static_cast<size_t>(row_ptr[p]) * rhs_cols;
      auto acc = Accumulator<T>::zero();
      for (int k = 0; k < rhs_cols; ++k) {
        acc += multiply_accumulate<T>(cotangent_ptr[cot_offset + k],
                                      rhs_ptr[rhs_offset + k]);
      }
      out_ptr[p] = Accumulator<T>::cast(acc);
    }
  });
}

} // namespace

void COOMatMulDataVJP::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &row = inputs[0];
  auto &col = inputs[1];
  auto &rhs = inputs[2];
  auto &cotangent = inputs[3];
  auto &out = outputs[0];

  if (row.dtype() != mx::int32 && row.dtype() != mx::int64) {
    throw std::runtime_error(
        "coo_matmul_data_vjp requires int32 or int64 indices.");
  }

#define DISPATCH_COO_MATMUL_DATA_VJP_VALUE(DTYPE, TYPE)                        \
  if (rhs.dtype() == DTYPE) {                                                  \
    if (row.dtype() == mx::int32) {                                            \
      coo_matmul_data_vjp_cpu_impl<TYPE, int32_t>(row, col, rhs, cotangent,    \
                                                  out, rhs_cols_, stream());   \
    } else {                                                                   \
      coo_matmul_data_vjp_cpu_impl<TYPE, int64_t>(row, col, rhs, cotangent,    \
                                                  out, rhs_cols_, stream());   \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_MATMUL_DATA_VJP_VALUE(mx::float32, float)
  DISPATCH_COO_MATMUL_DATA_VJP_VALUE(mx::float16, mx::float16_t)
  DISPATCH_COO_MATMUL_DATA_VJP_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_MATMUL_DATA_VJP_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_MATMUL_DATA_VJP_VALUE

  throw std::runtime_error("coo_matmul_data_vjp unsupported value dtype.");
}

#ifdef _METAL_
void COOMatMulDataVJP::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &row = inputs[0];
  auto &col = inputs[1];
  auto &rhs = inputs[2];
  auto &cotangent = inputs[3];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("coo_matmul_data_vjp", rhs.dtype(), row.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(row, 0);
  encoder.set_input_array(col, 1);
  encoder.set_input_array(rhs, 2);
  encoder.set_input_array(cotangent, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(rhs_cols_, 5);
  auto nnz = static_cast<int>(row.size());
  encoder.set_bytes(nnz, 6);
  auto threads = std::max<size_t>(row.size(), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOMatMulDataVJP::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_matmul_data_vjp has no GPU implementation in this build.");
}
#endif

mx::array coo_matmul_data_vjp(const mx::array &row, const mx::array &col,
                              const mx::array &rhs, const mx::array &cotangent,
                              int n_rows, int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "coo_matmul_data_vjp shape dimensions must be non-negative.");
  }
  require_rank(row, 1, "coo_matmul_data_vjp row");
  require_rank(col, 1, "coo_matmul_data_vjp col");
  require_rank(rhs, 2, "coo_matmul_data_vjp rhs");
  require_rank(cotangent, 2, "coo_matmul_data_vjp cotangent");
  require_same_index_dtype(row, col, "coo_matmul_data_vjp row",
                           "coo_matmul_data_vjp col");
  require_same_value_dtype(rhs, cotangent, "coo_matmul_data_vjp rhs",
                           "coo_matmul_data_vjp cotangent");
  if (row.size() != col.size()) {
    throw std::invalid_argument(
        "coo_matmul_data_vjp row and col must have equal length.");
  }
  if (rhs.shape(0) != n_cols || cotangent.shape(0) != n_rows) {
    throw std::invalid_argument(
        "coo_matmul_data_vjp sparse dimensions do not match RHS/cotangent.");
  }
  if (rhs.shape(1) != cotangent.shape(1)) {
    throw std::invalid_argument(
        "coo_matmul_data_vjp rhs and cotangent must have the same columns.");
  }

  auto stream = mx::to_stream(s);
  auto row_contig = mx::contiguous(row, false, stream);
  auto col_contig = mx::contiguous(col, false, stream);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  auto cotangent_contig = mx::contiguous(cotangent, false, stream);
  const int rhs_cols = rhs.shape(1);

  return mx::array(
      mx::Shape{static_cast<int>(row.size())}, rhs.dtype(),
      std::make_shared<COOMatMulDataVJP>(stream, n_rows, n_cols, rhs_cols),
      {row_contig, col_contig, rhs_contig, cotangent_contig});
}

} // namespace mlx_sparse
