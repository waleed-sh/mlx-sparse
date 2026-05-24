// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "sparse/csc_matmul/csc_matmul.h"

#include "sparse/csc_matmul_data_vjp/csc_matmul_data_vjp.h"
#include "sparse/csc_matmul_transpose/csc_matmul_transpose.h"
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

class CSCMatMul : public mx::Primitive {
public:
  CSCMatMul(mx::Stream stream, int n_rows, int n_cols, int rhs_cols)
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

  const char *name() const override { return "CSCMatMul"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCMatMul &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           rhs_cols_ == rhs.rhs_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
  int rhs_cols_;
};

template <typename T, typename I>
void csc_matmul_cpu_impl(const mx::array &data, const mx::array &indices,
                         const mx::array &indptr, const mx::array &rhs,
                         mx::array &out, int n_rows, int n_cols, int rhs_cols,
                         mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(rhs);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    rhs = mx::array::unsafe_weak_copy(rhs),
                    out = mx::array::unsafe_weak_copy(out), n_rows, n_cols,
                    rhs_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *rhs_ptr = rhs.data<T>();
    auto *out_ptr = out.data<T>();
    const size_t out_size = static_cast<size_t>(n_rows) * rhs_cols;

    if constexpr (std::is_same_v<AccT, T>) {
      std::fill(out_ptr, out_ptr + out_size, T{});
      for (int col = 0; col < n_cols; ++col) {
        const auto rhs_offset = static_cast<size_t>(col) * rhs_cols;
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          const auto out_offset =
              static_cast<size_t>(indices_ptr[p]) * rhs_cols;
          const T value = data_ptr[p];
          for (int k = 0; k < rhs_cols; ++k) {
            out_ptr[out_offset + k] += value * rhs_ptr[rhs_offset + k];
          }
        }
      }
    } else {
      std::vector<AccT> accum(out_size, Accumulator<T>::zero());
      for (int col = 0; col < n_cols; ++col) {
        const auto rhs_offset = static_cast<size_t>(col) * rhs_cols;
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          const auto out_offset =
              static_cast<size_t>(indices_ptr[p]) * rhs_cols;
          const T value = data_ptr[p];
          for (int k = 0; k < rhs_cols; ++k) {
            accum[out_offset + k] +=
                multiply_accumulate<T>(value, rhs_ptr[rhs_offset + k]);
          }
        }
      }
      for (size_t i = 0; i < out_size; ++i) {
        out_ptr[i] = Accumulator<T>::cast(accum[i]);
      }
    }
  });
}

} // namespace

void CSCMatMul::eval_cpu(const std::vector<mx::array> &inputs,
                         std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &rhs = inputs[3];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error("csc_matmul requires int32 or int64 indices.");
  }

#define DISPATCH_CSC_MATMUL_VALUE(DTYPE, TYPE)                                 \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_matmul_cpu_impl<TYPE, int32_t>(data, indices, indptr, rhs, out,      \
                                         n_rows_, n_cols_, rhs_cols_,          \
                                         stream());                            \
    } else {                                                                   \
      csc_matmul_cpu_impl<TYPE, int64_t>(data, indices, indptr, rhs, out,      \
                                         n_rows_, n_cols_, rhs_cols_,          \
                                         stream());                            \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_MATMUL_VALUE(mx::float32, float)
  DISPATCH_CSC_MATMUL_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSC_MATMUL_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_MATMUL_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_MATMUL_VALUE

  throw std::runtime_error("csc_matmul unsupported value dtype.");
}

#ifdef _METAL_
void CSCMatMul::eval_gpu(const std::vector<mx::array> &inputs,
                         std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &rhs = inputs[3];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  if (data.dtype() == mx::float32) {
    auto *zero_kernel = device.get_kernel("csc_matmul_zero_float32", lib);
    encoder.set_compute_pipeline_state(zero_kernel);
    encoder.set_output_array(out, 0);
    auto out_size = static_cast<int>(out.size());
    encoder.set_bytes(out_size, 1);
    auto zero_threads = std::max<size_t>(out.size(), 1);
    auto zero_group =
        std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                             MTL::Size(zero_group, 1, 1));

    auto kernel_name = std::string("csc_matmul_atomic_") +
                       index_kernel_suffix(indices.dtype());
    auto *kernel = device.get_kernel(kernel_name, lib);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(data, 0);
    encoder.set_input_array(indices, 1);
    encoder.set_input_array(indptr, 2);
    encoder.set_input_array(rhs, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(n_rows_, 5);
    encoder.set_bytes(n_cols_, 6);
    encoder.set_bytes(rhs_cols_, 7);
    auto threads = std::max<size_t>(n_cols_ * rhs_cols_, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
    return;
  }

  auto kernel_name =
      sparse_kernel_name("csc_matmul_serial", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(rhs, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(n_cols_, 6);
  encoder.set_bytes(rhs_cols_, 7);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
#else
void CSCMatMul::eval_gpu(const std::vector<mx::array> &,
                         std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_matmul has no GPU implementation in this build.");
}
#endif

std::vector<mx::array> CSCMatMul::jvp(const std::vector<mx::array> &primals,
                                      const std::vector<mx::array> &tangents,
                                      const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    if (argnums[i] == 0) {
      terms.push_back(csc_matmul(tangents[i], primals[1], primals[2],
                                 primals[3], n_rows_, n_cols_, stream()));
    } else if (argnums[i] == 3) {
      terms.push_back(csc_matmul(primals[0], primals[1], primals[2],
                                 tangents[i], n_rows_, n_cols_, stream()));
    } else {
      throw std::runtime_error(
          "CSCMatMul JVP is implemented only for data and dense RHS.");
    }
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array> CSCMatMul::vjp(const std::vector<mx::array> &primals,
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
      vjps.push_back(csc_matmul_data_vjp(primals[1], primals[2], rhs,
                                         cotangents[0], n_rows_, n_cols_,
                                         stream()));
    } else if (argnum == 3) {
      auto data = primals[0].dtype() == mx::complex64
                      ? mx::conjugate(primals[0], stream())
                      : primals[0];
      vjps.push_back(csc_matmul_transpose(data, primals[1], primals[2],
                                          cotangents[0], n_rows_, n_cols_,
                                          stream()));
    } else {
      throw std::runtime_error(
          "CSCMatMul VJP is implemented only for data and dense RHS.");
    }
  }
  return vjps;
}

mx::array csc_matmul(const mx::array &data, const mx::array &indices,
                     const mx::array &indptr, const mx::array &rhs, int n_rows,
                     int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csc_matmul shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csc_matmul data");
  require_rank(indices, 1, "csc_matmul indices");
  require_rank(indptr, 1, "csc_matmul indptr");
  require_rank(rhs, 2, "csc_matmul rhs");
  require_same_value_dtype(data, rhs, "csc_matmul data", "csc_matmul rhs");
  require_same_index_dtype(indices, indptr, "csc_matmul indices",
                           "csc_matmul indptr");
  require_size(indptr, n_cols + 1, "csc_matmul indptr");
  if (rhs.shape(0) != n_cols) {
    throw std::invalid_argument("csc_matmul rhs first dimension must equal "
                                "the sparse matrix column count.");
  }
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csc_matmul data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  const int rhs_cols = rhs.shape(1);

  return mx::array(
      mx::Shape{n_rows, rhs_cols}, data.dtype(),
      std::make_shared<CSCMatMul>(stream, n_rows, n_cols, rhs_cols),
      {data_contig, indices_contig, indptr_contig, rhs_contig});
}

} // namespace mlx_sparse
