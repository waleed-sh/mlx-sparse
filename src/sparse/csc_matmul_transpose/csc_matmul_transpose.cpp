// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "sparse/csc_matmul_transpose/csc_matmul_transpose.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "common/common.h"
#include "common/cpu_parallel.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

class CSCMatMulTranspose : public mx::Primitive {
public:
  CSCMatMulTranspose(mx::Stream stream, int n_rows, int n_cols, int rhs_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols),
        rhs_cols_(rhs_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCMatMulTranspose"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCMatMulTranspose &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           rhs_cols_ == rhs.rhs_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
  int rhs_cols_;
};

template <typename T, typename I>
void csc_matmul_transpose_cpu_impl(const mx::array &data,
                                   const mx::array &indices,
                                   const mx::array &indptr,
                                   const mx::array &rhs, mx::array &out,
                                   int n_cols, int rhs_cols,
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
                    out = mx::array::unsafe_weak_copy(out), n_cols,
                    rhs_cols]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *rhs_ptr = rhs.data<T>();
    auto *out_ptr = out.data<T>();
    auto compute_cols = [&](CpuRange range) {
      std::vector<typename Accumulator<T>::Type> acc(
          static_cast<size_t>(rhs_cols));
      for (int col = range.begin; col < range.end; ++col) {
        std::fill(acc.begin(), acc.end(), Accumulator<T>::zero());
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          const auto rhs_offset =
              static_cast<size_t>(indices_ptr[p]) * rhs_cols;
          const T value = data_ptr[p];
          for (int k = 0; k < rhs_cols; ++k) {
            acc[static_cast<size_t>(k)] +=
                multiply_accumulate<T>(value, rhs_ptr[rhs_offset + k]);
          }
        }
        const auto out_offset = static_cast<size_t>(col) * rhs_cols;
        for (int k = 0; k < rhs_cols; ++k) {
          out_ptr[out_offset + k] =
              Accumulator<T>::cast(acc[static_cast<size_t>(k)]);
        }
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_cols <= 0) {
      compute_cols({0, n_cols});
      return;
    }
    const auto ranges =
        cpu_ranges_for_compressed_segments(indptr_ptr, n_cols, workers);
    if (ranges.size() <= 1) {
      compute_cols({0, n_cols});
      return;
    }
    parallel_for_cpu_ranges(ranges, compute_cols);
  });
}

} // namespace

void CSCMatMulTranspose::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &rhs = inputs[3];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error(
        "csc_matmul_transpose requires int32 or int64 indices.");
  }

#define DISPATCH_CSC_MATMUL_T_VALUE(DTYPE, TYPE)                               \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_matmul_transpose_cpu_impl<TYPE, int32_t>(                            \
          data, indices, indptr, rhs, out, n_cols_, rhs_cols_, stream());      \
    } else {                                                                   \
      csc_matmul_transpose_cpu_impl<TYPE, int64_t>(                            \
          data, indices, indptr, rhs, out, n_cols_, rhs_cols_, stream());      \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_MATMUL_T_VALUE(mx::float32, float)
  DISPATCH_CSC_MATMUL_T_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSC_MATMUL_T_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_MATMUL_T_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_MATMUL_T_VALUE

  throw std::runtime_error("csc_matmul_transpose unsupported value dtype.");
}

#ifdef _METAL_
void CSCMatMulTranspose::eval_gpu(const std::vector<mx::array> &inputs,
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
  auto kernel_name =
      sparse_kernel_name("csc_matmul_transpose", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(rhs, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_cols_, 5);
  encoder.set_bytes(rhs_cols_, 6);
  auto threads = std::max<size_t>(n_cols_ * rhs_cols_, 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCMatMulTranspose::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_matmul_transpose has no GPU implementation in this build.");
}
#endif

mx::array csc_matmul_transpose(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, const mx::array &rhs,
                               int n_rows, int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csc_matmul_transpose shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csc_matmul_transpose data");
  require_rank(indices, 1, "csc_matmul_transpose indices");
  require_rank(indptr, 1, "csc_matmul_transpose indptr");
  require_rank(rhs, 2, "csc_matmul_transpose rhs");
  require_same_value_dtype(data, rhs, "csc_matmul_transpose data",
                           "csc_matmul_transpose rhs");
  require_same_index_dtype(indices, indptr, "csc_matmul_transpose indices",
                           "csc_matmul_transpose indptr");
  require_size(indptr, n_cols + 1, "csc_matmul_transpose indptr");
  if (rhs.shape(0) != n_rows) {
    throw std::invalid_argument("csc_matmul_transpose rhs first dimension must "
                                "equal the sparse matrix row count.");
  }
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csc_matmul_transpose data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  const int rhs_cols = rhs.shape(1);

  return mx::array(
      mx::Shape{n_cols, rhs_cols}, data.dtype(),
      std::make_shared<CSCMatMulTranspose>(stream, n_rows, n_cols, rhs_cols),
      {data_contig, indices_contig, indptr_contig, rhs_contig});
}

} // namespace mlx_sparse
