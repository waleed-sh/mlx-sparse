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

#include "sparse/csr_batched_matmul/csr_batched_matmul.h"

#include "sparse/csr_matmul_data_vjp/csr_matmul_data_vjp.h"
#include "sparse/csr_matmul_transpose/csr_matmul_transpose.h"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "common/common.h"
#include "common/cpu_parallel.h"
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

constexpr size_t kVectorThreads = 128;
constexpr size_t kVectorMinAverageNnz = 32;

class CSRBatchedMatMul : public mx::Primitive {
public:
  CSRBatchedMatMul(mx::Stream stream, int n_rows, int n_cols, int batch_size,
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
                             const std::vector<mx::array> &outputs) override;

  std::pair<std::vector<mx::array>, std::vector<int>>
  vmap(const std::vector<mx::array> &inputs,
       const std::vector<int> &axes) override;

  const char *name() const override { return "CSRBatchedMatMul"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRBatchedMatMul &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           batch_size_ == rhs.batch_size_ && rhs_cols_ == rhs.rhs_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
  int batch_size_;
  int rhs_cols_;
};

template <typename T, typename I, int RHSCols>
void csr_batched_matmul_small_rhs_cpu_impl(const T *data_ptr,
                                           const I *indices_ptr,
                                           const I *indptr_ptr,
                                           const T *rhs_ptr, T *out_ptr,
                                           int n_rows, int n_cols,
                                           int item_begin, int item_end) {
  using AccT = typename Accumulator<T>::Type;

  for (int item = item_begin; item < item_end; ++item) {
    const int batch = item / n_rows;
    const int row = item - batch * n_rows;
    const auto rhs_batch =
        static_cast<size_t>(batch) * static_cast<size_t>(n_cols) * RHSCols;
    const auto out_batch =
        static_cast<size_t>(batch) * static_cast<size_t>(n_rows) * RHSCols;
    std::array<AccT, RHSCols> acc{};
    for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
      const auto rhs_offset =
          rhs_batch + static_cast<size_t>(indices_ptr[p]) * RHSCols;
      const auto value = data_ptr[p];
      for (int k = 0; k < RHSCols; ++k) {
        acc[static_cast<size_t>(k)] +=
            multiply_accumulate<T>(value, rhs_ptr[rhs_offset + k]);
      }
    }
    const auto out_offset = out_batch + static_cast<size_t>(row) * RHSCols;
    for (int k = 0; k < RHSCols; ++k) {
      out_ptr[out_offset + k] =
          Accumulator<T>::cast(acc[static_cast<size_t>(k)]);
    }
  }
}

template <typename T, typename I>
void csr_batched_matmul_generic_cpu_impl(const T *data_ptr,
                                         const I *indices_ptr,
                                         const I *indptr_ptr, const T *rhs_ptr,
                                         T *out_ptr, int n_rows, int n_cols,
                                         int item_begin, int item_end,
                                         int rhs_cols) {
  using AccT = typename Accumulator<T>::Type;
  std::vector<AccT> row_acc(static_cast<size_t>(rhs_cols));

  for (int item = item_begin; item < item_end; ++item) {
    const int batch = item / n_rows;
    const int row = item - batch * n_rows;
    const auto rhs_batch =
        static_cast<size_t>(batch) * static_cast<size_t>(n_cols) * rhs_cols;
    const auto out_batch =
        static_cast<size_t>(batch) * static_cast<size_t>(n_rows) * rhs_cols;
    std::fill(row_acc.begin(), row_acc.end(), Accumulator<T>::zero());
    for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
      const auto col = static_cast<size_t>(indices_ptr[p]);
      const auto rhs_offset = rhs_batch + col * rhs_cols;
      const auto value = data_ptr[p];
      for (int k = 0; k < rhs_cols; ++k) {
        row_acc[static_cast<size_t>(k)] +=
            multiply_accumulate<T>(value, rhs_ptr[rhs_offset + k]);
      }
    }
    const auto out_offset = out_batch + static_cast<size_t>(row) * rhs_cols;
    for (int k = 0; k < rhs_cols; ++k) {
      out_ptr[out_offset + k] =
          Accumulator<T>::cast(row_acc[static_cast<size_t>(k)]);
    }
  }
}

template <typename T, typename I>
void csr_batched_matmul_rows_cpu_impl(const T *data_ptr, const I *indices_ptr,
                                      const I *indptr_ptr, const T *rhs_ptr,
                                      T *out_ptr, int n_rows, int n_cols,
                                      int item_begin, int item_end,
                                      int rhs_cols) {
  switch (rhs_cols) {
  case 1:
    csr_batched_matmul_small_rhs_cpu_impl<T, I, 1>(
        data_ptr, indices_ptr, indptr_ptr, rhs_ptr, out_ptr, n_rows, n_cols,
        item_begin, item_end);
    return;
  case 2:
    csr_batched_matmul_small_rhs_cpu_impl<T, I, 2>(
        data_ptr, indices_ptr, indptr_ptr, rhs_ptr, out_ptr, n_rows, n_cols,
        item_begin, item_end);
    return;
  case 4:
    csr_batched_matmul_small_rhs_cpu_impl<T, I, 4>(
        data_ptr, indices_ptr, indptr_ptr, rhs_ptr, out_ptr, n_rows, n_cols,
        item_begin, item_end);
    return;
  case 8:
    csr_batched_matmul_small_rhs_cpu_impl<T, I, 8>(
        data_ptr, indices_ptr, indptr_ptr, rhs_ptr, out_ptr, n_rows, n_cols,
        item_begin, item_end);
    return;
  case 16:
    csr_batched_matmul_small_rhs_cpu_impl<T, I, 16>(
        data_ptr, indices_ptr, indptr_ptr, rhs_ptr, out_ptr, n_rows, n_cols,
        item_begin, item_end);
    return;
  default:
    csr_batched_matmul_generic_cpu_impl<T, I>(data_ptr, indices_ptr, indptr_ptr,
                                              rhs_ptr, out_ptr, n_rows, n_cols,
                                              item_begin, item_end, rhs_cols);
  }
}

template <typename T, typename I>
void csr_batched_matmul_cpu_impl(const mx::array &data,
                                 const mx::array &indices,
                                 const mx::array &indptr, const mx::array &rhs,
                                 mx::array &out, int n_rows, int n_cols,
                                 int batch_size, int rhs_cols,
                                 mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));
  const int requested_workers = configured_cpu_worker_count();

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
                    batch_size, rhs_cols, requested_workers]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *rhs_ptr = rhs.data<T>();
    auto *out_ptr = out.data<T>();

    auto compute_items = [&](CpuRange range) {
      csr_batched_matmul_rows_cpu_impl<T, I>(data_ptr, indices_ptr, indptr_ptr,
                                             rhs_ptr, out_ptr, n_rows, n_cols,
                                             range.begin, range.end, rhs_cols);
    };

    const int total_items = n_rows * batch_size;
    if (requested_workers <= 1 || total_items <= 0) {
      compute_items({0, total_items});
      return;
    }

    std::vector<int64_t> item_work(static_cast<size_t>(total_items));
    for (int item = 0; item < total_items; ++item) {
      const int row = item % n_rows;
      item_work[static_cast<size_t>(item)] =
          static_cast<int64_t>(indptr_ptr[row + 1] - indptr_ptr[row]) *
          std::max<int64_t>(rhs_cols, 1);
    }
    const auto ranges =
        cpu_ranges_for_output_work(item_work, requested_workers);
    if (ranges.size() <= 1) {
      compute_items({0, total_items});
      return;
    }
    parallel_for_cpu_ranges(ranges, compute_items);
  });
}

} // namespace

void CSRBatchedMatMul::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &rhs = inputs[3];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error(
        "csr_batched_matmul requires int32 or int64 indices.");
  }

#define DISPATCH_CSR_BATCHED_MATMUL_VALUE(DTYPE, TYPE)                         \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csr_batched_matmul_cpu_impl<TYPE, int32_t>(                              \
          data, indices, indptr, rhs, out, n_rows_, n_cols_, batch_size_,      \
          rhs_cols_, stream());                                                \
    } else {                                                                   \
      csr_batched_matmul_cpu_impl<TYPE, int64_t>(                              \
          data, indices, indptr, rhs, out, n_rows_, n_cols_, batch_size_,      \
          rhs_cols_, stream());                                                \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_BATCHED_MATMUL_VALUE(mx::float32, float)
  DISPATCH_CSR_BATCHED_MATMUL_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_BATCHED_MATMUL_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_BATCHED_MATMUL_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_BATCHED_MATMUL_VALUE

  throw std::runtime_error("csr_batched_matmul unsupported value dtype.");
}

#ifdef _METAL_
void CSRBatchedMatMul::eval_gpu(const std::vector<mx::array> &inputs,
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
  const bool use_vector_kernel =
      n_rows_ > 0 &&
      data.size() >= static_cast<size_t>(n_rows_) * kVectorMinAverageNnz;
  auto kernel_name = sparse_kernel_name(
      use_vector_kernel ? "csr_batched_matmul_vector" : "csr_batched_matmul",
      data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(rhs, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(n_cols_, 6);
  encoder.set_bytes(batch_size_, 7);
  encoder.set_bytes(rhs_cols_, 8);

  const auto outputs_count =
      static_cast<size_t>(batch_size_) * n_rows_ * rhs_cols_;
  if (use_vector_kernel) {
    encoder.dispatch_threads(MTL::Size(outputs_count * kVectorThreads, 1, 1),
                             MTL::Size(kVectorThreads, 1, 1));
  } else {
    auto threads = std::max<size_t>(outputs_count, 1);
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
  }
}
#else
void CSRBatchedMatMul::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_batched_matmul has no GPU implementation in this build.");
}
#endif

std::vector<mx::array>
CSRBatchedMatMul::jvp(const std::vector<mx::array> &primals,
                      const std::vector<mx::array> &tangents,
                      const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    if (argnums[i] == 0) {
      terms.push_back(csr_batched_matmul(tangents[i], primals[1], primals[2],
                                         primals[3], n_rows_, n_cols_,
                                         stream()));
    } else if (argnums[i] == 3) {
      terms.push_back(csr_batched_matmul(primals[0], primals[1], primals[2],
                                         tangents[i], n_rows_, n_cols_,
                                         stream()));
    } else {
      throw std::runtime_error(
          "CSRBatchedMatMul JVP is implemented only for data and dense RHS.");
    }
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array>
CSRBatchedMatMul::vjp(const std::vector<mx::array> &primals,
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
      vjps.push_back(csr_matmul_data_vjp(primals[1], primals[2], rhs,
                                         cotangent_flat, n_rows_, n_cols_,
                                         stream()));
    } else if (argnum == 3) {
      auto data = primals[0].dtype() == mx::complex64
                      ? mx::conjugate(primals[0], stream())
                      : primals[0];
      auto rhs_vjp_flat =
          csr_matmul_transpose(data, primals[1], primals[2], cotangent_flat,
                               n_rows_, n_cols_, stream());
      auto rhs_vjp_transposed = mx::reshape(
          rhs_vjp_flat, mx::Shape{n_cols_, batch_size_, rhs_cols_}, stream());
      vjps.push_back(mx::transpose(rhs_vjp_transposed, {1, 0, 2}, stream()));
    } else {
      throw std::runtime_error(
          "CSRBatchedMatMul VJP is implemented only for data and dense RHS.");
    }
  }
  return vjps;
}

std::pair<std::vector<mx::array>, std::vector<int>>
CSRBatchedMatMul::vmap(const std::vector<mx::array> &inputs,
                       const std::vector<int> &axes) {
  require_vmap_arity(inputs, axes, 4, "CSRBatchedMatMul");
  require_fixed_sparse_vmap_axes(axes, 3, "CSRBatchedMatMul");

  auto rhs = dense_rhs_with_vmap_axis_front(inputs[3], axes[3], stream(),
                                            "CSRBatchedMatMul");
  require_vmap_rhs_rank(rhs, 4, "CSRBatchedMatMul");
  require_vmap_rhs_dim(rhs, 1, batch_size_, "dense RHS batch dimension",
                       "CSRBatchedMatMul");
  require_vmap_rhs_sparse_dim(rhs, 2, n_cols_, "CSRBatchedMatMul");
  require_vmap_rhs_dim(rhs, 3, rhs_cols_, "dense RHS column dimension",
                       "CSRBatchedMatMul");

  const int outer_batch = rhs.shape(0);
  const int flat_batch =
      checked_vmap_batch_product(outer_batch, batch_size_, "CSRBatchedMatMul");
  auto flat_rhs =
      mx::reshape(rhs, mx::Shape{flat_batch, n_cols_, rhs_cols_}, stream());
  auto flat_out = csr_batched_matmul(inputs[0], inputs[1], inputs[2], flat_rhs,
                                     n_rows_, n_cols_, stream());
  auto out = mx::reshape(
      flat_out, mx::Shape{outer_batch, batch_size_, n_rows_, rhs_cols_},
      stream());
  return {{out}, {0}};
}

mx::array csr_batched_matmul(const mx::array &data, const mx::array &indices,
                             const mx::array &indptr, const mx::array &rhs,
                             int n_rows, int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr_batched_matmul shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csr_batched_matmul data");
  require_rank(indices, 1, "csr_batched_matmul indices");
  require_rank(indptr, 1, "csr_batched_matmul indptr");
  require_rank(rhs, 3, "csr_batched_matmul rhs");
  require_same_value_dtype(data, rhs, "csr_batched_matmul data",
                           "csr_batched_matmul rhs");
  require_same_index_dtype(indices, indptr, "csr_batched_matmul indices",
                           "csr_batched_matmul indptr");
  require_size(indptr, n_rows + 1, "csr_batched_matmul indptr");
  if (rhs.shape(1) != n_cols) {
    throw std::invalid_argument("csr_batched_matmul rhs sparse dimension must "
                                "equal the sparse matrix column count.");
  }
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_batched_matmul data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  const int batch_size = rhs.shape(0);
  const int rhs_cols = rhs.shape(2);

  return mx::array(mx::Shape{batch_size, n_rows, rhs_cols}, data.dtype(),
                   std::make_shared<CSRBatchedMatMul>(stream, n_rows, n_cols,
                                                      batch_size, rhs_cols),
                   {data_contig, indices_contig, indptr_contig, rhs_contig});
}

} // namespace mlx_sparse
