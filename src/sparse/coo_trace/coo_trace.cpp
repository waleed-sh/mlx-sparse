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

#include "sparse/coo_trace/coo_trace.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/autodiff.h"
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

constexpr size_t kTraceThreads = 128;
constexpr int kTraceNnzPerBlock = 2048;

mx::Dtype trace_accumulator_dtype(mx::Dtype dtype) {
  if (dtype == mx::float16 || dtype == mx::bfloat16) {
    return mx::float32;
  }
  return dtype;
}

class COOTrace : public mx::Primitive {
public:
  COOTrace(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

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

  const char *name() const override { return "COOTrace"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOTrace &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void coo_trace_cpu_impl(const mx::array &data, const mx::array &row,
                        const mx::array &col, mx::array &out, int diag_size,
                        mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(row);
  encoder.set_input_array(col);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    row = mx::array::unsafe_weak_copy(row),
                    col = mx::array::unsafe_weak_copy(col),
                    out = mx::array::unsafe_weak_copy(out),
                    diag_size]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *row_ptr = row.data<I>();
    const auto *col_ptr = col.data<I>();
    using AccT = typename Accumulator<T>::Type;
    const int nnz = static_cast<int>(data.size());

    auto compute_nnz = [&](CpuRange range) -> AccT {
      auto local = Accumulator<T>::zero();
      for (int p = range.begin; p < range.end; ++p) {
        const auto r = row_ptr[p];
        if (r == col_ptr[p] && r >= 0 && r < static_cast<I>(diag_size)) {
          local += static_cast<AccT>(data_ptr[p]);
        }
      }
      return local;
    };

    AccT acc = Accumulator<T>::zero();
    const int workers = configured_cpu_worker_count();
    if (!should_parallelize_cpu_tree_reduction(workers, nnz)) {
      acc = compute_nnz({0, nnz});
    } else {
      const auto ranges = equal_cpu_ranges(nnz, workers);
      acc = parallel_reduce_cpu_ranges<AccT>(ranges, compute_nnz);
    }
    *out.data<T>() = Accumulator<T>::cast(acc);
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

void COOTrace::eval_cpu(const std::vector<mx::array> &inputs,
                        std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  const int diag_size = std::min(n_rows_, n_cols_);

#define DISPATCH_COO_TRACE(DTYPE, TYPE)                                        \
  if (data.dtype() == DTYPE) {                                                 \
    if (row.dtype() == mx::int32) {                                            \
      coo_trace_cpu_impl<TYPE, int32_t>(data, row, col, outputs[0], diag_size, \
                                        stream());                             \
    } else {                                                                   \
      coo_trace_cpu_impl<TYPE, int64_t>(data, row, col, outputs[0], diag_size, \
                                        stream());                             \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_TRACE(mx::float32, float)
  DISPATCH_COO_TRACE(mx::float16, mx::float16_t)
  DISPATCH_COO_TRACE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_TRACE(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_TRACE

  throw std::runtime_error("coo_trace unsupported value dtype.");
}

std::vector<mx::array> COOTrace::jvp(const std::vector<mx::array> &primals,
                                     const std::vector<mx::array> &tangents,
                                     const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    require_sparse_value_autodiff_arg(argnums[i], "COOTrace", "JVP");
    terms.push_back(coo_trace(tangents[i], primals[1], primals[2], n_rows_,
                              n_cols_, stream()));
  }
  if (terms.empty()) {
    throw std::runtime_error("COOTrace JVP requires at least one tangent.");
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array> COOTrace::vjp(const std::vector<mx::array> &primals,
                                     const std::vector<mx::array> &cotangents,
                                     const std::vector<int> &argnums,
                                     const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  const int diag_size = std::min(n_rows_, n_cols_);
  for (int argnum : argnums) {
    require_sparse_value_autodiff_arg(argnum, "COOTrace", "VJP");
    vjps.push_back(sparse_trace_cotangent_gather(cotangents[0], primals[1],
                                                 primals[2], primals[0],
                                                 diag_size, stream()));
  }
  return vjps;
}

#ifdef _METAL_
void COOTrace::eval_gpu(const std::vector<mx::array> &inputs,
                        std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  auto &out = outputs[0];
  const int diag_size = std::min(n_rows_, n_cols_);

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  auto nnz = static_cast<int>(data.size());
  const int num_blocks = (nnz + kTraceNnzPerBlock - 1) / kTraceNnzPerBlock;
  if (num_blocks > 1) {
    const auto partial_dtype = trace_accumulator_dtype(data.dtype());
    mx::array partials(mx::allocator::malloc(static_cast<size_t>(num_blocks) *
                                             mx::size_of(partial_dtype)),
                       mx::Shape{num_blocks}, partial_dtype);

    auto blocks_kernel_name =
        sparse_kernel_name("coo_trace_blocks", data.dtype(), row.dtype());
    auto *blocks_kernel = device.get_kernel(blocks_kernel_name, lib);
    encoder.set_compute_pipeline_state(blocks_kernel);
    encoder.set_input_array(data, 0);
    encoder.set_input_array(row, 1);
    encoder.set_input_array(col, 2);
    encoder.set_output_array(partials, 3);
    encoder.set_bytes(nnz, 4);
    encoder.set_bytes(diag_size, 5);
    encoder.set_bytes(kTraceNnzPerBlock, 6);
    encoder.dispatch_threads(
        MTL::Size(static_cast<size_t>(num_blocks) * kTraceThreads, 1, 1),
        MTL::Size(kTraceThreads, 1, 1));

    auto finalize_kernel_name =
        std::string("coo_trace_finalize_") + value_kernel_suffix(data.dtype());
    auto *finalize_kernel = device.get_kernel(finalize_kernel_name, lib);
    encoder.set_compute_pipeline_state(finalize_kernel);
    encoder.set_input_array(partials, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(num_blocks, 2);
    encoder.dispatch_threads(MTL::Size(kTraceThreads, 1, 1),
                             MTL::Size(kTraceThreads, 1, 1));
    encoder.add_temporary(std::move(partials));
    return;
  }

  auto kernel_name = sparse_kernel_name("coo_trace", data.dtype(), row.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(row, 1);
  encoder.set_input_array(col, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(nnz, 4);
  encoder.set_bytes(diag_size, 5);
  encoder.dispatch_threads(MTL::Size(128, 1, 1), MTL::Size(128, 1, 1));
}
#else
void COOTrace::eval_gpu(const std::vector<mx::array> &,
                        std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_trace has no GPU implementation in this build.");
}
#endif

mx::array coo_trace(const mx::array &data, const mx::array &row,
                    const mx::array &col, int n_rows, int n_cols,
                    mx::StreamOrDevice s) {
  validate_coo_reduction_inputs(data, row, col, n_rows, n_cols, "coo_trace");

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto row_contig = mx::contiguous(row, false, stream);
  auto col_contig = mx::contiguous(col, false, stream);

  return mx::array(mx::Shape{}, data.dtype(),
                   std::make_shared<COOTrace>(stream, n_rows, n_cols),
                   {data_contig, row_contig, col_contig});
}

} // namespace mlx_sparse
