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

#include "sparse/csc_diagonal/csc_diagonal.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "common/autodiff.h"
#include "common/cpu_parallel.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "sparse/csc_tocoo/csc_tocoo.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

constexpr size_t kVectorThreads = 128;
constexpr size_t kVectorMinAverageNnz = 32;

class CSCDiagonal : public mx::Primitive {
public:
  CSCDiagonal(mx::Stream stream, int n_rows, int n_cols)
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

  const char *name() const override { return "CSCDiagonal"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCDiagonal &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csc_diagonal_cpu_impl(const mx::array &data, const mx::array &indices,
                           const mx::array &indptr, mx::array &out,
                           int diag_size, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out = mx::array::unsafe_weak_copy(out),
                    diag_size]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *out_ptr = out.data<T>();

    auto compute_cols = [&](CpuRange range) {
      for (int col = range.begin; col < range.end; ++col) {
        auto acc = Accumulator<T>::zero();
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          if (indices_ptr[p] == static_cast<I>(col)) {
            acc += static_cast<typename Accumulator<T>::Type>(data_ptr[p]);
          }
        }
        out_ptr[col] = Accumulator<T>::cast(acc);
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || diag_size <= 0) {
      compute_cols({0, diag_size});
      return;
    }
    const auto ranges =
        cpu_ranges_for_compressed_segments(indptr_ptr, diag_size, workers);
    if (ranges.size() <= 1) {
      compute_cols({0, diag_size});
      return;
    }
    parallel_for_cpu_ranges(ranges, compute_cols);
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

void CSCDiagonal::eval_cpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  const int diag_size = std::min(n_rows_, n_cols_);

#define DISPATCH_CSC_DIAGONAL(DTYPE, TYPE)                                     \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_diagonal_cpu_impl<TYPE, int32_t>(data, indices, indptr, outputs[0],  \
                                           diag_size, stream());               \
    } else {                                                                   \
      csc_diagonal_cpu_impl<TYPE, int64_t>(data, indices, indptr, outputs[0],  \
                                           diag_size, stream());               \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_DIAGONAL(mx::float32, float)
  DISPATCH_CSC_DIAGONAL(mx::float16, mx::float16_t)
  DISPATCH_CSC_DIAGONAL(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_DIAGONAL(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_DIAGONAL

  throw std::runtime_error("csc_diagonal unsupported value dtype.");
}

std::vector<mx::array> CSCDiagonal::jvp(const std::vector<mx::array> &primals,
                                        const std::vector<mx::array> &tangents,
                                        const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    require_sparse_value_autodiff_arg(argnums[i], "CSCDiagonal", "JVP");
    terms.push_back(csc_diagonal(tangents[i], primals[1], primals[2], n_rows_,
                                 n_cols_, stream()));
  }
  if (terms.empty()) {
    throw std::runtime_error("CSCDiagonal JVP requires at least one tangent.");
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array>
CSCDiagonal::vjp(const std::vector<mx::array> &primals,
                 const std::vector<mx::array> &cotangents,
                 const std::vector<int> &argnums,
                 const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  const int diag_size = std::min(n_rows_, n_cols_);
  for (int argnum : argnums) {
    require_sparse_value_autodiff_arg(argnum, "CSCDiagonal", "VJP");
    auto [_, row, col] = csc_tocoo(primals[0], primals[1], primals[2], n_rows_,
                                   n_cols_, stream());
    vjps.push_back(sparse_diagonal_cotangent_gather(
        cotangents[0], row, col, primals[0], diag_size, stream()));
  }
  return vjps;
}

#ifdef _METAL_
void CSCDiagonal::eval_gpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out = outputs[0];
  const int diag_size = std::min(n_rows_, n_cols_);

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  const bool use_vector_kernel =
      diag_size > 0 &&
      data.size() >= static_cast<size_t>(diag_size) * kVectorMinAverageNnz;
  auto kernel_name = sparse_kernel_name(
      use_vector_kernel ? "csc_diagonal_vector" : "csc_diagonal", data.dtype(),
      indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(diag_size, 4);
  if (use_vector_kernel) {
    const auto threadgroups = static_cast<size_t>(diag_size);
    encoder.dispatch_threads(MTL::Size(threadgroups * kVectorThreads, 1, 1),
                             MTL::Size(kVectorThreads, 1, 1));
  } else {
    auto threads = static_cast<size_t>(std::max(diag_size, 1));
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
  }
}
#else
void CSCDiagonal::eval_gpu(const std::vector<mx::array> &,
                           std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_diagonal has no GPU implementation in this build.");
}
#endif

mx::array csc_diagonal(const mx::array &data, const mx::array &indices,
                       const mx::array &indptr, int n_rows, int n_cols,
                       mx::StreamOrDevice s) {
  validate_csc_reduction_inputs(data, indices, indptr, n_rows, n_cols,
                                "csc_diagonal");

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  const int diag_size = std::min(n_rows, n_cols);

  return mx::array(mx::Shape{diag_size}, data.dtype(),
                   std::make_shared<CSCDiagonal>(stream, n_rows, n_cols),
                   {data_contig, indices_contig, indptr_contig});
}

} // namespace mlx_sparse
