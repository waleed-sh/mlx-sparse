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

#include "sparse/csc_matvec_transpose/csc_matvec_transpose.h"

#include "sparse/csc_matmul_data_vjp/csc_matmul_data_vjp.h"
#include "sparse/csc_matvec/csc_matvec.h"
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

constexpr size_t kVectorThreads = 128;
constexpr size_t kVectorMinAverageNnz = 32;

class CSCMatVecTranspose : public mx::Primitive {
public:
  CSCMatVecTranspose(mx::Stream stream, int n_rows, int n_cols)
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

  const char *name() const override { return "CSCMatVecTranspose"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCMatVecTranspose &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csc_matvec_transpose_cpu_impl(const mx::array &data,
                                   const mx::array &indices,
                                   const mx::array &indptr, const mx::array &x,
                                   mx::array &out, int n_cols,
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
                    out = mx::array::unsafe_weak_copy(out), n_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *x_ptr = x.data<T>();
    auto *out_ptr = out.data<T>();

    auto compute_cols = [&](CpuRange range) {
      for (int col = range.begin; col < range.end; ++col) {
        auto acc = Accumulator<T>::zero();
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          acc += multiply_accumulate<T>(data_ptr[p], x_ptr[indices_ptr[p]]);
        }
        out_ptr[col] = Accumulator<T>::cast(static_cast<AccT>(acc));
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

void CSCMatVecTranspose::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &x = inputs[3];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error(
        "csc_matvec_transpose requires int32 or int64 indices.");
  }

#define DISPATCH_CSC_MATVEC_T_VALUE(DTYPE, TYPE)                               \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_matvec_transpose_cpu_impl<TYPE, int32_t>(data, indices, indptr, x,   \
                                                   out, n_cols_, stream());    \
    } else {                                                                   \
      csc_matvec_transpose_cpu_impl<TYPE, int64_t>(data, indices, indptr, x,   \
                                                   out, n_cols_, stream());    \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_MATVEC_T_VALUE(mx::float32, float)
  DISPATCH_CSC_MATVEC_T_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSC_MATVEC_T_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_MATVEC_T_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_MATVEC_T_VALUE

  throw std::runtime_error("csc_matvec_transpose unsupported value dtype.");
}

std::vector<mx::array>
CSCMatVecTranspose::jvp(const std::vector<mx::array> &primals,
                        const std::vector<mx::array> &tangents,
                        const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    if (argnums[i] == 0) {
      terms.push_back(csc_matvec_transpose(tangents[i], primals[1], primals[2],
                                           primals[3], n_rows_, n_cols_,
                                           stream()));
    } else if (argnums[i] == 3) {
      terms.push_back(csc_matvec_transpose(primals[0], primals[1], primals[2],
                                           tangents[i], n_rows_, n_cols_,
                                           stream()));
    } else {
      throw std::runtime_error(
          "CSCMatVecTranspose JVP is implemented only for data and dense RHS.");
    }
  }
  if (terms.empty()) {
    throw std::runtime_error(
        "CSCMatVecTranspose JVP requires at least one tangent.");
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array>
CSCMatVecTranspose::vjp(const std::vector<mx::array> &primals,
                        const std::vector<mx::array> &cotangents,
                        const std::vector<int> &argnums,
                        const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  for (int argnum : argnums) {
    if (argnum == 0) {
      auto x = primals[3].dtype() == mx::complex64
                   ? mx::conjugate(primals[3], stream())
                   : primals[3];
      auto rhs_matrix =
          mx::reshape(cotangents[0], mx::Shape{n_cols_, 1}, stream());
      auto cotangent_matrix = mx::reshape(x, mx::Shape{n_rows_, 1}, stream());
      vjps.push_back(csc_matmul_data_vjp(primals[1], primals[2], rhs_matrix,
                                         cotangent_matrix, n_rows_, n_cols_,
                                         stream()));
    } else if (argnum == 3) {
      auto data = primals[0].dtype() == mx::complex64
                      ? mx::conjugate(primals[0], stream())
                      : primals[0];
      vjps.push_back(csc_matvec(data, primals[1], primals[2], cotangents[0],
                                n_rows_, n_cols_, stream()));
    } else {
      throw std::runtime_error(
          "CSCMatVecTranspose VJP is implemented only for data and dense RHS.");
    }
  }
  return vjps;
}

#ifdef _METAL_
void CSCMatVecTranspose::eval_gpu(const std::vector<mx::array> &inputs,
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
  const bool use_vector_kernel =
      n_cols_ > 0 &&
      data.size() >= static_cast<size_t>(n_cols_) * kVectorMinAverageNnz;
  auto kernel_name =
      sparse_kernel_name(use_vector_kernel ? "csc_matvec_transpose_vector"
                                           : "csc_matvec_transpose",
                         data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(x, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_cols_, 5);

  if (use_vector_kernel) {
    const auto threadgroups = static_cast<size_t>(n_cols_);
    encoder.dispatch_threads(MTL::Size(threadgroups * kVectorThreads, 1, 1),
                             MTL::Size(kVectorThreads, 1, 1));
  } else {
    auto threads = static_cast<size_t>(std::max(n_cols_, 1));
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
  }
}
#else
void CSCMatVecTranspose::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_matvec_transpose has no GPU implementation in this build.");
}
#endif

mx::array csc_matvec_transpose(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, const mx::array &x,
                               int n_rows, int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csc_matvec_transpose shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csc_matvec_transpose data");
  require_rank(indices, 1, "csc_matvec_transpose indices");
  require_rank(indptr, 1, "csc_matvec_transpose indptr");
  require_rank(x, 1, "csc_matvec_transpose x");
  require_same_value_dtype(data, x, "csc_matvec_transpose data",
                           "csc_matvec_transpose x");
  require_same_index_dtype(indices, indptr, "csc_matvec_transpose indices",
                           "csc_matvec_transpose indptr");
  require_size(indptr, n_cols + 1, "csc_matvec_transpose indptr");
  require_size(x, n_rows, "csc_matvec_transpose x");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csc_matvec_transpose data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto x_contig = mx::contiguous(x, false, stream);

  return mx::array(mx::Shape{n_cols}, data.dtype(),
                   std::make_shared<CSCMatVecTranspose>(stream, n_rows, n_cols),
                   {data_contig, indices_contig, indptr_contig, x_contig});
}

} // namespace mlx_sparse
