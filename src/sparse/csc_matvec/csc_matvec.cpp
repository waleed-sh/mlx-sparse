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

#include "sparse/csc_matvec/csc_matvec.h"

#include "sparse/csc_matmul_data_vjp/csc_matmul_data_vjp.h"
#include "sparse/csc_matvec_transpose/csc_matvec_transpose.h"
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

class CSCMatVec : public mx::Primitive {
public:
  CSCMatVec(mx::Stream stream, int n_rows, int n_cols)
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

  const char *name() const override { return "CSCMatVec"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCMatVec &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csc_matvec_cpu_impl(const mx::array &data, const mx::array &indices,
                         const mx::array &indptr, const mx::array &x,
                         mx::array &out, int n_rows, int n_cols,
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
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    n_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *x_ptr = x.data<T>();
    auto *out_ptr = out.data<T>();

    if constexpr (std::is_same_v<AccT, T>) {
      std::fill(out_ptr, out_ptr + n_rows, T{});
      for (int col = 0; col < n_cols; ++col) {
        const T x_value = x_ptr[col];
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          out_ptr[indices_ptr[p]] += data_ptr[p] * x_value;
        }
      }
    } else {
      std::vector<AccT> accum(static_cast<size_t>(n_rows),
                              Accumulator<T>::zero());
      for (int col = 0; col < n_cols; ++col) {
        const T x_value = x_ptr[col];
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          accum[static_cast<size_t>(indices_ptr[p])] +=
              multiply_accumulate<T>(data_ptr[p], x_value);
        }
      }
      for (int row = 0; row < n_rows; ++row) {
        out_ptr[row] = Accumulator<T>::cast(accum[static_cast<size_t>(row)]);
      }
    }
  });
}

} // namespace

void CSCMatVec::eval_cpu(const std::vector<mx::array> &inputs,
                         std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &x = inputs[3];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error("csc_matvec requires int32 or int64 indices.");
  }

#define DISPATCH_CSC_MATVEC_VALUE(DTYPE, TYPE)                                 \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_matvec_cpu_impl<TYPE, int32_t>(data, indices, indptr, x, out,        \
                                         n_rows_, n_cols_, stream());          \
    } else {                                                                   \
      csc_matvec_cpu_impl<TYPE, int64_t>(data, indices, indptr, x, out,        \
                                         n_rows_, n_cols_, stream());          \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_MATVEC_VALUE(mx::float32, float)
  DISPATCH_CSC_MATVEC_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSC_MATVEC_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_MATVEC_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_MATVEC_VALUE

  throw std::runtime_error("csc_matvec unsupported value dtype.");
}

std::vector<mx::array> CSCMatVec::jvp(const std::vector<mx::array> &primals,
                                      const std::vector<mx::array> &tangents,
                                      const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    if (argnums[i] == 0) {
      terms.push_back(csc_matvec(tangents[i], primals[1], primals[2],
                                 primals[3], n_rows_, n_cols_, stream()));
    } else if (argnums[i] == 3) {
      terms.push_back(csc_matvec(primals[0], primals[1], primals[2],
                                 tangents[i], n_rows_, n_cols_, stream()));
    } else {
      throw std::runtime_error(
          "CSCMatVec JVP is implemented only for data and dense RHS.");
    }
  }
  if (terms.empty()) {
    throw std::runtime_error("CSCMatVec JVP requires at least one tangent.");
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array> CSCMatVec::vjp(const std::vector<mx::array> &primals,
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
      auto x_matrix = mx::reshape(x, mx::Shape{n_cols_, 1}, stream());
      auto cotangent_matrix =
          mx::reshape(cotangents[0], mx::Shape{n_rows_, 1}, stream());
      vjps.push_back(csc_matmul_data_vjp(primals[1], primals[2], x_matrix,
                                         cotangent_matrix, n_rows_, n_cols_,
                                         stream()));
    } else if (argnum == 3) {
      auto data = primals[0].dtype() == mx::complex64
                      ? mx::conjugate(primals[0], stream())
                      : primals[0];
      vjps.push_back(csc_matvec_transpose(data, primals[1], primals[2],
                                          cotangents[0], n_rows_, n_cols_,
                                          stream()));
    } else {
      throw std::runtime_error(
          "CSCMatVec VJP is implemented only for data and dense RHS.");
    }
  }
  return vjps;
}

#ifdef _METAL_
void CSCMatVec::eval_gpu(const std::vector<mx::array> &inputs,
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
  auto &encoder = mx::metal::get_command_encoder(s);

  if (data.dtype() == mx::float32) {
    auto *zero_kernel = device.get_kernel("csc_matvec_zero_float32", lib);
    encoder.set_compute_pipeline_state(zero_kernel);
    encoder.set_output_array(out, 0);
    encoder.set_bytes(n_rows_, 1);
    auto zero_threads = static_cast<size_t>(std::max(n_rows_, 1));
    auto zero_group =
        std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                             MTL::Size(zero_group, 1, 1));

    auto atomic_kernel_name = std::string("csc_matvec_atomic_") +
                              index_kernel_suffix(indices.dtype());
    auto *kernel = device.get_kernel(atomic_kernel_name, lib);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(data, 0);
    encoder.set_input_array(indices, 1);
    encoder.set_input_array(indptr, 2);
    encoder.set_input_array(x, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(n_rows_, 5);
    encoder.set_bytes(n_cols_, 6);
    auto threads = static_cast<size_t>(std::max(n_cols_, 1));
    auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
    return;
  }

  auto kernel_name =
      sparse_kernel_name("csc_matvec_serial", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(x, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(n_cols_, 6);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
#else
void CSCMatVec::eval_gpu(const std::vector<mx::array> &,
                         std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_matvec has no GPU implementation in this build.");
}
#endif

mx::array csc_matvec(const mx::array &data, const mx::array &indices,
                     const mx::array &indptr, const mx::array &x, int n_rows,
                     int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csc_matvec shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csc_matvec data");
  require_rank(indices, 1, "csc_matvec indices");
  require_rank(indptr, 1, "csc_matvec indptr");
  require_rank(x, 1, "csc_matvec x");
  require_same_value_dtype(data, x, "csc_matvec data", "csc_matvec x");
  require_same_index_dtype(indices, indptr, "csc_matvec indices",
                           "csc_matvec indptr");
  require_size(indptr, n_cols + 1, "csc_matvec indptr");
  require_size(x, n_cols, "csc_matvec x");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csc_matvec data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto x_contig = mx::contiguous(x, false, stream);

  return mx::array(mx::Shape{n_rows}, data.dtype(),
                   std::make_shared<CSCMatVec>(stream, n_rows, n_cols),
                   {data_contig, indices_contig, indptr_contig, x_contig});
}

} // namespace mlx_sparse
