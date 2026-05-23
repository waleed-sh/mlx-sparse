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

#include "sparse/csr_matmul_data_vjp/csr_matmul_data_vjp.h"

#include <algorithm>
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

class CSRMatMulDataVJP : public mx::Primitive {
public:
  CSRMatMulDataVJP(mx::Stream stream, int n_rows, int n_cols, int rhs_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols),
        rhs_cols_(rhs_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRMatMulDataVJP"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRMatMulDataVJP &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           rhs_cols_ == rhs.rhs_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
  int rhs_cols_;
};

template <typename T, typename I>
void csr_matmul_data_vjp_cpu_impl(const mx::array &indices,
                                  const mx::array &indptr, const mx::array &rhs,
                                  const mx::array &cotangent, mx::array &out,
                                  int n_rows, int rhs_cols, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(rhs);
  encoder.set_input_array(cotangent);
  encoder.set_output_array(out);

  encoder.dispatch([indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    rhs = mx::array::unsafe_weak_copy(rhs),
                    cotangent = mx::array::unsafe_weak_copy(cotangent),
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    rhs_cols]() mutable {
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *rhs_ptr = rhs.data<T>();
    const auto *cotangent_ptr = cotangent.data<T>();
    auto *out_ptr = out.data<T>();

    for (int row = 0; row < n_rows; ++row) {
      const auto cotangent_offset = static_cast<size_t>(row) * rhs_cols;
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const auto rhs_offset = static_cast<size_t>(indices_ptr[p]) * rhs_cols;
        auto acc = Accumulator<T>::zero();
        for (int k = 0; k < rhs_cols; ++k) {
          acc += multiply_accumulate<T>(cotangent_ptr[cotangent_offset + k],
                                        rhs_ptr[rhs_offset + k]);
        }
        out_ptr[p] = Accumulator<T>::cast(acc);
      }
    }
  });
}

} // namespace

#ifdef _METAL_
void CSRMatMulDataVJP::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &indices = inputs[0];
  auto &indptr = inputs[1];
  auto &rhs = inputs[2];
  auto &cotangent = inputs[3];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_matmul_data_vjp", rhs.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(indices, 0);
  encoder.set_input_array(indptr, 1);
  encoder.set_input_array(rhs, 2);
  encoder.set_input_array(cotangent, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(rhs_cols_, 6);
  auto threads = static_cast<size_t>(std::max(n_rows_, 1));
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRMatMulDataVJP::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_matmul_data_vjp has no GPU implementation in this build.");
}
#endif

void CSRMatMulDataVJP::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &indices = inputs[0];
  auto &indptr = inputs[1];
  auto &rhs = inputs[2];
  auto &cotangent = inputs[3];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error(
        "csr_matmul_data_vjp requires int32 or int64 indices.");
  }

#define DISPATCH_CSR_MATMUL_DATA_VJP_VALUE(DTYPE, TYPE)                        \
  if (rhs.dtype() == DTYPE) {                                                  \
    if (indices.dtype() == mx::int32) {                                        \
      csr_matmul_data_vjp_cpu_impl<TYPE, int32_t>(                             \
          indices, indptr, rhs, cotangent, out, n_rows_, rhs_cols_, stream()); \
    } else {                                                                   \
      csr_matmul_data_vjp_cpu_impl<TYPE, int64_t>(                             \
          indices, indptr, rhs, cotangent, out, n_rows_, rhs_cols_, stream()); \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_MATMUL_DATA_VJP_VALUE(mx::float32, float)
  DISPATCH_CSR_MATMUL_DATA_VJP_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_MATMUL_DATA_VJP_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_MATMUL_DATA_VJP_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_MATMUL_DATA_VJP_VALUE

  throw std::runtime_error("csr_matmul_data_vjp unsupported value dtype.");
}

mx::array csr_matmul_data_vjp(const mx::array &indices, const mx::array &indptr,
                              const mx::array &rhs, const mx::array &cotangent,
                              int n_rows, int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr_matmul_data_vjp shape dimensions must be non-negative.");
  }
  require_rank(indices, 1, "csr_matmul_data_vjp indices");
  require_rank(indptr, 1, "csr_matmul_data_vjp indptr");
  require_rank(rhs, 2, "csr_matmul_data_vjp rhs");
  require_rank(cotangent, 2, "csr_matmul_data_vjp cotangent");
  require_same_value_dtype(rhs, cotangent, "csr_matmul_data_vjp rhs",
                           "csr_matmul_data_vjp cotangent");
  require_same_index_dtype(indices, indptr, "csr_matmul_data_vjp indices",
                           "csr_matmul_data_vjp indptr");
  require_size(indptr, n_rows + 1, "csr_matmul_data_vjp indptr");
  if (rhs.shape(0) != n_cols) {
    throw std::invalid_argument("csr_matmul_data_vjp rhs first dimension must "
                                "equal the sparse matrix column count.");
  }
  if (cotangent.shape(0) != n_rows) {
    throw std::invalid_argument("csr_matmul_data_vjp cotangent first dimension "
                                "must equal the sparse matrix row count.");
  }
  if (rhs.shape(1) != cotangent.shape(1)) {
    throw std::invalid_argument("csr_matmul_data_vjp rhs and cotangent must "
                                "have the same number of columns.");
  }

  auto stream = mx::to_stream(s);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  auto cotangent_contig = mx::contiguous(cotangent, false, stream);
  const int rhs_cols = rhs.shape(1);

  return mx::array(
      mx::Shape{static_cast<int>(indices.size())}, rhs.dtype(),
      std::make_shared<CSRMatMulDataVJP>(stream, n_rows, n_cols, rhs_cols),
      {indices_contig, indptr_contig, rhs_contig, cotangent_contig});
}

} // namespace mlx_sparse
