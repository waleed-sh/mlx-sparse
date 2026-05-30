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

#include "sparse/csr_matvec_data_vjp/csr_matvec_data_vjp.h"

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

class CSRMatVecDataVJP : public mx::Primitive {
public:
  CSRMatVecDataVJP(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRMatVecDataVJP"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRMatVecDataVJP &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csr_matvec_data_vjp_cpu_impl(const mx::array &indices,
                                  const mx::array &indptr, const mx::array &x,
                                  const mx::array &cotangent, mx::array &out,
                                  int n_rows, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(x);
  encoder.set_input_array(cotangent);
  encoder.set_output_array(out);

  encoder.dispatch([indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    x = mx::array::unsafe_weak_copy(x),
                    cotangent = mx::array::unsafe_weak_copy(cotangent),
                    out = mx::array::unsafe_weak_copy(out), n_rows]() mutable {
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *x_ptr = x.data<T>();
    const auto *cotangent_ptr = cotangent.data<T>();
    auto *out_ptr = out.data<T>();

    auto compute_rows = [&](CpuRange range) {
      for (int row = range.begin; row < range.end; ++row) {
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          out_ptr[p] = Accumulator<T>::cast(multiply_accumulate<T>(
              cotangent_ptr[row], x_ptr[indices_ptr[p]]));
        }
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_rows <= 0) {
      compute_rows({0, n_rows});
      return;
    }
    const auto ranges =
        cpu_ranges_for_compressed_segments(indptr_ptr, n_rows, workers);
    if (ranges.size() <= 1) {
      compute_rows({0, n_rows});
      return;
    }
    parallel_for_cpu_ranges(ranges, compute_rows);
  });
}

} // namespace

void CSRMatVecDataVJP::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &indices = inputs[0];
  auto &indptr = inputs[1];
  auto &x = inputs[2];
  auto &cotangent = inputs[3];
  auto &out = outputs[0];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error(
        "csr_matvec_data_vjp requires int32 or int64 indices.");
  }

#define DISPATCH_CSR_MATVEC_DATA_VJP_VALUE(DTYPE, TYPE)                        \
  if (x.dtype() == DTYPE) {                                                    \
    if (indices.dtype() == mx::int32) {                                        \
      csr_matvec_data_vjp_cpu_impl<TYPE, int32_t>(                             \
          indices, indptr, x, cotangent, out, n_rows_, stream());              \
    } else {                                                                   \
      csr_matvec_data_vjp_cpu_impl<TYPE, int64_t>(                             \
          indices, indptr, x, cotangent, out, n_rows_, stream());              \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_MATVEC_DATA_VJP_VALUE(mx::float32, float)
  DISPATCH_CSR_MATVEC_DATA_VJP_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_MATVEC_DATA_VJP_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_MATVEC_DATA_VJP_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_MATVEC_DATA_VJP_VALUE

  throw std::runtime_error("csr_matvec_data_vjp unsupported value dtype.");
}

#ifdef _METAL_
void CSRMatVecDataVJP::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &indices = inputs[0];
  auto &indptr = inputs[1];
  auto &x = inputs[2];
  auto &cotangent = inputs[3];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_matvec_data_vjp", x.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(indices, 0);
  encoder.set_input_array(indptr, 1);
  encoder.set_input_array(x, 2);
  encoder.set_input_array(cotangent, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(n_rows_, 5);
  auto threads = static_cast<size_t>(std::max(n_rows_, 1));
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRMatVecDataVJP::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_matvec_data_vjp has no GPU implementation in this build.");
}
#endif

mx::array csr_matvec_data_vjp(const mx::array &indices, const mx::array &indptr,
                              const mx::array &x, const mx::array &cotangent,
                              int n_rows, int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr_matvec_data_vjp shape dimensions must be non-negative.");
  }
  require_rank(indices, 1, "csr_matvec_data_vjp indices");
  require_rank(indptr, 1, "csr_matvec_data_vjp indptr");
  require_rank(x, 1, "csr_matvec_data_vjp x");
  require_rank(cotangent, 1, "csr_matvec_data_vjp cotangent");
  require_same_value_dtype(x, cotangent, "csr_matvec_data_vjp x",
                           "csr_matvec_data_vjp cotangent");
  require_same_index_dtype(indices, indptr, "csr_matvec_data_vjp indices",
                           "csr_matvec_data_vjp indptr");
  require_size(indptr, n_rows + 1, "csr_matvec_data_vjp indptr");
  require_size(x, n_cols, "csr_matvec_data_vjp x");
  require_size(cotangent, n_rows, "csr_matvec_data_vjp cotangent");

  auto stream = mx::to_stream(s);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto x_contig = mx::contiguous(x, false, stream);
  auto cotangent_contig = mx::contiguous(cotangent, false, stream);

  return mx::array(mx::Shape{static_cast<int>(indices.size())}, x.dtype(),
                   std::make_shared<CSRMatVecDataVJP>(stream, n_rows, n_cols),
                   {indices_contig, indptr_contig, x_contig, cotangent_contig});
}

} // namespace mlx_sparse
