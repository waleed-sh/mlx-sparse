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

#include "sparse/csr_col_sums/csr_col_sums.h"

#include "sparse/csr_row_sums/csr_row_sums.h"
#include "sparse/csr_transpose/csr_transpose.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "common/autodiff.h"
#include "common/cpu_parallel.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "sparse/csr_matvec_data_vjp/csr_matvec_data_vjp.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

class CSRColSums : public mx::Primitive {
public:
  CSRColSums(mx::Stream stream, int n_rows, int n_cols)
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

  const char *name() const override { return "CSRColSums"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRColSums &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csr_col_sums_cpu_impl(const mx::array &data, const mx::array &indices,
                           const mx::array &indptr, mx::array &out, int n_rows,
                           int n_cols, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    n_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *out_ptr = out.data<T>();

    auto run_serial = [&]() {
      if constexpr (std::is_same_v<AccT, T>) {
        std::fill(out_ptr, out_ptr + n_cols, T{});
        for (int row = 0; row < n_rows; ++row) {
          for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
            out_ptr[indices_ptr[p]] += data_ptr[p];
          }
        }
      } else {
        std::vector<AccT> accum(static_cast<size_t>(n_cols),
                                Accumulator<T>::zero());
        for (int row = 0; row < n_rows; ++row) {
          for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
            accum[static_cast<size_t>(indices_ptr[p])] +=
                static_cast<AccT>(data_ptr[p]);
          }
        }
        for (int col = 0; col < n_cols; ++col) {
          out_ptr[col] = Accumulator<T>::cast(accum[static_cast<size_t>(col)]);
        }
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_rows <= 0 || n_cols <= 0) {
      run_serial();
      return;
    }
    const auto ranges =
        cpu_ranges_for_compressed_segments(indptr_ptr, n_rows, workers);
    if (ranges.size() <= 1) {
      run_serial();
      return;
    }

    const auto n_partitions = ranges.size();
    const size_t stride = static_cast<size_t>(n_cols);
    std::vector<AccT> partial(n_partitions * stride, Accumulator<T>::zero());
    parallel_for_cpu_ranges_indexed(ranges, [&](size_t worker, CpuRange range) {
      auto *accum = partial.data() + worker * stride;
      for (int row = range.begin; row < range.end; ++row) {
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          accum[static_cast<size_t>(indices_ptr[p])] +=
              static_cast<AccT>(data_ptr[p]);
        }
      }
    });

    auto reduce_cols = [&](CpuRange range) {
      for (int col = range.begin; col < range.end; ++col) {
        AccT total = Accumulator<T>::zero();
        for (size_t worker = 0; worker < n_partitions; ++worker) {
          total += partial[worker * stride + col];
        }
        out_ptr[col] = Accumulator<T>::cast(total);
      }
    };
    const auto col_ranges = equal_cpu_ranges(n_cols, workers);
    if (col_ranges.size() <= 1) {
      reduce_cols({0, n_cols});
    } else {
      parallel_for_cpu_ranges(col_ranges, reduce_cols);
    }
  });
}

void validate_csr_reduction_inputs(const mx::array &data,
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
  require_size(indptr, n_rows + 1, op);
  if (indices.size() != data.size()) {
    throw std::invalid_argument(std::string(op) +
                                " data and indices must have equal length.");
  }
}

} // namespace

void CSRColSums::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];

#define DISPATCH_CSR_COL_SUMS(DTYPE, TYPE)                                     \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csr_col_sums_cpu_impl<TYPE, int32_t>(data, indices, indptr, outputs[0],  \
                                           n_rows_, n_cols_, stream());        \
    } else {                                                                   \
      csr_col_sums_cpu_impl<TYPE, int64_t>(data, indices, indptr, outputs[0],  \
                                           n_rows_, n_cols_, stream());        \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSR_COL_SUMS(mx::float32, float)
  DISPATCH_CSR_COL_SUMS(mx::float16, mx::float16_t)
  DISPATCH_CSR_COL_SUMS(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_COL_SUMS(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_COL_SUMS

  throw std::runtime_error("csr_col_sums unsupported value dtype.");
}

std::vector<mx::array> CSRColSums::jvp(const std::vector<mx::array> &primals,
                                       const std::vector<mx::array> &tangents,
                                       const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    require_sparse_value_autodiff_arg(argnums[i], "CSRColSums", "JVP");
    terms.push_back(csr_col_sums(tangents[i], primals[1], primals[2], n_rows_,
                                 n_cols_, stream()));
  }
  if (terms.empty()) {
    throw std::runtime_error("CSRColSums JVP requires at least one tangent.");
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array> CSRColSums::vjp(const std::vector<mx::array> &primals,
                                       const std::vector<mx::array> &cotangents,
                                       const std::vector<int> &argnums,
                                       const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  auto ones = mx::ones(mx::Shape{n_rows_}, primals[0].dtype(), stream());
  for (int argnum : argnums) {
    require_sparse_value_autodiff_arg(argnum, "CSRColSums", "VJP");
    vjps.push_back(csr_matvec_data_vjp(primals[1], primals[2], cotangents[0],
                                       ones, n_rows_, n_cols_, stream()));
  }
  return vjps;
}

#ifdef _METAL_
void CSRColSums::eval_gpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());

  auto &encoder = mx::metal::get_command_encoder(s);
  auto *zero_kernel = device.get_kernel(data.dtype() == mx::complex64
                                            ? "csr_col_sums_zero_complex64"
                                            : "csr_col_sums_zero_float32",
                                        lib);
  encoder.set_compute_pipeline_state(zero_kernel);
  encoder.set_output_array(out, 0);
  encoder.set_bytes(n_cols_, 1);
  auto zero_threads = static_cast<size_t>(std::max(n_cols_, 1));
  auto zero_group =
      std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                           MTL::Size(zero_group, 1, 1));

  auto atomic_kernel_name =
      data.dtype() == mx::complex64
          ? std::string("csr_col_sums_atomic_complex64_") +
                index_kernel_suffix(indices.dtype())
          : std::string("csr_col_sums_atomic_") +
                index_kernel_suffix(indices.dtype());
  auto *kernel = device.get_kernel(atomic_kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(n_rows_, 4);
  encoder.set_bytes(n_cols_, 5);
  auto threads = static_cast<size_t>(std::max(n_rows_, 1));
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRColSums::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_col_sums has no GPU implementation in this build.");
}
#endif

mx::array csr_col_sums(const mx::array &data, const mx::array &indices,
                       const mx::array &indptr, int n_rows, int n_cols,
                       mx::StreamOrDevice s) {
  validate_csr_reduction_inputs(data, indices, indptr, n_rows, n_cols,
                                "csr_col_sums");

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);

  if (stream.device == mx::Device::gpu && data.dtype() != mx::float32 &&
      data.dtype() != mx::complex64) {
    auto [transpose_data, transpose_indices, transpose_indptr] = csr_transpose(
        data_contig, indices_contig, indptr_contig, n_rows, n_cols, stream);
    return csr_row_sums(transpose_data, transpose_indices, transpose_indptr,
                        n_cols, n_rows, stream);
  }

  return mx::array(mx::Shape{n_cols}, data.dtype(),
                   std::make_shared<CSRColSums>(stream, n_rows, n_cols),
                   {data_contig, indices_contig, indptr_contig});
}

} // namespace mlx_sparse
