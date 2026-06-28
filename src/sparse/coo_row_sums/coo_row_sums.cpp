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

#include "sparse/coo_row_sums/coo_row_sums.h"

#include "sparse/coo_tocsr/coo_tocsr.h"
#include "sparse/csr_row_sums/csr_row_sums.h"

#include <algorithm>
#include <stdexcept>
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

class COORowSums : public mx::Primitive {
public:
  COORowSums(mx::Stream stream, int n_rows, int n_cols)
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

  const char *name() const override { return "COORowSums"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COORowSums &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void coo_row_sums_cpu_impl(const mx::array &data, const mx::array &row,
                           mx::array &out, int n_rows, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(row);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    row = mx::array::unsafe_weak_copy(row),
                    out = mx::array::unsafe_weak_copy(out), n_rows]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *row_ptr = row.data<I>();
    auto *out_ptr = out.data<T>();

    const int nnz = static_cast<int>(data.size());
    auto run_serial = [&]() {
      if constexpr (std::is_same_v<AccT, T>) {
        std::fill(out_ptr, out_ptr + n_rows, T{});
        for (size_t p = 0; p < data.size(); ++p) {
          out_ptr[row_ptr[p]] += data_ptr[p];
        }
      } else {
        std::vector<AccT> accum(static_cast<size_t>(n_rows),
                                Accumulator<T>::zero());
        for (size_t p = 0; p < data.size(); ++p) {
          accum[static_cast<size_t>(row_ptr[p])] +=
              static_cast<AccT>(data_ptr[p]);
        }
        for (int r = 0; r < n_rows; ++r) {
          out_ptr[r] = Accumulator<T>::cast(accum[static_cast<size_t>(r)]);
        }
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_rows <= 0 || nnz == 0) {
      run_serial();
      return;
    }
    const auto ranges = equal_cpu_ranges(nnz, workers);
    if (ranges.size() <= 1) {
      run_serial();
      return;
    }

    const auto n_partitions = ranges.size();
    const size_t stride = static_cast<size_t>(n_rows);
    std::vector<AccT> partial(n_partitions * stride, Accumulator<T>::zero());
    parallel_for_cpu_ranges_indexed(ranges, [&](size_t worker, CpuRange range) {
      auto *accum = partial.data() + worker * stride;
      for (int p = range.begin; p < range.end; ++p) {
        accum[static_cast<size_t>(row_ptr[p])] +=
            static_cast<AccT>(data_ptr[p]);
      }
    });

    auto reduce_rows = [&](CpuRange range) {
      for (int row = range.begin; row < range.end; ++row) {
        AccT total = Accumulator<T>::zero();
        for (size_t worker = 0; worker < n_partitions; ++worker) {
          total += partial[worker * stride + row];
        }
        out_ptr[row] = Accumulator<T>::cast(total);
      }
    };
    const auto row_ranges = equal_cpu_ranges(n_rows, workers);
    if (row_ranges.size() <= 1) {
      reduce_rows({0, n_rows});
    } else {
      parallel_for_cpu_ranges(row_ranges, reduce_rows);
    }
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

void COORowSums::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];

#define DISPATCH_COO_ROW_SUMS(DTYPE, TYPE)                                     \
  if (data.dtype() == DTYPE) {                                                 \
    if (row.dtype() == mx::int32) {                                            \
      coo_row_sums_cpu_impl<TYPE, int32_t>(data, row, outputs[0], n_rows_,     \
                                           stream());                          \
    } else {                                                                   \
      coo_row_sums_cpu_impl<TYPE, int64_t>(data, row, outputs[0], n_rows_,     \
                                           stream());                          \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_ROW_SUMS(mx::float32, float)
  DISPATCH_COO_ROW_SUMS(mx::float16, mx::float16_t)
  DISPATCH_COO_ROW_SUMS(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_ROW_SUMS(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_ROW_SUMS

  throw std::runtime_error("coo_row_sums unsupported value dtype.");
}

std::vector<mx::array> COORowSums::jvp(const std::vector<mx::array> &primals,
                                       const std::vector<mx::array> &tangents,
                                       const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    require_sparse_value_autodiff_arg(argnums[i], "COORowSums", "JVP");
    terms.push_back(coo_row_sums(tangents[i], primals[1], primals[2], n_rows_,
                                 n_cols_, stream()));
  }
  if (terms.empty()) {
    throw std::runtime_error("COORowSums JVP requires at least one tangent.");
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array> COORowSums::vjp(const std::vector<mx::array> &primals,
                                       const std::vector<mx::array> &cotangents,
                                       const std::vector<int> &argnums,
                                       const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  for (int argnum : argnums) {
    require_sparse_value_autodiff_arg(argnum, "COORowSums", "VJP");
    vjps.push_back(
        sparse_vector_cotangent_gather(cotangents[0], primals[1], stream()));
  }
  return vjps;
}

#ifdef _METAL_
void COORowSums::eval_gpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  auto *zero_kernel = device.get_kernel(data.dtype() == mx::complex64
                                            ? "coo_row_sums_zero_complex64"
                                            : "coo_row_sums_zero_float32",
                                        lib);
  encoder.set_compute_pipeline_state(zero_kernel);
  encoder.set_output_array(out, 0);
  encoder.set_bytes(n_rows_, 1);
  auto zero_threads = static_cast<size_t>(std::max(n_rows_, 1));
  auto zero_group =
      std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                           MTL::Size(zero_group, 1, 1));

  auto kernel_name = data.dtype() == mx::complex64
                         ? std::string("coo_row_sums_atomic_complex64_") +
                               index_kernel_suffix(row.dtype())
                         : std::string("coo_row_sums_atomic_") +
                               index_kernel_suffix(row.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(row, 1);
  encoder.set_output_array(out, 2);
  auto nnz = static_cast<int>(data.size());
  encoder.set_bytes(nnz, 3);
  encoder.set_bytes(n_rows_, 4);
  auto threads = std::max<size_t>(data.size(), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COORowSums::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_row_sums has no GPU implementation in this build.");
}
#endif

mx::array coo_row_sums(const mx::array &data, const mx::array &row,
                       const mx::array &col, int n_rows, int n_cols,
                       mx::StreamOrDevice s) {
  validate_coo_reduction_inputs(data, row, col, n_rows, n_cols, "coo_row_sums");

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto row_contig = mx::contiguous(row, false, stream);
  auto col_contig = mx::contiguous(col, false, stream);

  if (stream.device == mx::Device::gpu && data.dtype() != mx::float32 &&
      data.dtype() != mx::complex64) {
    auto [csr_data, csr_indices, csr_indptr] =
        coo_tocsr(data_contig, row_contig, col_contig, n_rows, n_cols, stream);
    return csr_row_sums(csr_data, csr_indices, csr_indptr, n_rows, n_cols,
                        stream);
  }

  return mx::array(mx::Shape{n_rows}, data.dtype(),
                   std::make_shared<COORowSums>(stream, n_rows, n_cols),
                   {data_contig, row_contig, col_contig});
}

} // namespace mlx_sparse
