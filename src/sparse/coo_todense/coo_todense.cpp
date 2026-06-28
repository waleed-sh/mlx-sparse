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

#include "sparse/coo_todense/coo_todense.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "common/autodiff.h"
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

class COOToDense : public mx::Primitive {
public:
  COOToDense(mx::Stream stream, int n_rows, int n_cols)
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

  const char *name() const override { return "COOToDense"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOToDense &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void coo_todense_cpu_impl(const mx::array &data, const mx::array &row,
                          const mx::array &col, mx::array &out, int n_rows,
                          int n_cols, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(row);
  encoder.set_input_array(col);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    row = mx::array::unsafe_weak_copy(row),
                    col = mx::array::unsafe_weak_copy(col),
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    n_cols]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *data_ptr = data.data<T>();
    const auto *row_ptr = row.data<I>();
    const auto *col_ptr = col.data<I>();
    auto *out_ptr = out.data<T>();

    const int workers = configured_cpu_worker_count();
    if (workers > 1 &&
        out.size() <= static_cast<size_t>(std::numeric_limits<int>::max())) {
      parallel_for_cpu_ranges(
          equal_cpu_ranges(static_cast<int>(out.size()), workers),
          [&](CpuRange range) {
            std::fill(out_ptr + range.begin, out_ptr + range.end, T{});
          });
    } else {
      std::fill(out_ptr, out_ptr + out.size(), T{});
    }

    const int nnz = static_cast<int>(data.size());
    if constexpr (std::is_same_v<AccT, T>) {
      for (int p = 0; p < nnz; ++p) {
        out_ptr[static_cast<size_t>(row_ptr[p]) * n_cols + col_ptr[p]] +=
            data_ptr[p];
      }
    } else {
      std::vector<AccT> accum(out.size(), Accumulator<T>::zero());
      auto fill_range = [&](CpuRange range) {
        for (int p = range.begin; p < range.end; ++p) {
          accum[static_cast<size_t>(row_ptr[p]) * n_cols + col_ptr[p]] +=
              static_cast<AccT>(data_ptr[p]);
        }
      };
      if (workers <= 1 || nnz <= 0) {
        fill_range({0, nnz});
      } else {
        fill_range({0, nnz});
      }
      for (size_t i = 0; i < out.size(); ++i) {
        out_ptr[i] = Accumulator<T>::cast(accum[i]);
      }
    }
  });
}

void validate_coo_todense_inputs(const mx::array &data, const mx::array &row,
                                 const mx::array &col, int n_rows, int n_cols) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "coo_todense shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "coo_todense data");
  require_rank(row, 1, "coo_todense row");
  require_rank(col, 1, "coo_todense col");
  require_supported_value_dtype(data, "coo_todense data");
  require_same_index_dtype(row, col, "coo_todense row", "coo_todense col");
  if (row.size() != data.size() || col.size() != data.size()) {
    throw std::invalid_argument(
        "coo_todense data, row, and col must have equal length.");
  }
}

} // namespace

void COOToDense::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  auto &out = outputs[0];

  if (row.dtype() != mx::int32 && row.dtype() != mx::int64) {
    throw std::runtime_error("coo_todense requires int32 or int64 indices.");
  }

#define DISPATCH_COO_TODENSE(DTYPE, TYPE)                                      \
  if (data.dtype() == DTYPE) {                                                 \
    if (row.dtype() == mx::int32) {                                            \
      coo_todense_cpu_impl<TYPE, int32_t>(data, row, col, out, n_rows_,        \
                                          n_cols_, stream());                  \
    } else {                                                                   \
      coo_todense_cpu_impl<TYPE, int64_t>(data, row, col, out, n_rows_,        \
                                          n_cols_, stream());                  \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_TODENSE(mx::float32, float)
  DISPATCH_COO_TODENSE(mx::float16, mx::float16_t)
  DISPATCH_COO_TODENSE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_TODENSE(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_TODENSE

  throw std::runtime_error("coo_todense unsupported value dtype.");
}

std::vector<mx::array> COOToDense::jvp(const std::vector<mx::array> &primals,
                                       const std::vector<mx::array> &tangents,
                                       const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  terms.reserve(argnums.size());
  for (size_t i = 0; i < argnums.size(); ++i) {
    require_sparse_value_autodiff_arg(argnums[i], "COOToDense", "JVP");
    terms.push_back(coo_todense(tangents[i], primals[1], primals[2], n_rows_,
                                n_cols_, stream()));
  }
  if (terms.empty()) {
    throw std::runtime_error("COOToDense JVP requires at least one tangent.");
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array> COOToDense::vjp(const std::vector<mx::array> &primals,
                                       const std::vector<mx::array> &cotangents,
                                       const std::vector<int> &argnums,
                                       const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  for (int argnum : argnums) {
    require_sparse_value_autodiff_arg(argnum, "COOToDense", "VJP");
    vjps.push_back(sparse_dense_cotangent_gather(
        cotangents[0], primals[1], primals[2], n_cols_, stream()));
  }
  return vjps;
}

#ifdef _METAL_
void COOToDense::eval_gpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("coo_todense", data.dtype(), row.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(row, 1);
  encoder.set_input_array(col, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(n_rows_, 4);
  encoder.set_bytes(n_cols_, 5);
  auto nnz = static_cast<int>(data.size());
  encoder.set_bytes(nnz, 6);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
#else
void COOToDense::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_todense has no GPU implementation in this build.");
}
#endif

mx::array coo_todense(const mx::array &data, const mx::array &row,
                      const mx::array &col, int n_rows, int n_cols,
                      mx::StreamOrDevice s) {
  validate_coo_todense_inputs(data, row, col, n_rows, n_cols);

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto row_contig = mx::contiguous(row, false, stream);
  auto col_contig = mx::contiguous(col, false, stream);

  return mx::array(mx::Shape{n_rows, n_cols}, data.dtype(),
                   std::make_shared<COOToDense>(stream, n_rows, n_cols),
                   {data_contig, row_contig, col_contig});
}

} // namespace mlx_sparse
