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

#include "sparse/coo_tocsr/coo_tocsr.h"

#include <algorithm>
#include <numeric>
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

class COOToCSR : public mx::Primitive {
public:
  COOToCSR(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOToCSR"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOToCSR &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void coo_tocsr_cpu_impl(const mx::array &data, const mx::array &row,
                        const mx::array &col, mx::array &out_data,
                        mx::array &out_indices, mx::array &out_indptr,
                        int n_rows, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));
  out_indptr.set_data(mx::allocator::malloc(out_indptr.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(row);
  encoder.set_input_array(col);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_indices);
  encoder.set_output_array(out_indptr);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    row = mx::array::unsafe_weak_copy(row),
                    col = mx::array::unsafe_weak_copy(col),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_indices = mx::array::unsafe_weak_copy(out_indices),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    n_rows]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *row_ptr = row.data<I>();
    const auto *col_ptr = col.data<I>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_indices_ptr = out_indices.data<I>();
    auto *out_indptr_ptr = out_indptr.data<I>();
    const auto nnz = data.size();

    std::vector<size_t> order(nnz);
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
      if (row_ptr[lhs] != row_ptr[rhs]) {
        return row_ptr[lhs] < row_ptr[rhs];
      }
      return col_ptr[lhs] < col_ptr[rhs];
    });

    std::fill(out_indptr_ptr, out_indptr_ptr + n_rows + 1, I{0});
    for (size_t k = 0; k < nnz; ++k) {
      const auto src = order[k];
      out_data_ptr[k] = data_ptr[src];
      out_indices_ptr[k] = col_ptr[src];
      out_indptr_ptr[static_cast<size_t>(row_ptr[src]) + 1] += I{1};
    }
    for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
      out_indptr_ptr[row_idx + 1] += out_indptr_ptr[row_idx];
    }
  });
}

} // namespace

void COOToCSR::eval_cpu(const std::vector<mx::array> &inputs,
                        std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];

  if (row.dtype() != mx::int32 && row.dtype() != mx::int64) {
    throw std::runtime_error("coo_tocsr requires int32 or int64 coordinates.");
  }

#define DISPATCH_COO_TO_CSR_VALUE(DTYPE, TYPE)                                 \
  if (data.dtype() == DTYPE) {                                                 \
    if (row.dtype() == mx::int32) {                                            \
      coo_tocsr_cpu_impl<TYPE, int32_t>(data, row, col, outputs[0],            \
                                        outputs[1], outputs[2], n_rows_,       \
                                        stream());                             \
    } else {                                                                   \
      coo_tocsr_cpu_impl<TYPE, int64_t>(data, row, col, outputs[0],            \
                                        outputs[1], outputs[2], n_rows_,       \
                                        stream());                             \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_TO_CSR_VALUE(mx::float32, float)
  DISPATCH_COO_TO_CSR_VALUE(mx::float16, mx::float16_t)
  DISPATCH_COO_TO_CSR_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_TO_CSR_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_TO_CSR_VALUE

  throw std::runtime_error("coo_tocsr unsupported value dtype.");
}

#ifdef _METAL_
void COOToCSR::eval_gpu(const std::vector<mx::array> &inputs,
                        std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  auto &out_data = outputs[0];
  auto &out_indices = outputs[1];
  auto &out_indptr = outputs[2];

  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));
  out_indptr.set_data(mx::allocator::malloc(out_indptr.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());

  auto &encoder = mx::metal::get_command_encoder(s);
  auto rank_kernel_name =
      sparse_kernel_name("coo_tocsr_rank", data.dtype(), row.dtype());
  auto *rank_kernel = device.get_kernel(rank_kernel_name, lib);
  encoder.set_compute_pipeline_state(rank_kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(row, 1);
  encoder.set_input_array(col, 2);
  encoder.set_output_array(out_data, 3);
  encoder.set_output_array(out_indices, 4);
  encoder.set_bytes(static_cast<uint32_t>(data.size()), 5);

  auto rank_threads = std::max<size_t>(data.size(), 1);
  auto rank_group =
      std::min(rank_threads, rank_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(rank_threads, 1, 1),
                           MTL::Size(rank_group, 1, 1));

  auto indptr_kernel_name =
      std::string("coo_tocsr_indptr_") + index_kernel_suffix(row.dtype());
  auto *indptr_kernel = device.get_kernel(indptr_kernel_name, lib);
  encoder.set_compute_pipeline_state(indptr_kernel);
  encoder.set_input_array(row, 0);
  encoder.set_output_array(out_indptr, 1);
  encoder.set_bytes(static_cast<uint32_t>(data.size()), 2);
  encoder.set_bytes(static_cast<uint32_t>(n_rows_), 3);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
#else
void COOToCSR::eval_gpu(const std::vector<mx::array> &,
                        std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_tocsr has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
coo_tocsr(const mx::array &data, const mx::array &row, const mx::array &col,
          int n_rows, int n_cols, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "coo_tocsr shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "coo_tocsr data");
  require_rank(row, 1, "coo_tocsr row");
  require_rank(col, 1, "coo_tocsr col");
  require_supported_value_dtype(data, "coo_tocsr data");
  require_same_index_dtype(row, col, "coo_tocsr row", "coo_tocsr col");
  if (row.size() != data.size() || col.size() != data.size()) {
    throw std::invalid_argument(
        "coo_tocsr data, row, and col must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto row_contig = mx::contiguous(row, false, stream);
  auto col_contig = mx::contiguous(col, false, stream);

  auto primitive = std::make_shared<COOToCSR>(stream, n_rows, n_cols);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{static_cast<int>(data.size())},
       mx::Shape{static_cast<int>(data.size())}, mx::Shape{n_rows + 1}},
      {data.dtype(), col.dtype(), row.dtype()}, primitive,
      {data_contig, row_contig, col_contig});

  return {outputs[0], outputs[1], outputs[2]};
}

} // namespace mlx_sparse
