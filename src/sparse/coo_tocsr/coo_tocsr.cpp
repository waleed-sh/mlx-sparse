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

    auto sort_rows = [&](CpuRange range) {
      std::vector<size_t> order;
      std::vector<T> sorted_data;
      std::vector<I> sorted_indices;
      for (int row_idx = range.begin; row_idx < range.end; ++row_idx) {
        const auto start = static_cast<size_t>(out_indptr_ptr[row_idx]);
        const auto end = static_cast<size_t>(out_indptr_ptr[row_idx + 1]);
        const auto length = end - start;
        if (length <= 1) {
          continue;
        }
        order.resize(length);
        sorted_data.resize(length);
        sorted_indices.resize(length);
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t lhs, size_t rhs) {
                           return out_indices_ptr[start + lhs] <
                                  out_indices_ptr[start + rhs];
                         });
        for (size_t offset = 0; offset < length; ++offset) {
          const auto src = start + order[offset];
          sorted_data[offset] = out_data_ptr[src];
          sorted_indices[offset] = out_indices_ptr[src];
        }
        std::copy(sorted_data.begin(), sorted_data.end(), out_data_ptr + start);
        std::copy(sorted_indices.begin(), sorted_indices.end(),
                  out_indices_ptr + start);
      }
    };

    auto run_serial = [&]() {
      std::fill(out_indptr_ptr, out_indptr_ptr + n_rows + 1, I{0});
      for (size_t p = 0; p < nnz; ++p) {
        out_indptr_ptr[static_cast<size_t>(row_ptr[p]) + 1] += I{1};
      }
      for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
        out_indptr_ptr[row_idx + 1] += out_indptr_ptr[row_idx];
      }

      std::vector<I> next(out_indptr_ptr, out_indptr_ptr + n_rows);
      for (size_t p = 0; p < nnz; ++p) {
        const auto row_idx = static_cast<size_t>(row_ptr[p]);
        const auto dst = static_cast<size_t>(next[row_idx]++);
        out_data_ptr[dst] = data_ptr[p];
        out_indices_ptr[dst] = col_ptr[p];
      }
      sort_rows({0, n_rows});
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_rows <= 0 || nnz == 0) {
      run_serial();
      return;
    }

    const auto source_ranges = equal_cpu_ranges(static_cast<int>(nnz), workers);
    if (source_ranges.size() <= 1) {
      run_serial();
      return;
    }

    const auto n_partitions = source_ranges.size();
    const size_t counts_stride = static_cast<size_t>(n_rows);
    std::vector<I> local_counts(n_partitions * counts_stride, I{0});
    parallel_for_cpu_ranges_indexed(
        source_ranges, [&](size_t worker, CpuRange range) {
          auto *counts = local_counts.data() + worker * counts_stride;
          for (int p = range.begin; p < range.end; ++p) {
            counts[static_cast<size_t>(row_ptr[p])] += I{1};
          }
        });

    out_indptr_ptr[0] = I{0};
    for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
      I count = I{0};
      for (size_t worker = 0; worker < n_partitions; ++worker) {
        count += local_counts[worker * counts_stride + row_idx];
      }
      out_indptr_ptr[row_idx + 1] = out_indptr_ptr[row_idx] + count;
    }

    std::vector<I> next(n_partitions * counts_stride, I{0});
    for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
      I write = out_indptr_ptr[row_idx];
      for (size_t worker = 0; worker < n_partitions; ++worker) {
        const size_t offset = worker * counts_stride + row_idx;
        next[offset] = write;
        write += local_counts[offset];
      }
    }

    parallel_for_cpu_ranges_indexed(
        source_ranges, [&](size_t worker, CpuRange range) {
          auto *worker_next = next.data() + worker * counts_stride;
          for (int p = range.begin; p < range.end; ++p) {
            const auto row_idx = static_cast<size_t>(row_ptr[p]);
            const auto dst = static_cast<size_t>(worker_next[row_idx]++);
            out_data_ptr[dst] = data_ptr[p];
            out_indices_ptr[dst] = col_ptr[p];
          }
        });

    const auto row_ranges = cpu_ranges_for_output_work(
        compressed_segment_work(out_indptr_ptr, n_rows), workers);
    if (row_ranges.size() <= 1) {
      sort_rows({0, n_rows});
    } else {
      parallel_for_cpu_ranges(row_ranges, sort_rows);
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
