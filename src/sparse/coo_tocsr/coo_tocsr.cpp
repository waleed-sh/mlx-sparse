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
#include <tuple>
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

class COOToCSR : public mx::Primitive {
public:
  COOToCSR(mx::Stream stream, int n_rows, int n_cols)
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
                             const std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOToCSR"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOToCSR &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

class COOToCSRDataVJP : public mx::Primitive {
public:
  COOToCSRDataVJP(mx::Stream stream, int n_rows)
      : Primitive(stream), n_rows_(n_rows) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOToCSRDataVJP"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOToCSRDataVJP &>(other);
    return n_rows_ == rhs.n_rows_;
  }

private:
  int n_rows_;
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

template <typename T, typename I>
void coo_tocsr_data_vjp_cpu_impl(const mx::array &cotangent,
                                 const mx::array &row, const mx::array &col,
                                 const mx::array &out_indices,
                                 const mx::array &out_indptr, mx::array &out,
                                 int n_rows, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(cotangent);
  encoder.set_input_array(row);
  encoder.set_input_array(col);
  encoder.set_input_array(out_indices);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out);

  encoder.dispatch([cotangent = mx::array::unsafe_weak_copy(cotangent),
                    row = mx::array::unsafe_weak_copy(row),
                    col = mx::array::unsafe_weak_copy(col),
                    out_indices = mx::array::unsafe_weak_copy(out_indices),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    out = mx::array::unsafe_weak_copy(out), n_rows]() mutable {
    const auto *cotangent_ptr = cotangent.data<T>();
    const auto *row_ptr = row.data<I>();
    const auto *col_ptr = col.data<I>();
    const auto *out_indices_ptr = out_indices.data<I>();
    const auto *out_indptr_ptr = out_indptr.data<I>();
    auto *out_ptr = out.data<T>();
    const auto nnz = row.size();

    auto compute_range = [&](CpuRange range) {
      for (int p = range.begin; p < range.end; ++p) {
        const auto r = row_ptr[p];
        const auto c = col_ptr[p];
        if (r < I{0} || static_cast<int>(r) >= n_rows) {
          out_ptr[p] = T(0);
          continue;
        }

        size_t duplicate_ordinal = 0;
        for (int q = 0; q < p; ++q) {
          if (row_ptr[q] == r && col_ptr[q] == c) {
            duplicate_ordinal += 1;
          }
        }

        size_t seen = 0;
        T value = T(0);
        const auto start = static_cast<size_t>(out_indptr_ptr[r]);
        const auto end = static_cast<size_t>(out_indptr_ptr[r + I{1}]);
        for (size_t dst = start; dst < end; ++dst) {
          if (out_indices_ptr[dst] != c) {
            continue;
          }
          if (seen == duplicate_ordinal) {
            value = cotangent_ptr[dst];
            break;
          }
          seen += 1;
        }
        out_ptr[p] = value;
      }
    };

    const int workers = configured_cpu_worker_count();
    const auto ranges =
        equal_cpu_ranges(static_cast<int>(nnz), std::max(workers, 1));
    if (ranges.size() <= 1) {
      compute_range({0, static_cast<int>(nnz)});
    } else {
      parallel_for_cpu_ranges(ranges, compute_range);
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

std::vector<mx::array> COOToCSR::jvp(const std::vector<mx::array> &primals,
                                     const std::vector<mx::array> &tangents,
                                     const std::vector<int> &argnums) {
  if (argnums.empty()) {
    throw std::runtime_error("COOToCSR JVP requires a value tangent.");
  }
  require_sparse_value_autodiff_arg(argnums[0], "COOToCSR", "JVP");
  auto first_tangent = coo_tocsr(tangents[0], primals[1], primals[2], n_rows_,
                                 n_cols_, stream());
  auto data_tangent = std::get<0>(first_tangent);
  for (size_t i = 0; i < argnums.size(); ++i) {
    if (i == 0) {
      continue;
    }
    require_sparse_value_autodiff_arg(argnums[i], "COOToCSR", "JVP");
    auto tangent_outputs = coo_tocsr(tangents[i], primals[1], primals[2],
                                     n_rows_, n_cols_, stream());
    auto tangent_data = std::get<0>(tangent_outputs);
    data_tangent = mx::add(data_tangent, tangent_data, stream());
  }
  return {data_tangent,
          mx::zeros(mx::Shape{static_cast<int>(primals[2].size())},
                    primals[2].dtype(), stream()),
          mx::zeros(mx::Shape{n_rows_ + 1}, primals[1].dtype(), stream())};
}

std::vector<mx::array> COOToCSR::vjp(const std::vector<mx::array> &primals,
                                     const std::vector<mx::array> &cotangents,
                                     const std::vector<int> &argnums,
                                     const std::vector<mx::array> &outputs) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  for (int argnum : argnums) {
    require_sparse_value_autodiff_arg(argnum, "COOToCSR", "VJP");
    vjps.push_back(mx::array(
        mx::Shape{static_cast<int>(primals[0].size())}, primals[0].dtype(),
        std::make_shared<COOToCSRDataVJP>(stream(), n_rows_),
        {mx::contiguous(cotangents[0], false, stream()),
         mx::contiguous(primals[1], false, stream()),
         mx::contiguous(primals[2], false, stream()),
         mx::contiguous(outputs[1], false, stream()),
         mx::contiguous(outputs[2], false, stream())}));
  }
  return vjps;
}

void COOToCSRDataVJP::eval_cpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &cotangent = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  auto &out_indices = inputs[3];
  auto &out_indptr = inputs[4];

  if (row.dtype() != mx::int32 && row.dtype() != mx::int64) {
    throw std::runtime_error(
        "coo_tocsr_data_vjp requires int32 or int64 coordinates.");
  }

#define DISPATCH_COO_TO_CSR_DATA_VJP(DTYPE, TYPE)                              \
  if (cotangent.dtype() == DTYPE) {                                            \
    if (row.dtype() == mx::int32) {                                            \
      coo_tocsr_data_vjp_cpu_impl<TYPE, int32_t>(                              \
          cotangent, row, col, out_indices, out_indptr, outputs[0], n_rows_,   \
          stream());                                                           \
    } else {                                                                   \
      coo_tocsr_data_vjp_cpu_impl<TYPE, int64_t>(                              \
          cotangent, row, col, out_indices, out_indptr, outputs[0], n_rows_,   \
          stream());                                                           \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_TO_CSR_DATA_VJP(mx::float32, float)
  DISPATCH_COO_TO_CSR_DATA_VJP(mx::float16, mx::float16_t)
  DISPATCH_COO_TO_CSR_DATA_VJP(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_TO_CSR_DATA_VJP(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_TO_CSR_DATA_VJP

  throw std::runtime_error("coo_tocsr_data_vjp unsupported value dtype.");
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

void COOToCSRDataVJP::eval_gpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &cotangent = inputs[0];
  auto &row = inputs[1];
  auto &col = inputs[2];
  auto &out_indices = inputs[3];
  auto &out_indptr = inputs[4];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  auto kernel_name =
      sparse_kernel_name("coo_tocsr_data_vjp", cotangent.dtype(), row.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(cotangent, 0);
  encoder.set_input_array(row, 1);
  encoder.set_input_array(col, 2);
  encoder.set_input_array(out_indices, 3);
  encoder.set_input_array(out_indptr, 4);
  encoder.set_output_array(out, 5);
  encoder.set_bytes(static_cast<int>(row.size()), 6);
  encoder.set_bytes(n_rows_, 7);

  auto threads = std::max<size_t>(row.size(), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOToCSR::eval_gpu(const std::vector<mx::array> &,
                        std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_tocsr has no GPU implementation in this build.");
}

void COOToCSRDataVJP::eval_gpu(const std::vector<mx::array> &,
                               std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_tocsr_data_vjp has no GPU implementation in this build.");
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
