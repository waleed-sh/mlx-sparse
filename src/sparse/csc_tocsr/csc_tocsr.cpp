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

#include "sparse/csc_tocsr/csc_tocsr.h"

#include <algorithm>
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

class CSCToCSR : public mx::Primitive {
public:
  CSCToCSR(mx::Stream stream, int n_rows, int n_cols)
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

  const char *name() const override { return "CSCToCSR"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCToCSR &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

class CSCToCSRDataVJP : public mx::Primitive {
public:
  CSCToCSRDataVJP(mx::Stream stream, int n_rows, int n_cols)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSCToCSRDataVJP"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSCToCSRDataVJP &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_;
  }

private:
  int n_rows_;
  int n_cols_;
};

template <typename T, typename I>
void csc_tocsr_cpu_impl(const mx::array &data, const mx::array &indices,
                        const mx::array &indptr, mx::array &out_data,
                        mx::array &out_indices, mx::array &out_indptr,
                        int n_rows, int n_cols, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));
  out_indptr.set_data(mx::allocator::malloc(out_indptr.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_indices);
  encoder.set_output_array(out_indptr);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_indices = mx::array::unsafe_weak_copy(out_indices),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    n_rows, n_cols]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_indices_ptr = out_indices.data<I>();
    auto *out_indptr_ptr = out_indptr.data<I>();
    const auto nnz = data.size();

    auto run_serial = [&]() {
      std::fill(out_indptr_ptr, out_indptr_ptr + n_rows + 1, I{0});
      for (size_t p = 0; p < nnz; ++p) {
        out_indptr_ptr[static_cast<size_t>(indices_ptr[p]) + 1] += I{1};
      }
      for (int row = 0; row < n_rows; ++row) {
        out_indptr_ptr[row + 1] += out_indptr_ptr[row];
      }

      std::vector<I> next(out_indptr_ptr, out_indptr_ptr + n_rows);
      for (int col = 0; col < n_cols; ++col) {
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          const auto row = static_cast<size_t>(indices_ptr[p]);
          const auto dst = static_cast<size_t>(next[row]++);
          out_data_ptr[dst] = data_ptr[p];
          out_indices_ptr[dst] = static_cast<I>(col);
        }
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_rows <= 0 || n_cols <= 0) {
      run_serial();
      return;
    }

    const auto ranges =
        cpu_ranges_for_compressed_segments(indptr_ptr, n_cols, workers);
    if (ranges.size() <= 1) {
      run_serial();
      return;
    }

    const auto n_partitions = ranges.size();
    const size_t counts_stride = static_cast<size_t>(n_rows);
    std::vector<I> local_counts(n_partitions * counts_stride, I{0});
    parallel_for_cpu_ranges_indexed(ranges, [&](size_t worker, CpuRange range) {
      auto *counts = local_counts.data() + worker * counts_stride;
      for (int col = range.begin; col < range.end; ++col) {
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          counts[static_cast<size_t>(indices_ptr[p])] += I{1};
        }
      }
    });

    out_indptr_ptr[0] = I{0};
    for (int row = 0; row < n_rows; ++row) {
      I count = I{0};
      for (size_t worker = 0; worker < n_partitions; ++worker) {
        count += local_counts[worker * counts_stride + row];
      }
      out_indptr_ptr[row + 1] = out_indptr_ptr[row] + count;
    }

    std::vector<I> next(n_partitions * counts_stride, I{0});
    for (int row = 0; row < n_rows; ++row) {
      I write = out_indptr_ptr[row];
      for (size_t worker = 0; worker < n_partitions; ++worker) {
        const size_t offset = worker * counts_stride + row;
        next[offset] = write;
        write += local_counts[offset];
      }
    }

    parallel_for_cpu_ranges_indexed(ranges, [&](size_t worker, CpuRange range) {
      auto *worker_next = next.data() + worker * counts_stride;
      for (int col = range.begin; col < range.end; ++col) {
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          const auto row = static_cast<size_t>(indices_ptr[p]);
          const auto dst = static_cast<size_t>(worker_next[row]++);
          out_data_ptr[dst] = data_ptr[p];
          out_indices_ptr[dst] = static_cast<I>(col);
        }
      }
    });
  });
}

template <typename T, typename I>
void csc_tocsr_data_vjp_cpu_impl(const mx::array &cotangent,
                                 const mx::array &indices,
                                 const mx::array &indptr,
                                 const mx::array &out_indices,
                                 const mx::array &out_indptr, mx::array &out,
                                 int n_rows, int n_cols, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(cotangent);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(out_indices);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out);

  encoder.dispatch([cotangent = mx::array::unsafe_weak_copy(cotangent),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out_indices = mx::array::unsafe_weak_copy(out_indices),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    n_cols]() mutable {
    const auto *cotangent_ptr = cotangent.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *out_indices_ptr = out_indices.data<I>();
    const auto *out_indptr_ptr = out_indptr.data<I>();
    auto *out_ptr = out.data<T>();

    auto compute_cols = [&](CpuRange range) {
      for (int col = range.begin; col < range.end; ++col) {
        const auto start_p = static_cast<size_t>(indptr_ptr[col]);
        const auto end_p = static_cast<size_t>(indptr_ptr[col + 1]);
        for (size_t p = start_p; p < end_p; ++p) {
          const auto row = indices_ptr[p];
          if (row < I{0} || static_cast<int>(row) >= n_rows) {
            out_ptr[p] = T(0);
            continue;
          }

          size_t duplicate_ordinal = 0;
          for (size_t q = start_p; q < p; ++q) {
            if (indices_ptr[q] == row) {
              duplicate_ordinal += 1;
            }
          }

          size_t seen = 0;
          T value = T(0);
          const auto row_start = static_cast<size_t>(out_indptr_ptr[row]);
          const auto row_end = static_cast<size_t>(out_indptr_ptr[row + I{1}]);
          for (size_t dst = row_start; dst < row_end; ++dst) {
            if (out_indices_ptr[dst] != static_cast<I>(col)) {
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
      }
    };

    const int workers = configured_cpu_worker_count();
    const auto ranges = equal_cpu_ranges(n_cols, std::max(workers, 1));
    if (ranges.size() <= 1) {
      compute_cols({0, n_cols});
    } else {
      parallel_for_cpu_ranges(ranges, compute_cols);
    }
  });
}

} // namespace

void CSCToCSR::eval_cpu(const std::vector<mx::array> &inputs,
                        std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error("csc_tocsr requires int32 or int64 indices.");
  }

#define DISPATCH_CSC_TO_CSR_VALUE(DTYPE, TYPE)                                 \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      csc_tocsr_cpu_impl<TYPE, int32_t>(data, indices, indptr, outputs[0],     \
                                        outputs[1], outputs[2], n_rows_,       \
                                        n_cols_, stream());                    \
    } else {                                                                   \
      csc_tocsr_cpu_impl<TYPE, int64_t>(data, indices, indptr, outputs[0],     \
                                        outputs[1], outputs[2], n_rows_,       \
                                        n_cols_, stream());                    \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_TO_CSR_VALUE(mx::float32, float)
  DISPATCH_CSC_TO_CSR_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSC_TO_CSR_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_TO_CSR_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_TO_CSR_VALUE

  throw std::runtime_error("csc_tocsr unsupported value dtype.");
}

std::vector<mx::array> CSCToCSR::jvp(const std::vector<mx::array> &primals,
                                     const std::vector<mx::array> &tangents,
                                     const std::vector<int> &argnums) {
  if (argnums.empty()) {
    throw std::runtime_error("CSCToCSR JVP requires a value tangent.");
  }
  require_sparse_value_autodiff_arg(argnums[0], "CSCToCSR", "JVP");
  auto first_tangent = csc_tocsr(tangents[0], primals[1], primals[2], n_rows_,
                                 n_cols_, stream());
  auto data_tangent = std::get<0>(first_tangent);
  for (size_t i = 0; i < argnums.size(); ++i) {
    if (i == 0) {
      continue;
    }
    require_sparse_value_autodiff_arg(argnums[i], "CSCToCSR", "JVP");
    auto tangent_outputs = csc_tocsr(tangents[i], primals[1], primals[2],
                                     n_rows_, n_cols_, stream());
    auto tangent_data = std::get<0>(tangent_outputs);
    data_tangent = mx::add(data_tangent, tangent_data, stream());
  }
  return {data_tangent,
          mx::zeros(mx::Shape{static_cast<int>(primals[1].size())},
                    primals[1].dtype(), stream()),
          mx::zeros(mx::Shape{n_rows_ + 1}, primals[2].dtype(), stream())};
}

std::vector<mx::array> CSCToCSR::vjp(const std::vector<mx::array> &primals,
                                     const std::vector<mx::array> &cotangents,
                                     const std::vector<int> &argnums,
                                     const std::vector<mx::array> &outputs) {
  std::vector<mx::array> vjps;
  vjps.reserve(argnums.size());
  for (int argnum : argnums) {
    require_sparse_value_autodiff_arg(argnum, "CSCToCSR", "VJP");
    vjps.push_back(mx::array(
        mx::Shape{static_cast<int>(primals[0].size())}, primals[0].dtype(),
        std::make_shared<CSCToCSRDataVJP>(stream(), n_rows_, n_cols_),
        {mx::contiguous(cotangents[0], false, stream()),
         mx::contiguous(primals[1], false, stream()),
         mx::contiguous(primals[2], false, stream()),
         mx::contiguous(outputs[1], false, stream()),
         mx::contiguous(outputs[2], false, stream())}));
  }
  return vjps;
}

void CSCToCSRDataVJP::eval_cpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &cotangent = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out_indices = inputs[3];
  auto &out_indptr = inputs[4];

  if (indices.dtype() != mx::int32 && indices.dtype() != mx::int64) {
    throw std::runtime_error(
        "csc_tocsr_data_vjp requires int32 or int64 indices.");
  }

#define DISPATCH_CSC_TO_CSR_DATA_VJP(DTYPE, TYPE)                              \
  if (cotangent.dtype() == DTYPE) {                                            \
    if (indices.dtype() == mx::int32) {                                        \
      csc_tocsr_data_vjp_cpu_impl<TYPE, int32_t>(                              \
          cotangent, indices, indptr, out_indices, out_indptr, outputs[0],     \
          n_rows_, n_cols_, stream());                                         \
    } else {                                                                   \
      csc_tocsr_data_vjp_cpu_impl<TYPE, int64_t>(                              \
          cotangent, indices, indptr, out_indices, out_indptr, outputs[0],     \
          n_rows_, n_cols_, stream());                                         \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_TO_CSR_DATA_VJP(mx::float32, float)
  DISPATCH_CSC_TO_CSR_DATA_VJP(mx::float16, mx::float16_t)
  DISPATCH_CSC_TO_CSR_DATA_VJP(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_TO_CSR_DATA_VJP(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_TO_CSR_DATA_VJP

  throw std::runtime_error("csc_tocsr_data_vjp unsupported value dtype.");
}

#ifdef _METAL_
void CSCToCSR::eval_gpu(const std::vector<mx::array> &inputs,
                        std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
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

  mx::array counts(
      mx::allocator::malloc(static_cast<size_t>(n_rows_ + 1) * sizeof(int32_t)),
      mx::Shape{n_rows_ + 1}, mx::int32);
  mx::array next(
      mx::allocator::malloc(static_cast<size_t>(n_rows_ + 1) * sizeof(int32_t)),
      mx::Shape{n_rows_ + 1}, mx::int32);

  auto *zero_kernel = device.get_kernel("csc_tocsr_zero_offsets", lib);
  encoder.set_compute_pipeline_state(zero_kernel);
  encoder.set_output_array(counts, 0);
  encoder.set_bytes(n_rows_, 1);
  auto zero_threads = static_cast<size_t>(std::max(n_rows_ + 1, 1));
  auto zero_group =
      std::min(zero_threads, zero_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(zero_threads, 1, 1),
                           MTL::Size(zero_group, 1, 1));

  auto count_kernel_name =
      std::string("csc_tocsr_count_") + index_kernel_suffix(indices.dtype());
  auto *count_kernel = device.get_kernel(count_kernel_name, lib);
  encoder.set_compute_pipeline_state(count_kernel);
  encoder.set_input_array(indices, 0);
  encoder.set_output_array(counts, 1);
  encoder.set_bytes(static_cast<int>(data.size()), 2);
  auto count_threads = std::max<size_t>(data.size(), 1);
  auto count_group =
      std::min(count_threads, count_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(count_threads, 1, 1),
                           MTL::Size(count_group, 1, 1));

  auto prefix_kernel_name =
      std::string("csc_tocsr_prefix_") + index_kernel_suffix(indices.dtype());
  auto *prefix_kernel = device.get_kernel(prefix_kernel_name, lib);
  encoder.set_compute_pipeline_state(prefix_kernel);
  encoder.set_input_array(counts, 0);
  encoder.set_output_array(next, 1);
  encoder.set_output_array(out_indptr, 2);
  encoder.set_bytes(n_rows_, 3);
  encoder.set_bytes(static_cast<int>(data.size()), 4);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

  auto fill_kernel_name =
      sparse_kernel_name("csc_tocsr_fill", data.dtype(), indices.dtype());
  auto *fill_kernel = device.get_kernel(fill_kernel_name, lib);
  encoder.set_compute_pipeline_state(fill_kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(next, 3);
  encoder.set_output_array(out_data, 4);
  encoder.set_output_array(out_indices, 5);
  encoder.set_bytes(n_cols_, 6);
  auto fill_threads = static_cast<size_t>(std::max(n_cols_, 1));
  auto fill_group =
      std::min(fill_threads, fill_kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(fill_threads, 1, 1),
                           MTL::Size(fill_group, 1, 1));

  encoder.add_temporary(std::move(counts));
  encoder.add_temporary(std::move(next));
}

void CSCToCSRDataVJP::eval_gpu(const std::vector<mx::array> &inputs,
                               std::vector<mx::array> &outputs) {
  auto &cotangent = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &out_indices = inputs[3];
  auto &out_indptr = inputs[4];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto &encoder = mx::metal::get_command_encoder(s);

  auto kernel_name = sparse_kernel_name("csc_tocsr_data_vjp", cotangent.dtype(),
                                        indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(cotangent, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(out_indices, 3);
  encoder.set_input_array(out_indptr, 4);
  encoder.set_output_array(out, 5);
  encoder.set_bytes(n_rows_, 6);
  encoder.set_bytes(n_cols_, 7);

  auto threads = static_cast<size_t>(std::max(n_cols_, 1));
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCToCSR::eval_gpu(const std::vector<mx::array> &,
                        std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_tocsr has no GPU implementation in this build.");
}

void CSCToCSRDataVJP::eval_gpu(const std::vector<mx::array> &,
                               std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_tocsr_data_vjp has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array> csc_tocsr(const mx::array &data,
                                                      const mx::array &indices,
                                                      const mx::array &indptr,
                                                      int n_rows, int n_cols,
                                                      mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csc_tocsr shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "csc_tocsr data");
  require_rank(indices, 1, "csc_tocsr indices");
  require_rank(indptr, 1, "csc_tocsr indptr");
  require_supported_value_dtype(data, "csc_tocsr data");
  require_same_index_dtype(indices, indptr, "csc_tocsr indices",
                           "csc_tocsr indptr");
  require_size(indptr, n_cols + 1, "csc_tocsr indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csc_tocsr data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);

  auto primitive = std::make_shared<CSCToCSR>(stream, n_rows, n_cols);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{static_cast<int>(data.size())},
       mx::Shape{static_cast<int>(indices.size())}, mx::Shape{n_rows + 1}},
      {data.dtype(), indices.dtype(), indptr.dtype()}, primitive,
      {data_contig, indices_contig, indptr_contig});
  return {outputs[0], outputs[1], outputs[2]};
}

} // namespace mlx_sparse
