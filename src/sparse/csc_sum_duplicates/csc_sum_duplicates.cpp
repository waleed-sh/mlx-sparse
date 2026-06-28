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

#include "sparse/csc_sum_duplicates/csc_sum_duplicates.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "common/common.h"
#include "common/cpu_parallel.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

template <typename T> typename Accumulator<T>::Type accumulator_value(T value) {
  using AccT = typename Accumulator<T>::Type;
  if constexpr (std::is_same_v<T, mx::float16_t> ||
                std::is_same_v<T, mx::bfloat16_t>) {
    return static_cast<float>(value);
  } else {
    return static_cast<AccT>(value);
  }
}

class CSCDuplicateCounts : public mx::Primitive {
public:
  explicit CSCDuplicateCounts(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array> &,
                             const std::vector<mx::array> &,
                             const std::vector<int> &) override;

  std::vector<mx::array> vjp(const std::vector<mx::array> &,
                             const std::vector<mx::array> &,
                             const std::vector<int> &,
                             const std::vector<mx::array> &) override;

  const char *name() const override { return "CSCDuplicateCounts"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

class CSCDuplicateFill : public mx::Primitive {
public:
  explicit CSCDuplicateFill(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array> &,
                             const std::vector<mx::array> &,
                             const std::vector<int> &) override;

  std::vector<mx::array> vjp(const std::vector<mx::array> &,
                             const std::vector<mx::array> &,
                             const std::vector<int> &,
                             const std::vector<mx::array> &) override;

  const char *name() const override { return "CSCDuplicateFill"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

template <typename I>
void duplicate_counts_cpu_impl(const mx::array &indices,
                               const mx::array &indptr, mx::array &counts,
                               mx::Stream stream) {
  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_output_array(counts);

  encoder.dispatch([indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    counts = mx::array::unsafe_weak_copy(counts)]() mutable {
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *counts_ptr = counts.data<I>();
    const int n_cols = static_cast<int>(indptr.size()) - 1;

    auto count_cols = [&](CpuRange range) {
      for (int col = range.begin; col < range.end; ++col) {
        I count = I{0};
        I previous = I{0};
        bool have_previous = false;
        for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1]; ++p) {
          const I row = indices_ptr[p];
          if (!have_previous || row != previous) {
            count += I{1};
            previous = row;
            have_previous = true;
          }
        }
        counts_ptr[col] = count;
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_cols <= 0) {
      count_cols({0, n_cols});
      return;
    }
    const auto ranges =
        cpu_ranges_for_compressed_segments(indptr_ptr, n_cols, workers);
    if (ranges.size() <= 1) {
      count_cols({0, n_cols});
      return;
    }
    parallel_for_cpu_ranges(ranges, count_cols);
  });
}

template <typename T, typename I>
void duplicate_fill_cpu_impl(const mx::array &data, const mx::array &indices,
                             const mx::array &indptr,
                             const mx::array &out_indptr, mx::array &out_data,
                             mx::array &out_indices, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_indices);

  encoder.dispatch(
      [data = mx::array::unsafe_weak_copy(data),
       indices = mx::array::unsafe_weak_copy(indices),
       indptr = mx::array::unsafe_weak_copy(indptr),
       out_indptr = mx::array::unsafe_weak_copy(out_indptr),
       out_data = mx::array::unsafe_weak_copy(out_data),
       out_indices = mx::array::unsafe_weak_copy(out_indices)]() mutable {
        using AccT = typename Accumulator<T>::Type;

        const auto *data_ptr = data.data<T>();
        const auto *indices_ptr = indices.data<I>();
        const auto *indptr_ptr = indptr.data<I>();
        const auto *out_indptr_ptr = out_indptr.data<I>();
        auto *out_data_ptr = out_data.data<T>();
        auto *out_indices_ptr = out_indices.data<I>();
        const int n_cols = static_cast<int>(indptr.size()) - 1;

        auto fill_cols = [&](CpuRange range) {
          for (int col = range.begin; col < range.end; ++col) {
            I write = out_indptr_ptr[col];
            for (I p = indptr_ptr[col]; p < indptr_ptr[col + 1];) {
              const I row = indices_ptr[p];
              AccT acc = Accumulator<T>::zero();
              do {
                acc += accumulator_value<T>(data_ptr[p]);
                ++p;
              } while (p < indptr_ptr[col + 1] && indices_ptr[p] == row);

              out_indices_ptr[write] = row;
              out_data_ptr[write] = Accumulator<T>::cast(acc);
              ++write;
            }
          }
        };

        const int workers = configured_cpu_worker_count();
        if (workers <= 1 || n_cols <= 0) {
          fill_cols({0, n_cols});
          return;
        }
        const auto ranges =
            cpu_ranges_for_compressed_segments(indptr_ptr, n_cols, workers);
        if (ranges.size() <= 1) {
          fill_cols({0, n_cols});
          return;
        }
        parallel_for_cpu_ranges(ranges, fill_cols);
      });
}

template <typename I>
std::pair<mx::array, int> build_indptr_from_counts(const mx::array &counts,
                                                   int n_cols,
                                                   mx::Dtype index_dtype) {
  const auto *counts_ptr = counts.data<I>();
  std::vector<I> out_indptr(static_cast<size_t>(n_cols) + 1, I{0});
  int64_t total = 0;
  for (int col = 0; col < n_cols; ++col) {
    const auto count = static_cast<int64_t>(counts_ptr[col]);
    if (count < 0) {
      throw std::runtime_error(
          "csc_sum_duplicates produced a negative per-column count.");
    }
    total += count;
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error(
          "csc_sum_duplicates output nnz exceeds MLX shape limits.");
    }
    if (total > static_cast<int64_t>(std::numeric_limits<I>::max())) {
      throw std::overflow_error(
          "csc_sum_duplicates output nnz exceeds index dtype capacity.");
    }
    out_indptr[static_cast<size_t>(col) + 1] = static_cast<I>(total);
  }

  return {mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    index_dtype),
          static_cast<int>(total)};
}

std::pair<mx::array, int> build_indptr_from_counts(const mx::array &counts,
                                                   int n_cols) {
  if (counts.dtype() == mx::int32) {
    return build_indptr_from_counts<int32_t>(counts, n_cols, mx::int32);
  }
  return build_indptr_from_counts<int64_t>(counts, n_cols, mx::int64);
}

mx::array duplicate_counts(const mx::array &indices, const mx::array &indptr,
                           int n_cols, mx::Stream stream) {
  auto primitive = std::make_shared<CSCDuplicateCounts>(stream);
  return mx::array(mx::Shape{n_cols}, indices.dtype(), primitive,
                   {indices, indptr});
}

std::tuple<mx::array, mx::array>
duplicate_fill(const mx::array &data, const mx::array &indices,
               const mx::array &indptr, const mx::array &out_indptr,
               int out_nnz, mx::Stream stream) {
  auto primitive = std::make_shared<CSCDuplicateFill>(stream);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}}, {data.dtype(), indices.dtype()},
      primitive, {data, indices, indptr, out_indptr});
  return {outputs[0], outputs[1]};
}

[[noreturn]] void
throw_csc_sum_duplicates_autodiff_error(const char *primitive,
                                        const char *transform) {
  throw std::runtime_error(
      std::string("csc_sum_duplicates ") + transform +
      " is not supported for " + primitive +
      " because canonicalization sums duplicates and changes sparse topology. "
      "Use fixed-topology constructors when differentiating sparse values.");
}

} // namespace

void CSCDuplicateCounts::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  const auto &indices = inputs[0];
  const auto &indptr = inputs[1];
  auto &counts = outputs[0];

  if (indices.dtype() == mx::int32) {
    duplicate_counts_cpu_impl<int32_t>(indices, indptr, counts, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    duplicate_counts_cpu_impl<int64_t>(indices, indptr, counts, stream());
    return;
  }
  throw std::runtime_error(
      "csc_sum_duplicates requires int32 or int64 indices.");
}

std::vector<mx::array> CSCDuplicateCounts::jvp(const std::vector<mx::array> &,
                                               const std::vector<mx::array> &,
                                               const std::vector<int> &) {
  throw_csc_sum_duplicates_autodiff_error("CSCDuplicateCounts", "JVP");
}

std::vector<mx::array> CSCDuplicateCounts::vjp(const std::vector<mx::array> &,
                                               const std::vector<mx::array> &,
                                               const std::vector<int> &,
                                               const std::vector<mx::array> &) {
  throw_csc_sum_duplicates_autodiff_error("CSCDuplicateCounts", "VJP");
}

#ifdef _METAL_
void CSCDuplicateCounts::eval_gpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  const auto &indices = inputs[0];
  const auto &indptr = inputs[1];
  auto &counts = outputs[0];

  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name = std::string("csc_sum_duplicates_counts_") +
                     index_kernel_suffix(indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(indices, 0);
  encoder.set_input_array(indptr, 1);
  encoder.set_output_array(counts, 2);
  const int n_cols = static_cast<int>(indptr.size()) - 1;
  encoder.set_bytes(n_cols, 3);

  auto threads = std::max<size_t>(static_cast<size_t>(n_cols), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCDuplicateCounts::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_sum_duplicates has no GPU implementation in this build.");
}
#endif

void CSCDuplicateFill::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indices = inputs[1];
  const auto &indptr = inputs[2];
  const auto &out_indptr = inputs[3];

#define DISPATCH_CSC_SUM_DUP_VALUE(DTYPE, TYPE)                                \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      duplicate_fill_cpu_impl<TYPE, int32_t>(data, indices, indptr,            \
                                             out_indptr, outputs[0],           \
                                             outputs[1], stream());            \
    } else {                                                                   \
      duplicate_fill_cpu_impl<TYPE, int64_t>(data, indices, indptr,            \
                                             out_indptr, outputs[0],           \
                                             outputs[1], stream());            \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_CSC_SUM_DUP_VALUE(mx::float32, float)
  DISPATCH_CSC_SUM_DUP_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSC_SUM_DUP_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSC_SUM_DUP_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSC_SUM_DUP_VALUE

  throw std::runtime_error("csc_sum_duplicates unsupported value dtype.");
}

std::vector<mx::array> CSCDuplicateFill::jvp(const std::vector<mx::array> &,
                                             const std::vector<mx::array> &,
                                             const std::vector<int> &) {
  throw_csc_sum_duplicates_autodiff_error("CSCDuplicateFill", "JVP");
}

std::vector<mx::array> CSCDuplicateFill::vjp(const std::vector<mx::array> &,
                                             const std::vector<mx::array> &,
                                             const std::vector<int> &,
                                             const std::vector<mx::array> &) {
  throw_csc_sum_duplicates_autodiff_error("CSCDuplicateFill", "VJP");
}

#ifdef _METAL_
void CSCDuplicateFill::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indices = inputs[1];
  const auto &indptr = inputs[2];
  const auto &out_indptr = inputs[3];
  auto &out_data = outputs[0];
  auto &out_indices = outputs[1];

  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name = sparse_kernel_name("csc_sum_duplicates_fill", data.dtype(),
                                        indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(out_indptr, 3);
  encoder.set_output_array(out_data, 4);
  encoder.set_output_array(out_indices, 5);
  const int n_cols = static_cast<int>(indptr.size()) - 1;
  encoder.set_bytes(n_cols, 6);

  auto threads = std::max<size_t>(static_cast<size_t>(n_cols), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSCDuplicateFill::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csc_sum_duplicates has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
csc_sum_duplicates(const mx::array &data, const mx::array &indices,
                   const mx::array &indptr, mx::StreamOrDevice s) {
  require_rank(data, 1, "csc_sum_duplicates data");
  require_rank(indices, 1, "csc_sum_duplicates indices");
  require_rank(indptr, 1, "csc_sum_duplicates indptr");
  require_supported_value_dtype(data, "csc_sum_duplicates data");
  require_same_index_dtype(indices, indptr, "csc_sum_duplicates indices",
                           "csc_sum_duplicates indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csc_sum_duplicates data and indices must have equal length.");
  }
  if (indptr.size() == 0) {
    throw std::invalid_argument("csc_sum_duplicates indptr cannot be empty.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  const int n_cols = static_cast<int>(indptr.size()) - 1;

  auto counts = duplicate_counts(indices_contig, indptr_contig, n_cols, stream);
  mx::eval(counts);
  auto [out_indptr, out_nnz] = build_indptr_from_counts(counts, n_cols);
  auto [out_data, out_indices] = duplicate_fill(
      data_contig, indices_contig, indptr_contig, out_indptr, out_nnz, stream);
  return {out_data, out_indices, out_indptr};
}

} // namespace mlx_sparse
