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

#include "sparse/csr_sum_duplicates/csr_sum_duplicates.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "common/common.h"
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

class CSRDuplicateCounts : public mx::Primitive {
public:
  explicit CSRDuplicateCounts(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRDuplicateCounts"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

class CSRDuplicateFill : public mx::Primitive {
public:
  explicit CSRDuplicateFill(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRDuplicateFill"; }

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
    const int n_rows = static_cast<int>(indptr.size()) - 1;

    for (int row = 0; row < n_rows; ++row) {
      I count = I{0};
      I previous = I{0};
      bool have_previous = false;
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const I col = indices_ptr[p];
        if (!have_previous || col != previous) {
          count += I{1};
          previous = col;
          have_previous = true;
        }
      }
      counts_ptr[row] = count;
    }
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
        const int n_rows = static_cast<int>(indptr.size()) - 1;

        for (int row = 0; row < n_rows; ++row) {
          I write = out_indptr_ptr[row];
          for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1];) {
            const I col = indices_ptr[p];
            AccT acc = Accumulator<T>::zero();
            do {
              acc += accumulator_value<T>(data_ptr[p]);
              ++p;
            } while (p < indptr_ptr[row + 1] && indices_ptr[p] == col);

            out_indices_ptr[write] = col;
            out_data_ptr[write] = Accumulator<T>::cast(acc);
            ++write;
          }
        }
      });
}

template <typename I>
std::pair<mx::array, int> build_indptr_from_counts(const mx::array &counts,
                                                   int n_rows,
                                                   mx::Dtype index_dtype) {
  const auto *counts_ptr = counts.data<I>();
  std::vector<I> out_indptr(static_cast<size_t>(n_rows) + 1, I{0});
  int64_t total = 0;
  for (int row = 0; row < n_rows; ++row) {
    const auto count = static_cast<int64_t>(counts_ptr[row]);
    if (count < 0) {
      throw std::runtime_error(
          "csr_sum_duplicates produced a negative per-row count.");
    }
    total += count;
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error(
          "csr_sum_duplicates output nnz exceeds MLX shape limits.");
    }
    if (total > static_cast<int64_t>(std::numeric_limits<I>::max())) {
      throw std::overflow_error(
          "csr_sum_duplicates output nnz exceeds index dtype capacity.");
    }
    out_indptr[static_cast<size_t>(row) + 1] = static_cast<I>(total);
  }

  return {mx::array(out_indptr.begin(),
                    mx::Shape{static_cast<int>(out_indptr.size())},
                    index_dtype),
          static_cast<int>(total)};
}

std::pair<mx::array, int> build_indptr_from_counts(const mx::array &counts,
                                                   int n_rows) {
  if (counts.dtype() == mx::int32) {
    return build_indptr_from_counts<int32_t>(counts, n_rows, mx::int32);
  }
  return build_indptr_from_counts<int64_t>(counts, n_rows, mx::int64);
}

mx::array duplicate_counts(const mx::array &indices, const mx::array &indptr,
                           int n_rows, mx::Stream stream) {
  auto primitive = std::make_shared<CSRDuplicateCounts>(stream);
  return mx::array(mx::Shape{n_rows}, indices.dtype(), primitive,
                   {indices, indptr});
}

std::tuple<mx::array, mx::array>
duplicate_fill(const mx::array &data, const mx::array &indices,
               const mx::array &indptr, const mx::array &out_indptr,
               int out_nnz, mx::Stream stream) {
  auto primitive = std::make_shared<CSRDuplicateFill>(stream);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}}, {data.dtype(), indices.dtype()},
      primitive, {data, indices, indptr, out_indptr});
  return {outputs[0], outputs[1]};
}

} // namespace

void CSRDuplicateCounts::eval_cpu(const std::vector<mx::array> &inputs,
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
      "csr_sum_duplicates requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRDuplicateCounts::eval_gpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  const auto &indices = inputs[0];
  const auto &indptr = inputs[1];
  auto &counts = outputs[0];

  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name = std::string("csr_sum_duplicates_counts_") +
                     index_kernel_suffix(indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(indices, 0);
  encoder.set_input_array(indptr, 1);
  encoder.set_output_array(counts, 2);
  const int n_rows = static_cast<int>(indptr.size()) - 1;
  encoder.set_bytes(n_rows, 3);

  auto threads = std::max<size_t>(static_cast<size_t>(n_rows), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRDuplicateCounts::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_sum_duplicates has no GPU implementation in this build.");
}
#endif

void CSRDuplicateFill::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indices = inputs[1];
  const auto &indptr = inputs[2];
  const auto &out_indptr = inputs[3];

#define DISPATCH_CSR_SUM_DUP_VALUE(DTYPE, TYPE)                                \
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

  DISPATCH_CSR_SUM_DUP_VALUE(mx::float32, float)
  DISPATCH_CSR_SUM_DUP_VALUE(mx::float16, mx::float16_t)
  DISPATCH_CSR_SUM_DUP_VALUE(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_CSR_SUM_DUP_VALUE(mx::complex64, mx::complex64_t)
#undef DISPATCH_CSR_SUM_DUP_VALUE

  throw std::runtime_error("csr_sum_duplicates unsupported value dtype.");
}

#ifdef _METAL_
void CSRDuplicateFill::eval_gpu(const std::vector<mx::array> &inputs,
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
  auto kernel_name = sparse_kernel_name("csr_sum_duplicates_fill", data.dtype(),
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
  const int n_rows = static_cast<int>(indptr.size()) - 1;
  encoder.set_bytes(n_rows, 6);

  auto threads = std::max<size_t>(static_cast<size_t>(n_rows), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRDuplicateFill::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_sum_duplicates has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
csr_sum_duplicates(const mx::array &data, const mx::array &indices,
                   const mx::array &indptr, mx::StreamOrDevice s) {
  require_rank(data, 1, "csr_sum_duplicates data");
  require_rank(indices, 1, "csr_sum_duplicates indices");
  require_rank(indptr, 1, "csr_sum_duplicates indptr");
  require_supported_value_dtype(data, "csr_sum_duplicates data");
  require_same_index_dtype(indices, indptr, "csr_sum_duplicates indices",
                           "csr_sum_duplicates indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_sum_duplicates data and indices must have equal length.");
  }
  if (indptr.size() == 0) {
    throw std::invalid_argument("csr_sum_duplicates indptr cannot be empty.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  const int n_rows = static_cast<int>(indptr.size()) - 1;

  auto counts = duplicate_counts(indices_contig, indptr_contig, n_rows, stream);
  mx::eval(counts);
  auto [out_indptr, out_nnz] = build_indptr_from_counts(counts, n_rows);
  auto [out_data, out_indices] = duplicate_fill(
      data_contig, indices_contig, indptr_contig, out_indptr, out_nnz, stream);
  return {out_data, out_indices, out_indptr};
}

} // namespace mlx_sparse
