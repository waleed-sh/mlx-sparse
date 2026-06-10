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

#include "sparse/triangular_extract/triangular_extract.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
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

bool keep_triangular(int64_t row, int64_t col, int k, bool upper) {
  const int64_t diagonal = col - row;
  return upper ? diagonal >= static_cast<int64_t>(k)
               : diagonal <= static_cast<int64_t>(k);
}

std::string tri_count_kernel_name(const std::string &prefix,
                                  mx::Dtype index_dtype) {
  return prefix + "_" + index_kernel_suffix(index_dtype);
}

std::string tri_fill_kernel_name(const std::string &prefix,
                                 mx::Dtype value_dtype, mx::Dtype index_dtype) {
  return prefix + "_" + value_kernel_suffix(value_dtype) + "_" +
         index_kernel_suffix(index_dtype);
}

template <typename I>
std::pair<mx::array, int>
build_offsets_from_counts(const mx::array &counts, int n_segments,
                          mx::Dtype index_dtype, const char *op_name) {
  const auto *counts_ptr = counts.data<I>();
  std::vector<I> offsets(static_cast<size_t>(n_segments) + 1, I{0});
  int64_t total = 0;
  for (int i = 0; i < n_segments; ++i) {
    const auto count = static_cast<int64_t>(counts_ptr[i]);
    if (count < 0) {
      throw std::runtime_error(std::string(op_name) +
                               " produced a negative segment count.");
    }
    total += count;
    if (total > std::numeric_limits<int>::max()) {
      throw std::overflow_error(std::string(op_name) +
                                " output nnz exceeds MLX shape limits.");
    }
    if (total > static_cast<int64_t>(std::numeric_limits<I>::max())) {
      throw std::overflow_error(std::string(op_name) +
                                " output nnz exceeds index dtype capacity.");
    }
    offsets[static_cast<size_t>(i) + 1] = static_cast<I>(total);
  }

  return {mx::array(offsets.begin(),
                    mx::Shape{static_cast<int>(offsets.size())}, index_dtype),
          static_cast<int>(total)};
}

std::pair<mx::array, int> build_offsets_from_counts(const mx::array &counts,
                                                    int n_segments,
                                                    const char *op_name) {
  if (counts.dtype() == mx::int32) {
    return build_offsets_from_counts<int32_t>(counts, n_segments, mx::int32,
                                              op_name);
  }
  return build_offsets_from_counts<int64_t>(counts, n_segments, mx::int64,
                                            op_name);
}

class COOTriangularCounts : public mx::Primitive {
public:
  COOTriangularCounts(mx::Stream stream, int k, bool upper)
      : Primitive(stream), k_(k), upper_(upper) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOTriangularCounts"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOTriangularCounts &>(other);
    return k_ == rhs.k_ && upper_ == rhs.upper_;
  }

private:
  int k_;
  bool upper_;
};

class COOTriangularFill : public mx::Primitive {
public:
  explicit COOTriangularFill(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOTriangularFill"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

class CompressedTriangularCounts : public mx::Primitive {
public:
  CompressedTriangularCounts(mx::Stream stream, int n_segments, int k,
                             bool upper, bool csc)
      : Primitive(stream), n_segments_(n_segments), k_(k), upper_(upper),
        csc_(csc) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CompressedTriangularCounts"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CompressedTriangularCounts &>(other);
    return n_segments_ == rhs.n_segments_ && k_ == rhs.k_ &&
           upper_ == rhs.upper_ && csc_ == rhs.csc_;
  }

private:
  int n_segments_;
  int k_;
  bool upper_;
  bool csc_;
};

class CompressedTriangularFill : public mx::Primitive {
public:
  CompressedTriangularFill(mx::Stream stream, int n_segments, int k, bool upper,
                           bool csc)
      : Primitive(stream), n_segments_(n_segments), k_(k), upper_(upper),
        csc_(csc) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CompressedTriangularFill"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CompressedTriangularFill &>(other);
    return n_segments_ == rhs.n_segments_ && k_ == rhs.k_ &&
           upper_ == rhs.upper_ && csc_ == rhs.csc_;
  }

private:
  int n_segments_;
  int k_;
  bool upper_;
  bool csc_;
};

template <typename I>
void coo_counts_cpu_impl(const mx::array &row, const mx::array &col,
                         mx::array &counts, int k, bool upper,
                         mx::Stream stream) {
  counts.set_data(mx::allocator::malloc(counts.nbytes()));
  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(row);
  encoder.set_input_array(col);
  encoder.set_output_array(counts);
  encoder.dispatch([row = mx::array::unsafe_weak_copy(row),
                    col = mx::array::unsafe_weak_copy(col),
                    counts = mx::array::unsafe_weak_copy(counts), k,
                    upper]() mutable {
    const auto *row_ptr = row.data<I>();
    const auto *col_ptr = col.data<I>();
    auto *counts_ptr = counts.data<I>();
    const int nnz = static_cast<int>(row.size());

    auto count_range = [&](CpuRange range) {
      for (int p = range.begin; p < range.end; ++p) {
        counts_ptr[p] =
            keep_triangular(static_cast<int64_t>(row_ptr[p]),
                            static_cast<int64_t>(col_ptr[p]), k, upper)
                ? I{1}
                : I{0};
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || nnz <= 0) {
      count_range({0, nnz});
      return;
    }
    parallel_for_cpu_ranges(equal_cpu_ranges(nnz, workers), count_range);
  });
}

template <typename T, typename I>
void coo_fill_cpu_impl(const mx::array &data, const mx::array &row,
                       const mx::array &col, const mx::array &offsets,
                       mx::array &out_data, mx::array &out_row,
                       mx::array &out_col, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_row.set_data(mx::allocator::malloc(out_row.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(row);
  encoder.set_input_array(col);
  encoder.set_input_array(offsets);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_row);
  encoder.set_output_array(out_col);
  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    row = mx::array::unsafe_weak_copy(row),
                    col = mx::array::unsafe_weak_copy(col),
                    offsets = mx::array::unsafe_weak_copy(offsets),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_row = mx::array::unsafe_weak_copy(out_row),
                    out_col = mx::array::unsafe_weak_copy(out_col)]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *row_ptr = row.data<I>();
    const auto *col_ptr = col.data<I>();
    const auto *offsets_ptr = offsets.data<I>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_row_ptr = out_row.data<I>();
    auto *out_col_ptr = out_col.data<I>();
    const int nnz = static_cast<int>(data.size());

    auto fill_range = [&](CpuRange range) {
      for (int p = range.begin; p < range.end; ++p) {
        const I write = offsets_ptr[p];
        if (offsets_ptr[p + 1] == write) {
          continue;
        }
        out_data_ptr[write] = data_ptr[p];
        out_row_ptr[write] = row_ptr[p];
        out_col_ptr[write] = col_ptr[p];
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || nnz <= 0) {
      fill_range({0, nnz});
      return;
    }
    parallel_for_cpu_ranges(equal_cpu_ranges(nnz, workers), fill_range);
  });
}

template <typename I>
void compressed_counts_cpu_impl(const mx::array &indices,
                                const mx::array &indptr, mx::array &counts,
                                int n_segments, int k, bool upper, bool csc,
                                mx::Stream stream) {
  counts.set_data(mx::allocator::malloc(counts.nbytes()));
  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_output_array(counts);
  encoder.dispatch([indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    counts = mx::array::unsafe_weak_copy(counts), n_segments, k,
                    upper, csc]() mutable {
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    auto *counts_ptr = counts.data<I>();

    auto count_segments = [&](CpuRange range) {
      for (int segment = range.begin; segment < range.end; ++segment) {
        I count = I{0};
        for (I p = indptr_ptr[segment]; p < indptr_ptr[segment + 1]; ++p) {
          const int64_t row = csc ? static_cast<int64_t>(indices_ptr[p])
                                  : static_cast<int64_t>(segment);
          const int64_t col = csc ? static_cast<int64_t>(segment)
                                  : static_cast<int64_t>(indices_ptr[p]);
          if (keep_triangular(row, col, k, upper)) {
            ++count;
          }
        }
        counts_ptr[segment] = count;
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_segments <= 0) {
      count_segments({0, n_segments});
      return;
    }
    parallel_for_cpu_ranges(
        cpu_ranges_for_compressed_segments(indptr_ptr, n_segments, workers),
        count_segments);
  });
}

template <typename T, typename I>
void compressed_fill_cpu_impl(const mx::array &data, const mx::array &indices,
                              const mx::array &indptr,
                              const mx::array &out_indptr, mx::array &out_data,
                              mx::array &out_indices, int n_segments, int k,
                              bool upper, bool csc, mx::Stream stream) {
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_indices.set_data(mx::allocator::malloc(out_indices.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(out_indptr);
  encoder.set_output_array(out_data);
  encoder.set_output_array(out_indices);
  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    out_indptr = mx::array::unsafe_weak_copy(out_indptr),
                    out_data = mx::array::unsafe_weak_copy(out_data),
                    out_indices = mx::array::unsafe_weak_copy(out_indices),
                    n_segments, k, upper, csc]() mutable {
    const auto *data_ptr = data.data<T>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *out_indptr_ptr = out_indptr.data<I>();
    auto *out_data_ptr = out_data.data<T>();
    auto *out_indices_ptr = out_indices.data<I>();

    auto fill_segments = [&](CpuRange range) {
      for (int segment = range.begin; segment < range.end; ++segment) {
        I write = out_indptr_ptr[segment];
        for (I p = indptr_ptr[segment]; p < indptr_ptr[segment + 1]; ++p) {
          const int64_t row = csc ? static_cast<int64_t>(indices_ptr[p])
                                  : static_cast<int64_t>(segment);
          const int64_t col = csc ? static_cast<int64_t>(segment)
                                  : static_cast<int64_t>(indices_ptr[p]);
          if (keep_triangular(row, col, k, upper)) {
            out_data_ptr[write] = data_ptr[p];
            out_indices_ptr[write] = indices_ptr[p];
            ++write;
          }
        }
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || n_segments <= 0) {
      fill_segments({0, n_segments});
      return;
    }
    parallel_for_cpu_ranges(
        cpu_ranges_for_compressed_segments(indptr_ptr, n_segments, workers),
        fill_segments);
  });
}

mx::array coo_counts(const mx::array &row, const mx::array &col, int k,
                     bool upper, mx::Stream stream) {
  auto primitive = std::make_shared<COOTriangularCounts>(stream, k, upper);
  return mx::array(mx::Shape{static_cast<int>(row.size())}, row.dtype(),
                   primitive, {row, col});
}

std::tuple<mx::array, mx::array, mx::array>
coo_fill(const mx::array &data, const mx::array &row, const mx::array &col,
         const mx::array &offsets, int out_nnz, mx::Stream stream) {
  auto primitive = std::make_shared<COOTriangularFill>(stream);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}, mx::Shape{out_nnz}},
      {data.dtype(), row.dtype(), row.dtype()}, primitive,
      {data, row, col, offsets});
  return {outputs[0], outputs[1], outputs[2]};
}

mx::array compressed_counts(const mx::array &indices, const mx::array &indptr,
                            int n_segments, int k, bool upper, bool csc,
                            mx::Stream stream) {
  auto primitive = std::make_shared<CompressedTriangularCounts>(
      stream, n_segments, k, upper, csc);
  return mx::array(mx::Shape{n_segments}, indices.dtype(), primitive,
                   {indices, indptr});
}

std::tuple<mx::array, mx::array>
compressed_fill(const mx::array &data, const mx::array &indices,
                const mx::array &indptr, const mx::array &out_indptr,
                int out_nnz, int n_segments, int k, bool upper, bool csc,
                mx::Stream stream) {
  auto primitive = std::make_shared<CompressedTriangularFill>(
      stream, n_segments, k, upper, csc);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}}, {data.dtype(), indices.dtype()},
      primitive, {data, indices, indptr, out_indptr});
  return {outputs[0], outputs[1]};
}

void validate_coo_inputs(const mx::array &data, const mx::array &row,
                         const mx::array &col, int n_rows, int n_cols) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "coo_triangular shape dimensions must be non-negative.");
  }
  require_rank(data, 1, "coo_triangular data");
  require_rank(row, 1, "coo_triangular row");
  require_rank(col, 1, "coo_triangular col");
  require_supported_value_dtype(data, "coo_triangular data");
  require_same_index_dtype(row, col, "coo_triangular row",
                           "coo_triangular col");
  if (data.size() != row.size() || data.size() != col.size()) {
    throw std::invalid_argument(
        "coo_triangular data, row, and col arrays must have equal lengths.");
  }
}

void validate_compressed_inputs(const mx::array &data, const mx::array &indices,
                                const mx::array &indptr, int n_rows, int n_cols,
                                bool csc) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "triangular compressed shape dimensions must be non-negative.");
  }
  require_rank(data, 1, csc ? "csc_triangular data" : "csr_triangular data");
  require_rank(indices, 1,
               csc ? "csc_triangular indices" : "csr_triangular indices");
  require_rank(indptr, 1,
               csc ? "csc_triangular indptr" : "csr_triangular indptr");
  require_supported_value_dtype(data, csc ? "csc_triangular data"
                                          : "csr_triangular data");
  require_same_index_dtype(
      indices, indptr,
      csc ? "csc_triangular indices" : "csr_triangular indices",
      csc ? "csc_triangular indptr" : "csr_triangular indptr");
  require_size(indptr, (csc ? n_cols : n_rows) + 1,
               csc ? "csc_triangular indptr" : "csr_triangular indptr");
  if (data.size() != indices.size()) {
    throw std::invalid_argument(
        "triangular compressed data and indices must have equal lengths.");
  }
}

} // namespace

void COOTriangularCounts::eval_cpu(const std::vector<mx::array> &inputs,
                                   std::vector<mx::array> &outputs) {
  const auto &row = inputs[0];
  const auto &col = inputs[1];
  if (row.dtype() == mx::int32) {
    coo_counts_cpu_impl<int32_t>(row, col, outputs[0], k_, upper_, stream());
    return;
  }
  if (row.dtype() == mx::int64) {
    coo_counts_cpu_impl<int64_t>(row, col, outputs[0], k_, upper_, stream());
    return;
  }
  throw std::runtime_error("coo_triangular requires int32 or int64 indices.");
}

#ifdef _METAL_
void COOTriangularCounts::eval_gpu(const std::vector<mx::array> &inputs,
                                   std::vector<mx::array> &outputs) {
  const auto &row = inputs[0];
  const auto &col = inputs[1];
  auto &counts = outputs[0];
  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      tri_count_kernel_name("coo_triangular_counts", row.dtype()), lib);
  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(row, 0);
  encoder.set_input_array(col, 1);
  encoder.set_output_array(counts, 2);
  const int nnz = static_cast<int>(row.size());
  const int upper_i = upper_ ? 1 : 0;
  encoder.set_bytes(nnz, 3);
  encoder.set_bytes(k_, 4);
  encoder.set_bytes(upper_i, 5);
  const auto threads = std::max<size_t>(static_cast<size_t>(nnz), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOTriangularCounts::eval_gpu(const std::vector<mx::array> &,
                                   std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_triangular counts has no GPU implementation in this build.");
}
#endif

void COOTriangularFill::eval_cpu(const std::vector<mx::array> &inputs,
                                 std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &row = inputs[1];
  const auto &col = inputs[2];
  const auto &offsets = inputs[3];

#define DISPATCH_COO_TRI_FILL(DTYPE, TYPE)                                     \
  if (data.dtype() == DTYPE) {                                                 \
    if (row.dtype() == mx::int32) {                                            \
      coo_fill_cpu_impl<TYPE, int32_t>(data, row, col, offsets, outputs[0],    \
                                       outputs[1], outputs[2], stream());      \
    } else {                                                                   \
      coo_fill_cpu_impl<TYPE, int64_t>(data, row, col, offsets, outputs[0],    \
                                       outputs[1], outputs[2], stream());      \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COO_TRI_FILL(mx::float32, float)
  DISPATCH_COO_TRI_FILL(mx::float16, mx::float16_t)
  DISPATCH_COO_TRI_FILL(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_TRI_FILL(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_TRI_FILL

  throw std::runtime_error("coo_triangular fill unsupported value dtype.");
}

#ifdef _METAL_
void COOTriangularFill::eval_gpu(const std::vector<mx::array> &inputs,
                                 std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &row = inputs[1];
  const auto &col = inputs[2];
  const auto &offsets = inputs[3];
  auto &out_data = outputs[0];
  auto &out_row = outputs[1];
  auto &out_col = outputs[2];
  out_data.set_data(mx::allocator::malloc(out_data.nbytes()));
  out_row.set_data(mx::allocator::malloc(out_row.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      tri_fill_kernel_name("coo_triangular_fill", data.dtype(), row.dtype()),
      lib);
  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(row, 1);
  encoder.set_input_array(col, 2);
  encoder.set_input_array(offsets, 3);
  encoder.set_output_array(out_data, 4);
  encoder.set_output_array(out_row, 5);
  encoder.set_output_array(out_col, 6);
  const int nnz = static_cast<int>(data.size());
  encoder.set_bytes(nnz, 7);
  const auto threads = std::max<size_t>(static_cast<size_t>(nnz), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOTriangularFill::eval_gpu(const std::vector<mx::array> &,
                                 std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_triangular fill has no GPU implementation in this build.");
}
#endif

void CompressedTriangularCounts::eval_cpu(const std::vector<mx::array> &inputs,
                                          std::vector<mx::array> &outputs) {
  const auto &indices = inputs[0];
  const auto &indptr = inputs[1];
  if (indices.dtype() == mx::int32) {
    compressed_counts_cpu_impl<int32_t>(
        indices, indptr, outputs[0], n_segments_, k_, upper_, csc_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    compressed_counts_cpu_impl<int64_t>(
        indices, indptr, outputs[0], n_segments_, k_, upper_, csc_, stream());
    return;
  }
  throw std::runtime_error(
      "triangular compressed counts require int32 or int64 indices.");
}

#ifdef _METAL_
void CompressedTriangularCounts::eval_gpu(const std::vector<mx::array> &inputs,
                                          std::vector<mx::array> &outputs) {
  const auto &indices = inputs[0];
  const auto &indptr = inputs[1];
  auto &counts = outputs[0];
  counts.set_data(mx::allocator::malloc(counts.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  const auto prefix = csc_ ? "csc_triangular_counts" : "csr_triangular_counts";
  auto *kernel =
      device.get_kernel(tri_count_kernel_name(prefix, indices.dtype()), lib);
  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(indices, 0);
  encoder.set_input_array(indptr, 1);
  encoder.set_output_array(counts, 2);
  const int upper_i = upper_ ? 1 : 0;
  encoder.set_bytes(n_segments_, 3);
  encoder.set_bytes(k_, 4);
  encoder.set_bytes(upper_i, 5);
  const auto threads = std::max<size_t>(static_cast<size_t>(n_segments_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CompressedTriangularCounts::eval_gpu(const std::vector<mx::array> &,
                                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "triangular compressed counts have no GPU implementation in this build.");
}
#endif

void CompressedTriangularFill::eval_cpu(const std::vector<mx::array> &inputs,
                                        std::vector<mx::array> &outputs) {
  const auto &data = inputs[0];
  const auto &indices = inputs[1];
  const auto &indptr = inputs[2];
  const auto &out_indptr = inputs[3];

#define DISPATCH_COMPRESSED_TRI_FILL(DTYPE, TYPE)                              \
  if (data.dtype() == DTYPE) {                                                 \
    if (indices.dtype() == mx::int32) {                                        \
      compressed_fill_cpu_impl<TYPE, int32_t>(                                 \
          data, indices, indptr, out_indptr, outputs[0], outputs[1],           \
          n_segments_, k_, upper_, csc_, stream());                            \
    } else {                                                                   \
      compressed_fill_cpu_impl<TYPE, int64_t>(                                 \
          data, indices, indptr, out_indptr, outputs[0], outputs[1],           \
          n_segments_, k_, upper_, csc_, stream());                            \
    }                                                                          \
    return;                                                                    \
  }

  DISPATCH_COMPRESSED_TRI_FILL(mx::float32, float)
  DISPATCH_COMPRESSED_TRI_FILL(mx::float16, mx::float16_t)
  DISPATCH_COMPRESSED_TRI_FILL(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COMPRESSED_TRI_FILL(mx::complex64, mx::complex64_t)
#undef DISPATCH_COMPRESSED_TRI_FILL

  throw std::runtime_error(
      "triangular compressed fill unsupported value dtype.");
}

#ifdef _METAL_
void CompressedTriangularFill::eval_gpu(const std::vector<mx::array> &inputs,
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
  const auto prefix = csc_ ? "csc_triangular_fill" : "csr_triangular_fill";
  auto *kernel = device.get_kernel(
      tri_fill_kernel_name(prefix, data.dtype(), indices.dtype()), lib);
  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(out_indptr, 3);
  encoder.set_output_array(out_data, 4);
  encoder.set_output_array(out_indices, 5);
  const int upper_i = upper_ ? 1 : 0;
  encoder.set_bytes(n_segments_, 6);
  encoder.set_bytes(k_, 7);
  encoder.set_bytes(upper_i, 8);
  const auto threads = std::max<size_t>(static_cast<size_t>(n_segments_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CompressedTriangularFill::eval_gpu(const std::vector<mx::array> &,
                                        std::vector<mx::array> &) {
  throw std::runtime_error(
      "triangular compressed fill has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
coo_triangular(const mx::array &data, const mx::array &row,
               const mx::array &col, int n_rows, int n_cols, int k, bool upper,
               mx::StreamOrDevice s) {
  validate_coo_inputs(data, row, col, n_rows, n_cols);
  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto row_contig = mx::contiguous(row, false, stream);
  auto col_contig = mx::contiguous(col, false, stream);
  auto counts = coo_counts(row_contig, col_contig, k, upper, stream);
  mx::eval(counts);
  auto [offsets, out_nnz] = build_offsets_from_counts(
      counts, static_cast<int>(data.size()), "coo_triangular");
  return coo_fill(data_contig, row_contig, col_contig, offsets, out_nnz,
                  stream);
}

std::tuple<mx::array, mx::array, mx::array>
csr_triangular(const mx::array &data, const mx::array &indices,
               const mx::array &indptr, int n_rows, int n_cols, int k,
               bool upper, mx::StreamOrDevice s) {
  validate_compressed_inputs(data, indices, indptr, n_rows, n_cols, false);
  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto counts = compressed_counts(indices_contig, indptr_contig, n_rows, k,
                                  upper, false, stream);
  mx::eval(counts);
  auto [out_indptr, out_nnz] =
      build_offsets_from_counts(counts, n_rows, "csr_triangular");
  auto [out_data, out_indices] =
      compressed_fill(data_contig, indices_contig, indptr_contig, out_indptr,
                      out_nnz, n_rows, k, upper, false, stream);
  return {out_data, out_indices, out_indptr};
}

std::tuple<mx::array, mx::array, mx::array>
csc_triangular(const mx::array &data, const mx::array &indices,
               const mx::array &indptr, int n_rows, int n_cols, int k,
               bool upper, mx::StreamOrDevice s) {
  validate_compressed_inputs(data, indices, indptr, n_rows, n_cols, true);
  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto counts = compressed_counts(indices_contig, indptr_contig, n_cols, k,
                                  upper, true, stream);
  mx::eval(counts);
  auto [out_indptr, out_nnz] =
      build_offsets_from_counts(counts, n_cols, "csc_triangular");
  auto [out_data, out_indices] =
      compressed_fill(data_contig, indices_contig, indptr_contig, out_indptr,
                      out_nnz, n_cols, k, upper, true, stream);
  return {out_data, out_indices, out_indptr};
}

} // namespace mlx_sparse
