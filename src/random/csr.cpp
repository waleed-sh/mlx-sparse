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

#include "random/random.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "common/cpu_parallel.h"
#include "mlx/ops.h"
#include "random/common.h"

namespace mlx_sparse {

namespace {

template <typename I>
std::tuple<mx::array, mx::array>
random_csr_indices_host(mx::array key, int64_t n_rows, int64_t n_cols,
                        int64_t nnz, mx::Dtype index_dtype) {
  key.eval();
  std::vector<uint64_t> linear(static_cast<size_t>(nnz));
  if (nnz > 0) {
    const auto *key_ptr = key.data<uint32_t>();
    const uint64_t seed = keyed_seed(key_ptr, n_rows, n_cols, nnz);
    const uint64_t total =
        static_cast<uint64_t>(n_rows) * static_cast<uint64_t>(n_cols);
    const int workers = configured_cpu_worker_count();
    auto fill_range = [&](CpuRange range) {
      for (int64_t i = range.begin; i < range.end; ++i) {
        linear[static_cast<size_t>(i)] =
            random_linear_index(static_cast<uint64_t>(i), total, seed);
      }
    };
    if (workers > 1 && nnz > 0) {
      parallel_for_cpu_ranges(equal_cpu_ranges(static_cast<int>(nnz), workers),
                              fill_range);
    } else {
      fill_range({0, static_cast<int>(nnz)});
    }
    std::sort(linear.begin(), linear.end());
  }

  std::vector<I> indices(static_cast<size_t>(nnz));
  std::vector<I> indptr(static_cast<size_t>(n_rows) + 1, I{0});
  size_t read = 0;
  for (int64_t row = 0; row < n_rows; ++row) {
    while (read < linear.size() &&
           linear[read] / static_cast<uint64_t>(n_cols) ==
               static_cast<uint64_t>(row)) {
      indices[read] =
          static_cast<I>(linear[read] % static_cast<uint64_t>(n_cols));
      ++read;
    }
    indptr[static_cast<size_t>(row) + 1] = static_cast<I>(read);
  }
  if (read != linear.size()) {
    throw std::runtime_error("random_csr_indices internal count mismatch.");
  }

  return {
      mx::array(indices.begin(), mx::Shape{static_cast<int>(nnz)}, index_dtype),
      mx::array(indptr.begin(), mx::Shape{static_cast<int>(n_rows + 1)},
                index_dtype)};
}

} // namespace

std::tuple<mx::array, mx::array>
random_csr_indices(const mx::array &key, int64_t n_rows, int64_t n_cols,
                   int64_t nnz, int index_dtype_bits, mx::StreamOrDevice s) {
  check_random_key(key);
  const auto index_dtype = random_index_dtype_from_bits(index_dtype_bits);
  check_random_shape(n_rows, n_cols, nnz, index_dtype);

  auto stream = mx::to_stream(s);
  auto key_contig = mx::contiguous(key, false, stream);
  if (nnz == 0) {
    return {mx::zeros(mx::Shape{0}, index_dtype, stream),
            mx::zeros(mx::Shape{static_cast<int>(n_rows + 1)}, index_dtype,
                      stream)};
  }
  if (stream.device == mx::Device::cpu) {
    if (index_dtype == mx::int32) {
      return random_csr_indices_host<int32_t>(std::move(key_contig), n_rows,
                                              n_cols, nnz, index_dtype);
    }
    return random_csr_indices_host<int64_t>(std::move(key_contig), n_rows,
                                            n_cols, nnz, index_dtype);
  }

  auto counts = random_compressed_counts(key_contig, n_rows, n_cols, nnz,
                                         /*csc=*/false, stream);
  auto indptr = random_compressed_indptr(counts, index_dtype, stream);
  auto keys = random_structural_keys(key_contig, n_rows, n_cols, nnz,
                                     /*csc=*/false, stream);
  auto sorted_keys = mx::sort(keys, stream);
  auto indices = random_compressed_unpack_sorted_keys(
      sorted_keys, n_cols, index_dtype, /*csc=*/false, stream);
  return {indices, indptr};
}

} // namespace mlx_sparse
