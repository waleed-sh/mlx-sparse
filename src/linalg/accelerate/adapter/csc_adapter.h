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

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "mlx/array.h"

#ifndef MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
#define MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK 0
#endif

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
#include <Accelerate/Accelerate.h>
#endif

namespace mlx_sparse {

namespace mx = mlx::core;

struct AccelerateCscAdapterOptions {
  bool require_square = false;
  bool require_non_empty = true;
  bool canonicalize = true;
};

struct AccelerateCscMatrixFloat {
  int row_count = 0;
  int column_count = 0;
  std::vector<long> column_starts;
  std::vector<int> row_indices;
  std::vector<float> values;

  std::size_t nnz() const { return values.size(); }

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
  SparseMatrixStructure structure(SparseAttributes_t attributes) const;
  SparseMatrixStructure structure() const;
  SparseMatrix_Float matrix(SparseAttributes_t attributes) const;
  SparseMatrix_Float matrix() const;
#endif
};

AccelerateCscMatrixFloat make_accelerate_csc_matrix_float32(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    std::int64_t n_rows, std::int64_t n_cols,
    AccelerateCscAdapterOptions options = {},
    std::string_view operation = "accelerate_csc_adapter");

AccelerateCscMatrixFloat make_accelerate_csc_matrix_float32_from_csr(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    std::int64_t n_rows, std::int64_t n_cols,
    AccelerateCscAdapterOptions options = {},
    std::string_view operation = "accelerate_csr_adapter");

AccelerateCscMatrixFloat make_accelerate_csc_matrix_float32_from_coo(
    const mx::array &data, const mx::array &row, const mx::array &col,
    std::int64_t n_rows, std::int64_t n_cols,
    AccelerateCscAdapterOptions options = {},
    std::string_view operation = "accelerate_coo_adapter");

} // namespace mlx_sparse
