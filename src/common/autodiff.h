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

#include <stdexcept>
#include <string>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace mlx_sparse {

namespace mx = mlx::core;

inline mx::array sparse_vector_cotangent_gather(const mx::array &cotangent,
                                                const mx::array &index,
                                                mx::StreamOrDevice s) {
  return mx::take(cotangent, index, s);
}

inline mx::array sparse_dense_cotangent_gather(const mx::array &cotangent,
                                               const mx::array &row,
                                               const mx::array &col, int n_cols,
                                               mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  auto flat_cotangent = mx::flatten(cotangent, stream);
  auto flat_index = row * n_cols + col;
  return mx::take(flat_cotangent, flat_index, stream);
}

inline mx::array sparse_diagonal_cotangent_gather(
    const mx::array &cotangent, const mx::array &row, const mx::array &col,
    const mx::array &data_like, int diag_size, mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  if (diag_size <= 0) {
    return mx::zeros_like(data_like, stream);
  }
  auto in_range = row < diag_size;
  auto on_diagonal = mx::equal(row, col, stream);
  auto mask = mx::logical_and(on_diagonal, in_range, stream);
  auto safe_row = mx::where(in_range, row, mx::zeros_like(row, stream), stream);
  auto gathered = mx::take(cotangent, safe_row, stream);
  return mx::where(mask, gathered, mx::zeros_like(data_like, stream), stream);
}

inline mx::array
sparse_trace_cotangent_gather(const mx::array &cotangent, const mx::array &row,
                              const mx::array &col, const mx::array &data_like,
                              int diag_size, mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  auto in_range = row < diag_size;
  auto on_diagonal = mx::equal(row, col, stream);
  auto mask = mx::logical_and(on_diagonal, in_range, stream);
  auto gathered = mx::full_like(data_like, cotangent, stream);
  return mx::where(mask, gathered, mx::zeros_like(data_like, stream), stream);
}

inline void require_sparse_value_autodiff_arg(int argnum, const char *op,
                                              const char *transform) {
  if (argnum != 0) {
    throw std::runtime_error(std::string(op) + " " + transform +
                             " is implemented only for sparse data values; "
                             "sparse index buffers define fixed topology and "
                             "are not differentiable.");
  }
}

} // namespace mlx_sparse
