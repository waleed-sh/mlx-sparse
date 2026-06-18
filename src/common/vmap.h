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

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/common.h"
#include "mlx/ops.h"

namespace mlx_sparse {

inline void require_vmap_arity(const std::vector<mx::array> &inputs,
                               const std::vector<int> &axes, int expected,
                               const char *op_name) {
  if (static_cast<int>(inputs.size()) != expected ||
      static_cast<int>(axes.size()) != expected) {
    throw std::invalid_argument(std::string(op_name) +
                                " vmap received an invalid argument count.");
  }
}

inline void require_fixed_sparse_vmap_axes(const std::vector<int> &axes,
                                           int rhs_axis_index,
                                           const char *op_name) {
  for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
    if (i == rhs_axis_index) {
      continue;
    }
    if (axes[static_cast<size_t>(i)] != -1) {
      throw std::invalid_argument(
          std::string(op_name) +
          " vmap supports batching only the dense RHS. Sparse data, indices, "
          "and indptr/coordinate axes must be unmapped; batched sparse values "
          "with fixed structure are a documented v0.0.6b0 limitation.");
    }
  }
}

inline mx::array dense_rhs_with_vmap_axis_front(const mx::array &rhs, int axis,
                                                const mx::Stream &stream,
                                                const char *op_name) {
  if (axis < 0) {
    throw std::invalid_argument(
        std::string(op_name) +
        " vmap requires the dense RHS to have a mapped axis.");
  }
  if (axis >= rhs.ndim()) {
    throw std::invalid_argument(std::string(op_name) +
                                " vmap dense RHS axis is out of range.");
  }
  return axis == 0 ? rhs : mx::moveaxis(rhs, axis, 0, stream);
}

inline void require_vmap_rhs_rank(const mx::array &rhs, int expected_rank,
                                  const char *op_name) {
  if (rhs.ndim() != expected_rank) {
    throw std::invalid_argument(std::string(op_name) +
                                " vmap expected dense RHS rank " +
                                std::to_string(expected_rank) +
                                " after moving the mapped axis to the front.");
  }
}

inline void require_vmap_rhs_sparse_dim(const mx::array &rhs, int axis,
                                        int expected, const char *op_name) {
  if (rhs.shape(axis) != expected) {
    throw std::invalid_argument(
        std::string(op_name) +
        " vmap dense RHS sparse dimension does not match the sparse matrix.");
  }
}

inline void require_vmap_rhs_dim(const mx::array &rhs, int axis, int expected,
                                 const char *dimension_name,
                                 const char *op_name) {
  if (rhs.shape(axis) != expected) {
    throw std::invalid_argument(std::string(op_name) + " vmap " +
                                dimension_name + " changed under batching.");
  }
}

inline int checked_vmap_batch_product(int lhs, int rhs, const char *op_name) {
  if (lhs < 0 || rhs < 0) {
    throw std::invalid_argument(std::string(op_name) +
                                " vmap batch dimensions must be non-negative.");
  }
  if (rhs != 0 && lhs > std::numeric_limits<int>::max() / rhs) {
    throw std::overflow_error(std::string(op_name) +
                              " vmap flattened batch dimension exceeds int.");
  }
  return lhs * rhs;
}

} // namespace mlx_sparse
