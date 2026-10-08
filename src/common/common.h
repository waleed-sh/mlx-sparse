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

#include <string>
#include <type_traits>

#include "mlx/array.h"
#include "mlx/dtype.h"
#include "mlx/stream.h"
#include "mlx/types/complex.h"
#include "mlx/types/half_types.h"
#include "mlx/utils.h"

namespace mlx_sparse {

namespace mx = mlx::core;

std::string current_binary_dir();
std::string dtype_name(mx::Dtype dtype);
std::string value_kernel_suffix(mx::Dtype dtype);
std::string index_kernel_suffix(mx::Dtype dtype);
std::string sparse_kernel_name(const std::string &op, mx::Dtype value_dtype,
                               mx::Dtype index_dtype);

void require_rank(const mx::array &array, int ndim, const char *name);
void require_float32(const mx::array &array, const char *name);
void require_supported_value_dtype(const mx::array &array, const char *name);
void require_index_dtype(const mx::array &array, const char *name);
void require_same_value_dtype(const mx::array &lhs, const mx::array &rhs,
                              const char *lhs_name, const char *rhs_name);
void require_same_index_dtype(const mx::array &lhs, const mx::array &rhs,
                              const char *lhs_name, const char *rhs_name);
void require_size(const mx::array &array, int expected, const char *name);

bool is_supported_value_dtype(const mx::array &array);
bool is_int32(const mx::array &array);
bool is_int64(const mx::array &array);

template <typename T> struct Accumulator {
  using Type = T;
  static Type zero() { return Type{}; }
  static T cast(Type value) { return value; }
};

template <> struct Accumulator<mx::float16_t> {
  using Type = float;
  static Type zero() { return 0.0f; }
  static mx::float16_t cast(Type value) { return mx::float16_t(value); }
};

template <> struct Accumulator<mx::bfloat16_t> {
  using Type = float;
  static Type zero() { return 0.0f; }
  static mx::bfloat16_t cast(Type value) { return mx::bfloat16_t(value); }
};

// Whether a COO entry addresses a position inside the declared shape.
//
// The CPU counting sorts and scatters index their output by a coordinate, so a
// coordinate outside the shape writes past the end of the allocation. The
// comparisons stay in the index type: casting to size_t first turns a negative
// coordinate into a huge positive one, and casting to int folds a 64-bit
// coordinate above INT_MAX onto a small value that passes a naive check.
template <typename I> inline bool sparse_index_in_range(I i, int n) {
  return i >= I{0} && i < static_cast<I>(n);
}

template <typename I>
inline bool coo_entry_in_range(I r, I c, int n_rows, int n_cols) {
  return sparse_index_in_range(r, n_rows) && sparse_index_in_range(c, n_cols);
}

template <typename T>
typename Accumulator<T>::Type multiply_accumulate(T lhs, T rhs) {
  using AccT = typename Accumulator<T>::Type;
  if constexpr (std::is_same_v<T, mx::float16_t> ||
                std::is_same_v<T, mx::bfloat16_t>) {
    return static_cast<float>(lhs) * static_cast<float>(rhs);
  } else {
    return static_cast<AccT>(lhs) * static_cast<AccT>(rhs);
  }
}

} // namespace mlx_sparse
