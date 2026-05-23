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

#include "common/metal_common.h"

template <typename T> inline bool fromdense_keep_value(T value, float threshold) {
  if (threshold == 0.0f) {
    return !(value == T(0));
  }
  return fabs(float(value)) > threshold;
}

template <>
inline bool fromdense_keep_value<complex64_t>(complex64_t value,
                                              float threshold) {
  if (threshold == 0.0f) {
    return value.real != 0.0f || value.imag != 0.0f;
  }
  return sqrt(value.real * value.real + value.imag * value.imag) > threshold;
}

template <typename T, typename I>
[[kernel]] void fromdense_counts_kernel(
    device const T *dense [[buffer(0)]], device I *counts [[buffer(1)]],
    constant int &n_rows [[buffer(2)]], constant int &n_cols [[buffer(3)]],
    constant float &threshold [[buffer(4)]], uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(n_rows)) {
    return;
  }

  I count = I(0);
  const int base = static_cast<int>(row) * n_cols;
  for (int col = 0; col < n_cols; ++col) {
    if (fromdense_keep_value<T>(dense[base + col], threshold)) {
      count += I(1);
    }
  }
  counts[row] = count;
}

template <typename T, typename I>
[[kernel]] void fromdense_fill_kernel(
    device const T *dense [[buffer(0)]],
    device const I *out_indptr [[buffer(1)]],
    device T *out_data [[buffer(2)]], device I *out_indices [[buffer(3)]],
    constant int &n_rows [[buffer(4)]], constant int &n_cols [[buffer(5)]],
    constant float &threshold [[buffer(6)]], uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(n_rows)) {
    return;
  }

  I write = out_indptr[row];
  const int base = static_cast<int>(row) * n_cols;
  for (int col = 0; col < n_cols; ++col) {
    const T value = dense[base + col];
    if (fromdense_keep_value<T>(value, threshold)) {
      out_data[write] = value;
      out_indices[write] = I(col);
      ++write;
    }
  }
}

#define INSTANTIATE_FROMDENSE_COUNTS(NAME, T, I)                               \
  template [[host_name("fromdense_counts_" #NAME)]] [[kernel]] void            \
  fromdense_counts_kernel<T, I>(device const T *, device I *, constant int &,  \
                                constant int &, constant float &, uint)

INSTANTIATE_FROMDENSE_COUNTS(float32_int32, float, int);
INSTANTIATE_FROMDENSE_COUNTS(float32_int64, float, long);
INSTANTIATE_FROMDENSE_COUNTS(float16_int32, half, int);
INSTANTIATE_FROMDENSE_COUNTS(float16_int64, half, long);
INSTANTIATE_FROMDENSE_COUNTS(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_FROMDENSE_COUNTS(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_FROMDENSE_COUNTS(complex64_int32, complex64_t, int);
INSTANTIATE_FROMDENSE_COUNTS(complex64_int64, complex64_t, long);

#undef INSTANTIATE_FROMDENSE_COUNTS

#define INSTANTIATE_FROMDENSE_FILL(NAME, T, I)                                 \
  template [[host_name("fromdense_fill_" #NAME)]] [[kernel]] void              \
  fromdense_fill_kernel<T, I>(device const T *, device const I *, device T *,  \
                              device I *, constant int &, constant int &,      \
                              constant float &, uint)

INSTANTIATE_FROMDENSE_FILL(float32_int32, float, int);
INSTANTIATE_FROMDENSE_FILL(float32_int64, float, long);
INSTANTIATE_FROMDENSE_FILL(float16_int32, half, int);
INSTANTIATE_FROMDENSE_FILL(float16_int64, half, long);
INSTANTIATE_FROMDENSE_FILL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_FROMDENSE_FILL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_FROMDENSE_FILL(complex64_int32, complex64_t, int);
INSTANTIATE_FROMDENSE_FILL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_FROMDENSE_FILL
