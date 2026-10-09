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

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/complex.h"

using namespace metal;

inline float sparse_simd_sum(float value) { return simd_sum(value); }

inline complex64_t sparse_simd_sum(complex64_t value) {
  return complex64_t(simd_sum(value.real), simd_sum(value.imag));
}

// Exchange one value per SIMD group with one threadgroup barrier.
template <typename T>
inline T sparse_cooperative_sum_128(T value, threadgroup T *partial,
                                    uint simd_lane, uint simd_group,
                                    uint simd_width) {
  const T group_sum = sparse_simd_sum(value);
  if (simd_lane == 0) {
    partial[simd_group] = group_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    T sum = T(0);
    for (uint group = simd_lane; group < 128 / simd_width;
         group += simd_width) {
      sum += partial[group];
    }
    return sparse_simd_sum(sum);
  }
  return T(0);
}

template <typename T> struct sparse_accumulator {
  typedef T type;
  static inline type zero() { return type(0); }
  static inline T cast(type value) { return value; }
};

template <> struct sparse_accumulator<half> {
  typedef float type;
  static inline type zero() { return 0.0f; }
  static inline half cast(type value) { return half(value); }
};

template <> struct sparse_accumulator<bfloat16_t> {
  typedef float type;
  static inline type zero() { return 0.0f; }
  static inline bfloat16_t cast(type value) { return bfloat16_t(value); }
};

template <typename T>
inline typename sparse_accumulator<T>::type sparse_multiply(T lhs, T rhs) {
  typedef typename sparse_accumulator<T>::type acc_t;
  return acc_t(lhs) * acc_t(rhs);
}

template <>
inline sparse_accumulator<complex64_t>::type sparse_multiply(complex64_t lhs,
                                                             complex64_t rhs) {
  return lhs * rhs;
}

template <typename T> inline T sparse_add_storage(T lhs, T rhs) {
  typedef typename sparse_accumulator<T>::type acc_t;
  return sparse_accumulator<T>::cast(acc_t(lhs) + acc_t(rhs));
}
