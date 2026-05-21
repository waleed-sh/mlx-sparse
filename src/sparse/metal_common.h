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
