// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

template <typename T, typename I>
[[kernel]] void csc_diagonal_kernel(device const T *data [[buffer(0)]],
                                    device const I *indices [[buffer(1)]],
                                    device const I *indptr [[buffer(2)]],
                                    device T *out [[buffer(3)]],
                                    constant int &diag_size [[buffer(4)]],
                                    uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= diag_size) {
    return;
  }

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    if (indices[p] == static_cast<I>(col)) {
      acc += typename sparse_accumulator<T>::type(data[p]);
    }
  }
  out[col] = sparse_accumulator<T>::cast(acc);
}

#define INSTANTIATE_CSC_DIAGONAL(NAME, T, I)                                   \
  template [[host_name("csc_diagonal_" #NAME)]] [[kernel]] void                \
  csc_diagonal_kernel<T, I>(device const T *, device const I *,                \
                            device const I *, device T *, constant int &,      \
                            uint)

INSTANTIATE_CSC_DIAGONAL(float32_int32, float, int);
INSTANTIATE_CSC_DIAGONAL(float32_int64, float, long);
INSTANTIATE_CSC_DIAGONAL(float16_int32, half, int);
INSTANTIATE_CSC_DIAGONAL(float16_int64, half, long);
INSTANTIATE_CSC_DIAGONAL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_DIAGONAL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_DIAGONAL(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_DIAGONAL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_DIAGONAL

template <typename T, typename I>
[[kernel]] void csc_diagonal_vector_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device T *out [[buffer(3)]],
    constant int &diag_size [[buffer(4)]],
    uint col [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[128];

  if (static_cast<int>(col) >= diag_size) {
    return;
  }

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  const I start = indptr[col];
  const I end = indptr[col + 1];
  for (I p = start + static_cast<I>(lane); p < end; p += 128) {
    if (indices[p] == static_cast<I>(col)) {
      acc += typename sparse_accumulator<T>::type(data[p]);
    }
  }

  partial[lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    out[col] = sparse_accumulator<T>::cast(partial[0]);
  }
}

#define INSTANTIATE_CSC_DIAGONAL_VECTOR(NAME, T, I)                            \
  template [[host_name("csc_diagonal_vector_" #NAME)]] [[kernel]] void         \
  csc_diagonal_vector_kernel<T, I>(device const T *, device const I *,         \
                                   device const I *, device T *,               \
                                   constant int &, uint, uint)

INSTANTIATE_CSC_DIAGONAL_VECTOR(float32_int32, float, int);
INSTANTIATE_CSC_DIAGONAL_VECTOR(float32_int64, float, long);
INSTANTIATE_CSC_DIAGONAL_VECTOR(float16_int32, half, int);
INSTANTIATE_CSC_DIAGONAL_VECTOR(float16_int64, half, long);
INSTANTIATE_CSC_DIAGONAL_VECTOR(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_DIAGONAL_VECTOR(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_DIAGONAL_VECTOR(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_DIAGONAL_VECTOR(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_DIAGONAL_VECTOR
