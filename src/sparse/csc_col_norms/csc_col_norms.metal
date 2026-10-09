// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

template <typename T> inline float sparse_norm_square(T value) {
  const float x = float(value);
  return x * x;
}

template <> inline float sparse_norm_square(complex64_t value) {
  return value.real * value.real + value.imag * value.imag;
}

template <typename T, typename I>
[[kernel]] void csc_col_norms_kernel(device const T *data [[buffer(0)]],
                                     device const I *indptr [[buffer(1)]],
                                     device float *out [[buffer(2)]],
                                     constant int &n_cols [[buffer(3)]],
                                     uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= n_cols) {
    return;
  }

  float acc = 0.0f;
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    acc += sparse_norm_square<T>(data[p]);
  }
  out[col] = sqrt(acc);
}

#define INSTANTIATE_CSC_COL_NORMS(NAME, T, I)                                  \
  template [[host_name("csc_col_norms_" #NAME)]] [[kernel]] void               \
  csc_col_norms_kernel<T, I>(device const T *, device const I *,               \
                             device float *, constant int &, uint)

INSTANTIATE_CSC_COL_NORMS(float32_int32, float, int);
INSTANTIATE_CSC_COL_NORMS(float32_int64, float, long);
INSTANTIATE_CSC_COL_NORMS(float16_int32, half, int);
INSTANTIATE_CSC_COL_NORMS(float16_int64, half, long);
INSTANTIATE_CSC_COL_NORMS(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_COL_NORMS(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_COL_NORMS(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_COL_NORMS(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_COL_NORMS

template <typename T, typename I>
[[kernel]] void csc_col_norms_vector_kernel(
    device const T *data [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device float *out [[buffer(2)]], constant int &n_cols [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 group_count [[threadgroups_per_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  const uint col = group_id.y * group_count.x + group_id.x;
  threadgroup float partial[32];

  if (col >= uint(n_cols)) {
    return;
  }

  float acc = 0.0f;
  const I start = indptr[col];
  const I end = indptr[col + 1];
  for (I p = start + static_cast<I>(lane); p < end; p += 128) {
    acc += sparse_norm_square<T>(data[p]);
  }

  const auto sum = sparse_cooperative_sum_128(acc, partial, simd_lane,
                                              simd_group, simd_width);

  if (lane == 0) {
    out[col] = sqrt(sum);
  }
}

#define INSTANTIATE_CSC_COL_NORMS_VECTOR(NAME, T, I)                           \
  template [[host_name("csc_col_norms_vector_" #NAME)]] [[kernel]] void        \
  csc_col_norms_vector_kernel<T, I>(device const T *, device const I *,        \
                                    device float *, constant int &, uint2,     \
                                    uint2, uint, uint, uint, uint)

INSTANTIATE_CSC_COL_NORMS_VECTOR(float32_int32, float, int);
INSTANTIATE_CSC_COL_NORMS_VECTOR(float32_int64, float, long);
INSTANTIATE_CSC_COL_NORMS_VECTOR(float16_int32, half, int);
INSTANTIATE_CSC_COL_NORMS_VECTOR(float16_int64, half, long);
INSTANTIATE_CSC_COL_NORMS_VECTOR(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_COL_NORMS_VECTOR(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_COL_NORMS_VECTOR(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_COL_NORMS_VECTOR(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_COL_NORMS_VECTOR
