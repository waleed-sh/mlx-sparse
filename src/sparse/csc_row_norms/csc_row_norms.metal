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

[[kernel]] void csc_row_norms_zero_float32(device float *out [[buffer(0)]],
                                           constant int &n_rows [[buffer(1)]],
                                           uint row
                                           [[thread_position_in_grid]]) {
  if (static_cast<int>(row) < n_rows) {
    out[row] = 0.0f;
  }
}

template <typename T, typename I>
[[kernel]] void csc_row_norms_atomic_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device float *out [[buffer(3)]],
    constant int &n_rows [[buffer(4)]], constant int &n_cols [[buffer(5)]],
    uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= n_cols) {
    return;
  }

  device atomic_float *atomic_out =
      reinterpret_cast<device atomic_float *>(out);
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    const int row = static_cast<int>(indices[p]);
    if (row >= 0 && row < n_rows) {
      atomic_fetch_add_explicit(&atomic_out[row],
                                sparse_norm_square<T>(data[p]),
                                memory_order_relaxed);
    }
  }
}

#define INSTANTIATE_CSC_ROW_NORMS_ATOMIC(NAME, T, I)                           \
  template [[host_name("csc_row_norms_atomic_" #NAME)]] [[kernel]] void        \
  csc_row_norms_atomic_kernel<T, I>(device const T *, device const I *,        \
                                    device const I *, device float *,          \
                                    constant int &, constant int &, uint)

INSTANTIATE_CSC_ROW_NORMS_ATOMIC(float32_int32, float, int);
INSTANTIATE_CSC_ROW_NORMS_ATOMIC(float32_int64, float, long);
INSTANTIATE_CSC_ROW_NORMS_ATOMIC(float16_int32, half, int);
INSTANTIATE_CSC_ROW_NORMS_ATOMIC(float16_int64, half, long);
INSTANTIATE_CSC_ROW_NORMS_ATOMIC(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_ROW_NORMS_ATOMIC(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_ROW_NORMS_ATOMIC(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_ROW_NORMS_ATOMIC(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_ROW_NORMS_ATOMIC

[[kernel]] void csc_row_norms_sqrt_float32(device float *out [[buffer(0)]],
                                           constant int &n_rows [[buffer(1)]],
                                           uint row
                                           [[thread_position_in_grid]]) {
  if (static_cast<int>(row) < n_rows) {
    out[row] = sqrt(out[row]);
  }
}
