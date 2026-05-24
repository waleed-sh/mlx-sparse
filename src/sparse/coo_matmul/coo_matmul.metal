// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

[[kernel]] void coo_matmul_zero_float32(device float *out [[buffer(0)]],
                                        constant int &size [[buffer(1)]],
                                        uint tid [[thread_position_in_grid]]) {
  if (static_cast<int>(tid) < size) {
    out[tid] = 0.0f;
  }
}

template <typename I>
[[kernel]] void coo_matmul_atomic_kernel(
    device const float *data [[buffer(0)]], device const I *row [[buffer(1)]],
    device const I *col [[buffer(2)]], device const float *rhs [[buffer(3)]],
    device float *out [[buffer(4)]], constant int &rhs_cols [[buffer(5)]],
    constant int &total [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  if (static_cast<int>(tid) >= total) {
    return;
  }
  const int p = static_cast<int>(tid) / rhs_cols;
  const int k = static_cast<int>(tid) - p * rhs_cols;
  device atomic_float *atomic_out =
      reinterpret_cast<device atomic_float *>(out);
  atomic_fetch_add_explicit(
      &atomic_out[static_cast<int>(row[p]) * rhs_cols + k],
      data[p] * rhs[static_cast<int>(col[p]) * rhs_cols + k],
      memory_order_relaxed);
}

template <typename T, typename I>
[[kernel]] void coo_matmul_serial_kernel(
    device const T *data [[buffer(0)]], device const I *row [[buffer(1)]],
    device const I *col [[buffer(2)]], device const T *rhs [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_rows [[buffer(5)]],
    constant int &rhs_cols [[buffer(6)]], constant int &nnz [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }
  const int out_size = n_rows * rhs_cols;
  for (int i = 0; i < out_size; ++i) {
    out[i] = T(0);
  }
  for (int p = 0; p < nnz; ++p) {
    const int out_offset = static_cast<int>(row[p]) * rhs_cols;
    const int rhs_offset = static_cast<int>(col[p]) * rhs_cols;
    for (int k = 0; k < rhs_cols; ++k) {
      typedef typename sparse_accumulator<T>::type acc_t;
      const acc_t updated = acc_t(out[out_offset + k]) +
                            sparse_multiply<T>(data[p], rhs[rhs_offset + k]);
      out[out_offset + k] = sparse_accumulator<T>::cast(updated);
    }
  }
}

template [[host_name("coo_matmul_atomic_int32")]] [[kernel]] void
coo_matmul_atomic_kernel<int>(device const float *, device const int *,
                              device const int *, device const float *,
                              device float *, constant int &, constant int &,
                              uint);
template [[host_name("coo_matmul_atomic_int64")]] [[kernel]] void
coo_matmul_atomic_kernel<long>(device const float *, device const long *,
                               device const long *, device const float *,
                               device float *, constant int &, constant int &,
                               uint);

#define INSTANTIATE_COO_MATMUL_SERIAL(NAME, T, I)                              \
  template [[host_name("coo_matmul_serial_" #NAME)]] [[kernel]] void           \
  coo_matmul_serial_kernel<T, I>(                                              \
      device const T *, device const I *, device const I *, device const T *,  \
      device T *, constant int &, constant int &, constant int &, uint)

INSTANTIATE_COO_MATMUL_SERIAL(float16_int32, half, int);
INSTANTIATE_COO_MATMUL_SERIAL(float16_int64, half, long);
INSTANTIATE_COO_MATMUL_SERIAL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_COO_MATMUL_SERIAL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_COO_MATMUL_SERIAL(complex64_int32, complex64_t, int);
INSTANTIATE_COO_MATMUL_SERIAL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_COO_MATMUL_SERIAL
