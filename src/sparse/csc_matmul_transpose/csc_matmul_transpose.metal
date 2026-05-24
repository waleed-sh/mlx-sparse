// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

template <typename T, typename I>
[[kernel]] void csc_matmul_transpose_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *rhs [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_cols [[buffer(5)]],
    constant int &rhs_cols [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  const int total = n_cols * rhs_cols;
  if (static_cast<int>(tid) >= total) {
    return;
  }
  const int col = static_cast<int>(tid) / rhs_cols;
  const int k = static_cast<int>(tid) - col * rhs_cols;
  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    acc += sparse_multiply<T>(data[p],
                              rhs[static_cast<int>(indices[p]) * rhs_cols + k]);
  }
  out[tid] = sparse_accumulator<T>::cast(acc);
}

#define INSTANTIATE_CSC_MATMUL_TRANSPOSE(NAME, T, I)                           \
  template [[host_name("csc_matmul_transpose_" #NAME)]] [[kernel]] void        \
  csc_matmul_transpose_kernel<T, I>(                                           \
      device const T *, device const I *, device const I *, device const T *,  \
      device T *, constant int &, constant int &, uint)

INSTANTIATE_CSC_MATMUL_TRANSPOSE(float32_int32, float, int);
INSTANTIATE_CSC_MATMUL_TRANSPOSE(float32_int64, float, long);
INSTANTIATE_CSC_MATMUL_TRANSPOSE(float16_int32, half, int);
INSTANTIATE_CSC_MATMUL_TRANSPOSE(float16_int64, half, long);
INSTANTIATE_CSC_MATMUL_TRANSPOSE(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_MATMUL_TRANSPOSE(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_MATMUL_TRANSPOSE(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_MATMUL_TRANSPOSE(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_MATMUL_TRANSPOSE
