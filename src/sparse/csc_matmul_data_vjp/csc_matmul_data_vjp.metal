// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

template <typename T, typename I>
[[kernel]] void csc_matmul_data_vjp_kernel(
    device const I *indices [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device const T *rhs [[buffer(2)]], device const T *cotangent [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_cols [[buffer(5)]],
    constant int &rhs_cols [[buffer(6)]],
    uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= n_cols) {
    return;
  }
  const int rhs_offset = static_cast<int>(col) * rhs_cols;
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    const int cot_offset = static_cast<int>(indices[p]) * rhs_cols;
    typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
    for (int k = 0; k < rhs_cols; ++k) {
      acc += sparse_multiply<T>(cotangent[cot_offset + k], rhs[rhs_offset + k]);
    }
    out[p] = sparse_accumulator<T>::cast(acc);
  }
}

#define INSTANTIATE_CSC_MATMUL_DATA_VJP(NAME, T, I)                            \
  template [[host_name("csc_matmul_data_vjp_" #NAME)]] [[kernel]] void         \
  csc_matmul_data_vjp_kernel<T, I>(                                            \
      device const I *, device const I *, device const T *, device const T *,  \
      device T *, constant int &, constant int &, uint)

INSTANTIATE_CSC_MATMUL_DATA_VJP(float32_int32, float, int);
INSTANTIATE_CSC_MATMUL_DATA_VJP(float32_int64, float, long);
INSTANTIATE_CSC_MATMUL_DATA_VJP(float16_int32, half, int);
INSTANTIATE_CSC_MATMUL_DATA_VJP(float16_int64, half, long);
INSTANTIATE_CSC_MATMUL_DATA_VJP(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_MATMUL_DATA_VJP(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_MATMUL_DATA_VJP(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_MATMUL_DATA_VJP(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_MATMUL_DATA_VJP
