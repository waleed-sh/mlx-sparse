// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

template <typename T, typename I>
[[kernel]] void coo_matmul_data_vjp_kernel(
    device const I *row [[buffer(0)]], device const I *col [[buffer(1)]],
    device const T *rhs [[buffer(2)]], device const T *cotangent [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &rhs_cols [[buffer(5)]],
    constant int &nnz [[buffer(6)]], uint p [[thread_position_in_grid]]) {
  if (static_cast<int>(p) >= nnz) {
    return;
  }
  const int rhs_offset = static_cast<int>(col[p]) * rhs_cols;
  const int cot_offset = static_cast<int>(row[p]) * rhs_cols;
  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (int k = 0; k < rhs_cols; ++k) {
    acc += sparse_multiply<T>(cotangent[cot_offset + k], rhs[rhs_offset + k]);
  }
  out[p] = sparse_accumulator<T>::cast(acc);
}

#define INSTANTIATE_COO_MATMUL_DATA_VJP(NAME, T, I)                            \
  template [[host_name("coo_matmul_data_vjp_" #NAME)]] [[kernel]] void         \
  coo_matmul_data_vjp_kernel<T, I>(                                            \
      device const I *, device const I *, device const T *, device const T *,  \
      device T *, constant int &, constant int &, uint)

INSTANTIATE_COO_MATMUL_DATA_VJP(float32_int32, float, int);
INSTANTIATE_COO_MATMUL_DATA_VJP(float32_int64, float, long);
INSTANTIATE_COO_MATMUL_DATA_VJP(float16_int32, half, int);
INSTANTIATE_COO_MATMUL_DATA_VJP(float16_int64, half, long);
INSTANTIATE_COO_MATMUL_DATA_VJP(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_COO_MATMUL_DATA_VJP(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_COO_MATMUL_DATA_VJP(complex64_int32, complex64_t, int);
INSTANTIATE_COO_MATMUL_DATA_VJP(complex64_int64, complex64_t, long);

#undef INSTANTIATE_COO_MATMUL_DATA_VJP
