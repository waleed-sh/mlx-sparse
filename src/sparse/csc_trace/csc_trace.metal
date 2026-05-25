// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

constant uint k_csc_trace_threads = 128;

template <typename T, typename I>
[[kernel]] void csc_trace_kernel(device const T *data [[buffer(0)]],
                                 device const I *indices [[buffer(1)]],
                                 device const I *indptr [[buffer(2)]],
                                 device T *out [[buffer(3)]],
                                 constant int &diag_size [[buffer(4)]],
                                 uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[k_csc_trace_threads];

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (int col = static_cast<int>(lane); col < diag_size;
       col += static_cast<int>(k_csc_trace_threads)) {
    for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
      if (indices[p] == static_cast<I>(col)) {
        acc += typename sparse_accumulator<T>::type(data[p]);
      }
    }
  }

  partial[lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = k_csc_trace_threads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    out[0] = sparse_accumulator<T>::cast(partial[0]);
  }
}

#define INSTANTIATE_CSC_TRACE(NAME, T, I)                                      \
  template [[host_name("csc_trace_" #NAME)]] [[kernel]] void                   \
  csc_trace_kernel<T, I>(device const T *, device const I *, device const I *, \
                         device T *, constant int &, uint)

INSTANTIATE_CSC_TRACE(float32_int32, float, int);
INSTANTIATE_CSC_TRACE(float32_int64, float, long);
INSTANTIATE_CSC_TRACE(float16_int32, half, int);
INSTANTIATE_CSC_TRACE(float16_int64, half, long);
INSTANTIATE_CSC_TRACE(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_TRACE(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_TRACE(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_TRACE(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_TRACE
