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
