// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

[[kernel]] void coo_diagonal_zero_float32(device float *out [[buffer(0)]],
                                          constant int &diag_size [[buffer(1)]],
                                          uint idx
                                          [[thread_position_in_grid]]) {
  if (static_cast<int>(idx) < diag_size) {
    out[idx] = 0.0f;
  }
}

template <typename I>
[[kernel]] void coo_diagonal_atomic_kernel(
    device const float *data [[buffer(0)]], device const I *row [[buffer(1)]],
    device const I *col [[buffer(2)]], device float *out [[buffer(3)]],
    constant int &nnz [[buffer(4)]], constant int &diag_size [[buffer(5)]],
    uint p [[thread_position_in_grid]]) {
  if (static_cast<int>(p) >= nnz) {
    return;
  }

  const int r = static_cast<int>(row[p]);
  if (r == static_cast<int>(col[p]) && r >= 0 && r < diag_size) {
    device atomic_float *atomic_out =
        reinterpret_cast<device atomic_float *>(out);
    atomic_fetch_add_explicit(&atomic_out[r], data[p], memory_order_relaxed);
  }
}

template [[host_name("coo_diagonal_atomic_int32")]] [[kernel]] void
coo_diagonal_atomic_kernel<int>(device const float *, device const int *,
                                device const int *, device float *,
                                constant int &, constant int &, uint);
template [[host_name("coo_diagonal_atomic_int64")]] [[kernel]] void
coo_diagonal_atomic_kernel<long>(device const float *, device const long *,
                                 device const long *, device float *,
                                 constant int &, constant int &, uint);
