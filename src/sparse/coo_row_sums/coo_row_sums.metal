// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

[[kernel]] void coo_row_sums_zero_float32(device float *out [[buffer(0)]],
                                          constant int &n_rows [[buffer(1)]],
                                          uint row
                                          [[thread_position_in_grid]]) {
  if (static_cast<int>(row) < n_rows) {
    out[row] = 0.0f;
  }
}

[[kernel]] void coo_row_sums_zero_complex64(device complex64_t *out
                                            [[buffer(0)]],
                                            constant int &n_rows [[buffer(1)]],
                                            uint row
                                            [[thread_position_in_grid]]) {
  if (static_cast<int>(row) < n_rows) {
    out[row] = complex64_t(0.0f, 0.0f);
  }
}

template <typename I>
[[kernel]] void coo_row_sums_atomic_kernel(
    device const float *data [[buffer(0)]], device const I *row [[buffer(1)]],
    device float *out [[buffer(2)]], constant int &nnz [[buffer(3)]],
    constant int &n_rows [[buffer(4)]], device const I *col [[buffer(5)]],
    constant int &n_cols [[buffer(6)]], uint p [[thread_position_in_grid]]) {
  if (static_cast<int>(p) >= nnz) {
    return;
  }

  // Both axes and in the index type. The conversion detour this operation
  // takes off float32 drops an entry whose other coordinate is out of range,
  // and casting to int first folds an index above INT_MAX back into range.
  if (coo_entry_in_range(row[p], col[p], n_rows, n_cols)) {
    const int r = static_cast<int>(row[p]);
    device atomic_float *atomic_out =
        reinterpret_cast<device atomic_float *>(out);
    atomic_fetch_add_explicit(&atomic_out[r], data[p], memory_order_relaxed);
  }
}

template <typename I>
[[kernel]] void coo_row_sums_atomic_complex64_kernel(
    device const complex64_t *data [[buffer(0)]],
    device const I *row [[buffer(1)]], device complex64_t *out [[buffer(2)]],
    constant int &nnz [[buffer(3)]], constant int &n_rows [[buffer(4)]],
    device const I *col [[buffer(5)]], constant int &n_cols [[buffer(6)]],
    uint p [[thread_position_in_grid]]) {
  if (static_cast<int>(p) >= nnz) {
    return;
  }

  // Both axes and in the index type. The conversion detour this operation
  // takes off float32 drops an entry whose other coordinate is out of range,
  // and casting to int first folds an index above INT_MAX back into range.
  if (coo_entry_in_range(row[p], col[p], n_rows, n_cols)) {
    const int r = static_cast<int>(row[p]);
    device atomic_float *atomic_out =
        reinterpret_cast<device atomic_float *>(out);
    atomic_fetch_add_explicit(&atomic_out[2 * r], data[p].real,
                              memory_order_relaxed);
    atomic_fetch_add_explicit(&atomic_out[2 * r + 1], data[p].imag,
                              memory_order_relaxed);
  }
}

template [[host_name("coo_row_sums_atomic_int32")]] [[kernel]] void
coo_row_sums_atomic_kernel<int>(device const float *, device const int *,
                                device float *, constant int &, constant int &,
                                device const int *, constant int &, uint);
template [[host_name("coo_row_sums_atomic_int64")]] [[kernel]] void
coo_row_sums_atomic_kernel<long>(device const float *, device const long *,
                                 device float *, constant int &,
                                 constant int &, device const long *,
                                 constant int &, uint);

template [[host_name("coo_row_sums_atomic_complex64_int32")]] [[kernel]] void
coo_row_sums_atomic_complex64_kernel<int>(device const complex64_t *,
                                          device const int *,
                                          device complex64_t *, constant int &,
                                          constant int &, device const int *,
                                          constant int &, uint);
template [[host_name("coo_row_sums_atomic_complex64_int64")]] [[kernel]] void
coo_row_sums_atomic_complex64_kernel<long>(device const complex64_t *,
                                           device const long *,
                                           device complex64_t *,
                                           constant int &, constant int &,
                                           device const long *,
                                           constant int &, uint);
