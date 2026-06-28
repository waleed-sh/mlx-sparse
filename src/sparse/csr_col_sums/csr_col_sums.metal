// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "common/metal_common.h"

[[kernel]] void csr_col_sums_zero_float32(device float *out [[buffer(0)]],
                                          constant int &n_cols [[buffer(1)]],
                                          uint col
                                          [[thread_position_in_grid]]) {
  if (static_cast<int>(col) < n_cols) {
    out[col] = 0.0f;
  }
}

[[kernel]] void csr_col_sums_zero_complex64(device complex64_t *out
                                            [[buffer(0)]],
                                            constant int &n_cols [[buffer(1)]],
                                            uint col
                                            [[thread_position_in_grid]]) {
  if (static_cast<int>(col) < n_cols) {
    out[col] = complex64_t(0.0f, 0.0f);
  }
}

template <typename I>
[[kernel]] void csr_col_sums_atomic_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device float *out [[buffer(3)]], constant int &n_rows [[buffer(4)]],
    constant int &n_cols [[buffer(5)]], uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  device atomic_float *atomic_out =
      reinterpret_cast<device atomic_float *>(out);
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    const int col = static_cast<int>(indices[p]);
    if (col >= 0 && col < n_cols) {
      atomic_fetch_add_explicit(&atomic_out[col], data[p],
                                memory_order_relaxed);
    }
  }
}

template <typename I>
[[kernel]] void csr_col_sums_atomic_complex64_kernel(
    device const complex64_t *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device complex64_t *out [[buffer(3)]], constant int &n_rows [[buffer(4)]],
    constant int &n_cols [[buffer(5)]], uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  device atomic_float *atomic_out =
      reinterpret_cast<device atomic_float *>(out);
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    const int col = static_cast<int>(indices[p]);
    if (col >= 0 && col < n_cols) {
      atomic_fetch_add_explicit(&atomic_out[2 * col], data[p].real,
                                memory_order_relaxed);
      atomic_fetch_add_explicit(&atomic_out[2 * col + 1], data[p].imag,
                                memory_order_relaxed);
    }
  }
}

template [[host_name("csr_col_sums_atomic_int32")]] [[kernel]] void
csr_col_sums_atomic_kernel<int>(device const float *, device const int *,
                                device const int *, device float *,
                                constant int &, constant int &, uint);
template [[host_name("csr_col_sums_atomic_int64")]] [[kernel]] void
csr_col_sums_atomic_kernel<long>(device const float *, device const long *,
                                 device const long *, device float *,
                                 constant int &, constant int &, uint);

template [[host_name("csr_col_sums_atomic_complex64_int32")]] [[kernel]] void
csr_col_sums_atomic_complex64_kernel<int>(device const complex64_t *,
                                          device const int *,
                                          device const int *,
                                          device complex64_t *, constant int &,
                                          constant int &, uint);
template [[host_name("csr_col_sums_atomic_complex64_int64")]] [[kernel]] void
csr_col_sums_atomic_complex64_kernel<long>(device const complex64_t *,
                                           device const long *,
                                           device const long *,
                                           device complex64_t *, constant int &,
                                           constant int &, uint);
