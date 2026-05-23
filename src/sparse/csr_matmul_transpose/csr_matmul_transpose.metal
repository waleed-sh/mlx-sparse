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

[[kernel]] void
csr_matmul_transpose_zero_float32(device float *out [[buffer(0)]],
                                  constant int &n_cols [[buffer(1)]],
                                  constant int &rhs_cols [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]]) {
  const int total = n_cols * rhs_cols;
  if (static_cast<int>(tid) < total) {
    out[tid] = 0.0f;
  }
}

template <typename I>
[[kernel]] void csr_matmul_transpose_atomic_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *rhs [[buffer(3)]], device float *out [[buffer(4)]],
    constant int &n_rows [[buffer(5)]], constant int &n_cols [[buffer(6)]],
    constant int &rhs_cols [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  const int total = n_rows * rhs_cols;
  if (static_cast<int>(tid) >= total) {
    return;
  }

  const int row = static_cast<int>(tid) / rhs_cols;
  const int rhs_col = static_cast<int>(tid) - row * rhs_cols;
  const float rhs_value = rhs[row * rhs_cols + rhs_col];
  device atomic_float *atomic_out =
      reinterpret_cast<device atomic_float *>(out);

  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    const int col = static_cast<int>(indices[p]);
    if (col >= 0 && col < n_cols) {
      const int out_offset = col * rhs_cols + rhs_col;
      atomic_fetch_add_explicit(&atomic_out[out_offset], data[p] * rhs_value,
                                memory_order_relaxed);
    }
  }
}

template [[host_name("csr_matmul_transpose_atomic_int32")]] [[kernel]] void
csr_matmul_transpose_atomic_kernel<int>(device const float *,
                                        device const int *, device const int *,
                                        device const float *, device float *,
                                        constant int &, constant int &,
                                        constant int &, uint);
template [[host_name("csr_matmul_transpose_atomic_int64")]] [[kernel]] void
csr_matmul_transpose_atomic_kernel<long>(device const float *,
                                         device const long *,
                                         device const long *,
                                         device const float *, device float *,
                                         constant int &, constant int &,
                                         constant int &, uint);
