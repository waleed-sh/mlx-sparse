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

[[kernel]] void csc_matvec_zero_float32(device float *out [[buffer(0)]],
                                        constant int &n_rows [[buffer(1)]],
                                        uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) < n_rows) {
    out[row] = 0.0f;
  }
}

template <typename I>
[[kernel]] void csc_matvec_atomic_kernel(device const float *data [[buffer(0)]],
                                         device const I *indices [[buffer(1)]],
                                         device const I *indptr [[buffer(2)]],
                                         device const float *x [[buffer(3)]],
                                         device float *out [[buffer(4)]],
                                         constant int &n_rows [[buffer(5)]],
                                         constant int &n_cols [[buffer(6)]],
                                         uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= n_cols) {
    return;
  }

  device atomic_float *atomic_out =
      reinterpret_cast<device atomic_float *>(out);
  const float x_value = x[col];
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    const int row = static_cast<int>(indices[p]);
    if (row >= 0 && row < n_rows) {
      atomic_fetch_add_explicit(&atomic_out[row], data[p] * x_value,
                                memory_order_relaxed);
    }
  }
}

template <typename T, typename I>
[[kernel]] void csc_matvec_serial_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *x [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_rows [[buffer(5)]],
    constant int &n_cols [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }

  for (int row = 0; row < n_rows; ++row) {
    out[row] = T(0);
  }
  for (int col = 0; col < n_cols; ++col) {
    const T x_value = x[col];
    for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
      const int row = static_cast<int>(indices[p]);
      typedef typename sparse_accumulator<T>::type acc_t;
      const acc_t updated =
          acc_t(out[row]) + sparse_multiply<T>(data[p], x_value);
      out[row] = sparse_accumulator<T>::cast(updated);
    }
  }
}

template [[host_name("csc_matvec_atomic_int32")]] [[kernel]] void
csc_matvec_atomic_kernel<int>(device const float *, device const int *,
                              device const int *, device const float *,
                              device float *, constant int &, constant int &,
                              uint);
template [[host_name("csc_matvec_atomic_int64")]] [[kernel]] void
csc_matvec_atomic_kernel<long>(device const float *, device const long *,
                               device const long *, device const float *,
                               device float *, constant int &, constant int &,
                               uint);

#define INSTANTIATE_CSC_MATVEC_SERIAL(NAME, T, I)                              \
  template [[host_name("csc_matvec_serial_" #NAME)]] [[kernel]] void           \
  csc_matvec_serial_kernel<T, I>(                                              \
      device const T *, device const I *, device const I *, device const T *,  \
      device T *, constant int &, constant int &, uint)

INSTANTIATE_CSC_MATVEC_SERIAL(float16_int32, half, int);
INSTANTIATE_CSC_MATVEC_SERIAL(float16_int64, half, long);
INSTANTIATE_CSC_MATVEC_SERIAL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_MATVEC_SERIAL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_MATVEC_SERIAL(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_MATVEC_SERIAL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_MATVEC_SERIAL
