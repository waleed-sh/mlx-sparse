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

template <typename T, typename I>
[[kernel]] void csr_matvec_data_vjp_kernel(
    device const I *indices [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device const T *x [[buffer(2)]], device const T *cotangent [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_rows [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    out[p] = sparse_accumulator<T>::cast(
        sparse_multiply<T>(cotangent[row], x[indices[p]]));
  }
}

#define INSTANTIATE_CSR_MATVEC_DATA_VJP(NAME, T, I)                            \
  template [[host_name("csr_matvec_data_vjp_" #NAME)]] [[kernel]] void         \
  csr_matvec_data_vjp_kernel<T, I>(device const I *, device const I *,         \
                                   device const T *, device const T *,         \
                                   device T *, constant int &, uint)

INSTANTIATE_CSR_MATVEC_DATA_VJP(float32_int32, float, int);
INSTANTIATE_CSR_MATVEC_DATA_VJP(float32_int64, float, long);
INSTANTIATE_CSR_MATVEC_DATA_VJP(float16_int32, half, int);
INSTANTIATE_CSR_MATVEC_DATA_VJP(float16_int64, half, long);
INSTANTIATE_CSR_MATVEC_DATA_VJP(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_MATVEC_DATA_VJP(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_MATVEC_DATA_VJP(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_MATVEC_DATA_VJP(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_MATVEC_DATA_VJP
