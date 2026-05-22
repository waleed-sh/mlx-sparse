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
[[kernel]] void csr_matmul_data_vjp_kernel(
    device const I *indices [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device const T *rhs [[buffer(2)]], device const T *cotangent [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_rows [[buffer(5)]],
    constant int &rhs_cols [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
    const int col = static_cast<int>(indices[p]);
    for (int k = 0; k < rhs_cols; ++k) {
      acc += sparse_multiply<T>(cotangent[row * rhs_cols + k],
                                rhs[col * rhs_cols + k]);
    }
    out[p] = sparse_accumulator<T>::cast(acc);
  }
}

#define INSTANTIATE_CSR_MATMUL_DATA_VJP(NAME, T, I)                            \
  template [[host_name("csr_matmul_data_vjp_" #NAME)]] [[kernel]] void         \
  csr_matmul_data_vjp_kernel<T, I>(                                            \
      device const I *, device const I *, device const T *, device const T *,  \
      device T *, constant int &, constant int &, uint)

INSTANTIATE_CSR_MATMUL_DATA_VJP(float32_int32, float, int);
INSTANTIATE_CSR_MATMUL_DATA_VJP(float32_int64, float, long);
INSTANTIATE_CSR_MATMUL_DATA_VJP(float16_int32, half, int);
INSTANTIATE_CSR_MATMUL_DATA_VJP(float16_int64, half, long);
INSTANTIATE_CSR_MATMUL_DATA_VJP(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_MATMUL_DATA_VJP(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_MATMUL_DATA_VJP(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_MATMUL_DATA_VJP(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_MATMUL_DATA_VJP
