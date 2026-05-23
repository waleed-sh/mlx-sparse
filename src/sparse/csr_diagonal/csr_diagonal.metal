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
[[kernel]] void csr_diagonal_kernel(device const T *data [[buffer(0)]],
                                    device const I *indices [[buffer(1)]],
                                    device const I *indptr [[buffer(2)]],
                                    device T *out [[buffer(3)]],
                                    constant int &diag_size [[buffer(4)]],
                                    uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) >= diag_size) {
    return;
  }

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    if (indices[p] == static_cast<I>(row)) {
      acc += typename sparse_accumulator<T>::type(data[p]);
    }
  }
  out[row] = sparse_accumulator<T>::cast(acc);
}

#define INSTANTIATE_CSR_DIAGONAL(NAME, T, I)                                   \
  template [[host_name("csr_diagonal_" #NAME)]] [[kernel]] void                \
  csr_diagonal_kernel<T, I>(device const T *, device const I *,                \
                            device const I *, device T *, constant int &,      \
                            uint)

INSTANTIATE_CSR_DIAGONAL(float32_int32, float, int);
INSTANTIATE_CSR_DIAGONAL(float32_int64, float, long);
INSTANTIATE_CSR_DIAGONAL(float16_int32, half, int);
INSTANTIATE_CSR_DIAGONAL(float16_int64, half, long);
INSTANTIATE_CSR_DIAGONAL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_DIAGONAL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_DIAGONAL(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_DIAGONAL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_DIAGONAL
