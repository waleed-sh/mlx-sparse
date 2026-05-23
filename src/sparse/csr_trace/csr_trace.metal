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

constant uint k_csr_trace_threads = 128;

template <typename T, typename I>
[[kernel]] void csr_trace_kernel(device const T *data [[buffer(0)]],
                                 device const I *indices [[buffer(1)]],
                                 device const I *indptr [[buffer(2)]],
                                 device T *out [[buffer(3)]],
                                 constant int &diag_size [[buffer(4)]],
                                 uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[k_csr_trace_threads];

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (int row = static_cast<int>(lane); row < diag_size;
       row += static_cast<int>(k_csr_trace_threads)) {
    for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
      if (indices[p] == static_cast<I>(row)) {
        acc += typename sparse_accumulator<T>::type(data[p]);
      }
    }
  }

  partial[lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = k_csr_trace_threads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    out[0] = sparse_accumulator<T>::cast(partial[0]);
  }
}

#define INSTANTIATE_CSR_TRACE(NAME, T, I)                                      \
  template [[host_name("csr_trace_" #NAME)]] [[kernel]] void                   \
  csr_trace_kernel<T, I>(device const T *, device const I *, device const I *, \
                         device T *, constant int &, uint)

INSTANTIATE_CSR_TRACE(float32_int32, float, int);
INSTANTIATE_CSR_TRACE(float32_int64, float, long);
INSTANTIATE_CSR_TRACE(float16_int32, half, int);
INSTANTIATE_CSR_TRACE(float16_int64, half, long);
INSTANTIATE_CSR_TRACE(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_TRACE(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_TRACE(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_TRACE(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_TRACE
