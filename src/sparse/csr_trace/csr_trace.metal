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

template <typename T, typename I>
[[kernel]] void csr_trace_blocks_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
    device typename sparse_accumulator<T>::type *partials [[buffer(3)]],
    constant int &diag_size [[buffer(4)]],
    constant int &rows_per_block [[buffer(5)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[k_csr_trace_threads];

  const int block_start = static_cast<int>(block) * rows_per_block;
  const int block_end = min(block_start + rows_per_block, diag_size);

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (int row = block_start + static_cast<int>(lane); row < block_end;
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
    partials[block] = partial[0];
  }
}

template [[host_name("csr_trace_blocks_float32_int32")]] [[kernel]] void
csr_trace_blocks_kernel<float, int>(device const float *, device const int *,
                                    device const int *, device float *,
                                    constant int &, constant int &, uint, uint);
template [[host_name("csr_trace_blocks_float32_int64")]] [[kernel]] void
csr_trace_blocks_kernel<float, long>(device const float *, device const long *,
                                     device const long *, device float *,
                                     constant int &, constant int &, uint,
                                     uint);
template [[host_name("csr_trace_blocks_float16_int32")]] [[kernel]] void
csr_trace_blocks_kernel<half, int>(device const half *, device const int *,
                                   device const int *, device float *,
                                   constant int &, constant int &, uint, uint);
template [[host_name("csr_trace_blocks_float16_int64")]] [[kernel]] void
csr_trace_blocks_kernel<half, long>(device const half *, device const long *,
                                    device const long *, device float *,
                                    constant int &, constant int &, uint, uint);
template [[host_name("csr_trace_blocks_bfloat16_int32")]] [[kernel]] void
csr_trace_blocks_kernel<bfloat16_t, int>(device const bfloat16_t *,
                                         device const int *, device const int *,
                                         device float *, constant int &,
                                         constant int &, uint, uint);
template [[host_name("csr_trace_blocks_bfloat16_int64")]] [[kernel]] void
csr_trace_blocks_kernel<bfloat16_t, long>(device const bfloat16_t *,
                                          device const long *,
                                          device const long *, device float *,
                                          constant int &, constant int &, uint,
                                          uint);
template [[host_name("csr_trace_blocks_complex64_int32")]] [[kernel]] void
csr_trace_blocks_kernel<complex64_t, int>(device const complex64_t *,
                                          device const int *,
                                          device const int *,
                                          device complex64_t *, constant int &,
                                          constant int &, uint, uint);
template [[host_name("csr_trace_blocks_complex64_int64")]] [[kernel]] void
csr_trace_blocks_kernel<complex64_t, long>(device const complex64_t *,
                                           device const long *,
                                           device const long *,
                                           device complex64_t *, constant int &,
                                           constant int &, uint, uint);

template <typename T>
[[kernel]] void csr_trace_finalize_kernel(
    device const typename sparse_accumulator<T>::type *partials [[buffer(0)]],
    device T *out [[buffer(1)]], constant int &num_blocks [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[k_csr_trace_threads];

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (int block = static_cast<int>(lane); block < num_blocks;
       block += static_cast<int>(k_csr_trace_threads)) {
    acc += partials[block];
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

template [[host_name("csr_trace_finalize_float32")]] [[kernel]] void
csr_trace_finalize_kernel<float>(device const float *, device float *,
                                 constant int &, uint);
template [[host_name("csr_trace_finalize_float16")]] [[kernel]] void
csr_trace_finalize_kernel<half>(device const float *, device half *,
                                constant int &, uint);
template [[host_name("csr_trace_finalize_bfloat16")]] [[kernel]] void
csr_trace_finalize_kernel<bfloat16_t>(device const float *, device bfloat16_t *,
                                      constant int &, uint);
template [[host_name("csr_trace_finalize_complex64")]] [[kernel]] void
csr_trace_finalize_kernel<complex64_t>(device const complex64_t *,
                                       device complex64_t *, constant int &,
                                       uint);
