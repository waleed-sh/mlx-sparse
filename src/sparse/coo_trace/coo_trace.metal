// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "common/metal_common.h"

constant uint k_coo_trace_threads = 128;

template <typename T, typename I>
[[kernel]] void coo_trace_kernel(device const T *data [[buffer(0)]],
                                 device const I *row [[buffer(1)]],
                                 device const I *col [[buffer(2)]],
                                 device T *out [[buffer(3)]],
                                 constant int &nnz [[buffer(4)]],
                                 constant int &diag_size [[buffer(5)]],
                                 uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[k_coo_trace_threads];

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (int p = static_cast<int>(lane); p < nnz;
       p += static_cast<int>(k_coo_trace_threads)) {
    const int r = static_cast<int>(row[p]);
    if (r == static_cast<int>(col[p]) && r >= 0 && r < diag_size) {
      acc += typename sparse_accumulator<T>::type(data[p]);
    }
  }

  partial[lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = k_coo_trace_threads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    out[0] = sparse_accumulator<T>::cast(partial[0]);
  }
}

#define INSTANTIATE_COO_TRACE(NAME, T, I)                                      \
  template [[host_name("coo_trace_" #NAME)]] [[kernel]] void                   \
  coo_trace_kernel<T, I>(device const T *, device const I *, device const I *, \
                         device T *, constant int &, constant int &, uint)

INSTANTIATE_COO_TRACE(float32_int32, float, int);
INSTANTIATE_COO_TRACE(float32_int64, float, long);
INSTANTIATE_COO_TRACE(float16_int32, half, int);
INSTANTIATE_COO_TRACE(float16_int64, half, long);
INSTANTIATE_COO_TRACE(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_COO_TRACE(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_COO_TRACE(complex64_int32, complex64_t, int);
INSTANTIATE_COO_TRACE(complex64_int64, complex64_t, long);

#undef INSTANTIATE_COO_TRACE

template <typename T, typename I>
[[kernel]] void coo_trace_blocks_kernel(
    device const T *data [[buffer(0)]], device const I *row [[buffer(1)]],
    device const I *col [[buffer(2)]],
    device typename sparse_accumulator<T>::type *partials [[buffer(3)]],
    constant int &nnz [[buffer(4)]], constant int &diag_size [[buffer(5)]],
    constant int &entries_per_block [[buffer(6)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[k_coo_trace_threads];

  const int block_start = static_cast<int>(block) * entries_per_block;
  const int block_end = min(block_start + entries_per_block, nnz);

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (int p = block_start + static_cast<int>(lane); p < block_end;
       p += static_cast<int>(k_coo_trace_threads)) {
    const int r = static_cast<int>(row[p]);
    if (r == static_cast<int>(col[p]) && r >= 0 && r < diag_size) {
      acc += typename sparse_accumulator<T>::type(data[p]);
    }
  }

  partial[lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = k_coo_trace_threads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    partials[block] = partial[0];
  }
}

template [[host_name("coo_trace_blocks_float32_int32")]] [[kernel]] void
coo_trace_blocks_kernel<float, int>(device const float *, device const int *,
                                    device const int *, device float *,
                                    constant int &, constant int &,
                                    constant int &, uint, uint);
template [[host_name("coo_trace_blocks_float32_int64")]] [[kernel]] void
coo_trace_blocks_kernel<float, long>(device const float *, device const long *,
                                     device const long *, device float *,
                                     constant int &, constant int &,
                                     constant int &, uint, uint);
template [[host_name("coo_trace_blocks_float16_int32")]] [[kernel]] void
coo_trace_blocks_kernel<half, int>(device const half *, device const int *,
                                   device const int *, device float *,
                                   constant int &, constant int &,
                                   constant int &, uint, uint);
template [[host_name("coo_trace_blocks_float16_int64")]] [[kernel]] void
coo_trace_blocks_kernel<half, long>(device const half *, device const long *,
                                    device const long *, device float *,
                                    constant int &, constant int &,
                                    constant int &, uint, uint);
template [[host_name("coo_trace_blocks_bfloat16_int32")]] [[kernel]] void
coo_trace_blocks_kernel<bfloat16_t, int>(device const bfloat16_t *,
                                         device const int *, device const int *,
                                         device float *, constant int &,
                                         constant int &, constant int &, uint,
                                         uint);
template [[host_name("coo_trace_blocks_bfloat16_int64")]] [[kernel]] void
coo_trace_blocks_kernel<bfloat16_t, long>(device const bfloat16_t *,
                                          device const long *,
                                          device const long *, device float *,
                                          constant int &, constant int &,
                                          constant int &, uint, uint);
template [[host_name("coo_trace_blocks_complex64_int32")]] [[kernel]] void
coo_trace_blocks_kernel<complex64_t, int>(device const complex64_t *,
                                          device const int *,
                                          device const int *,
                                          device complex64_t *, constant int &,
                                          constant int &, constant int &, uint,
                                          uint);
template [[host_name("coo_trace_blocks_complex64_int64")]] [[kernel]] void
coo_trace_blocks_kernel<complex64_t, long>(device const complex64_t *,
                                           device const long *,
                                           device const long *,
                                           device complex64_t *, constant int &,
                                           constant int &, constant int &, uint,
                                           uint);

template <typename T>
[[kernel]] void coo_trace_finalize_kernel(
    device const typename sparse_accumulator<T>::type *partials [[buffer(0)]],
    device T *out [[buffer(1)]], constant int &num_blocks [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[k_coo_trace_threads];

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (int block = static_cast<int>(lane); block < num_blocks;
       block += static_cast<int>(k_coo_trace_threads)) {
    acc += partials[block];
  }

  partial[lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = k_coo_trace_threads / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    out[0] = sparse_accumulator<T>::cast(partial[0]);
  }
}

template [[host_name("coo_trace_finalize_float32")]] [[kernel]] void
coo_trace_finalize_kernel<float>(device const float *, device float *,
                                 constant int &, uint);
template [[host_name("coo_trace_finalize_float16")]] [[kernel]] void
coo_trace_finalize_kernel<half>(device const float *, device half *,
                                constant int &, uint);
template [[host_name("coo_trace_finalize_bfloat16")]] [[kernel]] void
coo_trace_finalize_kernel<bfloat16_t>(device const float *, device bfloat16_t *,
                                      constant int &, uint);
template [[host_name("coo_trace_finalize_complex64")]] [[kernel]] void
coo_trace_finalize_kernel<complex64_t>(device const complex64_t *,
                                       device complex64_t *, constant int &,
                                       uint);
