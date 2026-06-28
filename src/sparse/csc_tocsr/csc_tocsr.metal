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

[[kernel]] void csc_tocsr_zero_offsets(device int *offsets [[buffer(0)]],
                                       constant int &n_rows [[buffer(1)]],
                                       uint tid [[thread_position_in_grid]]) {
  if (tid <= static_cast<uint>(n_rows)) {
    offsets[tid] = 0;
  }
}

template <typename I>
[[kernel]] void csc_tocsr_count_kernel(device const I *indices [[buffer(0)]],
                                       device int *offsets [[buffer(1)]],
                                       constant int &nnz [[buffer(2)]],
                                       uint tid [[thread_position_in_grid]]) {
  if (tid >= static_cast<uint>(nnz)) {
    return;
  }
  const int row = static_cast<int>(indices[tid]);
  device atomic_int *atomic_offsets =
      reinterpret_cast<device atomic_int *>(offsets);
  atomic_fetch_add_explicit(&atomic_offsets[row + 1], 1, memory_order_relaxed);
}

template <typename I>
[[kernel]] void csc_tocsr_prefix_kernel(device const int *counts [[buffer(0)]],
                                        device int *next [[buffer(1)]],
                                        device I *out_indptr [[buffer(2)]],
                                        constant int &n_rows [[buffer(3)]],
                                        constant int &nnz [[buffer(4)]],
                                        uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }

  int running = 0;
  for (int row = 0; row < n_rows; ++row) {
    const int count = counts[row + 1];
    next[row] = running;
    out_indptr[row] = static_cast<I>(running);
    running += count;
  }
  next[n_rows] = running;
  out_indptr[n_rows] = static_cast<I>(nnz);
}

template <typename T, typename I>
[[kernel]] void csc_tocsr_fill_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device int *offsets [[buffer(3)]],
    device T *out_data [[buffer(4)]], device I *out_indices [[buffer(5)]],
    constant int &n_cols [[buffer(6)]], uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= n_cols) {
    return;
  }

  device atomic_int *atomic_offsets =
      reinterpret_cast<device atomic_int *>(offsets);
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    const int row = static_cast<int>(indices[p]);
    const int dst = atomic_fetch_add_explicit(&atomic_offsets[row], 1,
                                              memory_order_relaxed);
    out_data[dst] = data[p];
    out_indices[dst] = static_cast<I>(col);
  }
}

template <typename T, typename I>
[[kernel]] void csc_tocsr_data_vjp_kernel(
    device const T *cotangent [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const I *out_indices [[buffer(3)]],
    device const I *out_indptr [[buffer(4)]], device T *out [[buffer(5)]],
    constant int &n_rows [[buffer(6)]], constant int &n_cols [[buffer(7)]],
    uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= n_cols) {
    return;
  }

  const I col_i = static_cast<I>(col);
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    const I row = indices[p];
    if (row < I(0) || static_cast<int>(row) >= n_rows) {
      out[p] = T(0);
      continue;
    }

    int duplicate_ordinal = 0;
    for (I q = indptr[col]; q < p; ++q) {
      if (indices[q] == row) {
        duplicate_ordinal += 1;
      }
    }

    int seen = 0;
    T value = T(0);
    for (I dst = out_indptr[row]; dst < out_indptr[row + I(1)]; ++dst) {
      if (out_indices[dst] != col_i) {
        continue;
      }
      if (seen == duplicate_ordinal) {
        value = cotangent[dst];
        break;
      }
      seen += 1;
    }
    out[p] = value;
  }
}

template [[host_name("csc_tocsr_count_int32")]] [[kernel]] void
csc_tocsr_count_kernel<int>(device const int *, device int *, constant int &,
                            uint);
template [[host_name("csc_tocsr_count_int64")]] [[kernel]] void
csc_tocsr_count_kernel<long>(device const long *, device int *, constant int &,
                             uint);

template [[host_name("csc_tocsr_prefix_int32")]] [[kernel]] void
csc_tocsr_prefix_kernel<int>(device const int *, device int *, device int *,
                             constant int &, constant int &, uint);
template [[host_name("csc_tocsr_prefix_int64")]] [[kernel]] void
csc_tocsr_prefix_kernel<long>(device const int *, device int *, device long *,
                              constant int &, constant int &, uint);

#define INSTANTIATE_CSC_TOCSR_FILL(NAME, T, I)                                 \
  template [[host_name("csc_tocsr_fill_" #NAME)]] [[kernel]] void              \
  csc_tocsr_fill_kernel<T, I>(device const T *, device const I *,              \
                              device const I *, device int *, device T *,      \
                              device I *, constant int &, uint)

INSTANTIATE_CSC_TOCSR_FILL(float32_int32, float, int);
INSTANTIATE_CSC_TOCSR_FILL(float32_int64, float, long);
INSTANTIATE_CSC_TOCSR_FILL(float16_int32, half, int);
INSTANTIATE_CSC_TOCSR_FILL(float16_int64, half, long);
INSTANTIATE_CSC_TOCSR_FILL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_TOCSR_FILL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_TOCSR_FILL(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_TOCSR_FILL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_TOCSR_FILL

#define INSTANTIATE_CSC_TOCSR_DATA_VJP(NAME, T, I)                             \
  template [[host_name("csc_tocsr_data_vjp_" #NAME)]] [[kernel]] void          \
  csc_tocsr_data_vjp_kernel<T, I>(                                             \
      device const T *, device const I *, device const I *, device const I *,  \
      device const I *, device T *, constant int &, constant int &, uint)

INSTANTIATE_CSC_TOCSR_DATA_VJP(float32_int32, float, int);
INSTANTIATE_CSC_TOCSR_DATA_VJP(float32_int64, float, long);
INSTANTIATE_CSC_TOCSR_DATA_VJP(float16_int32, half, int);
INSTANTIATE_CSC_TOCSR_DATA_VJP(float16_int64, half, long);
INSTANTIATE_CSC_TOCSR_DATA_VJP(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_TOCSR_DATA_VJP(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_TOCSR_DATA_VJP(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_TOCSR_DATA_VJP(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_TOCSR_DATA_VJP
