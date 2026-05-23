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
csr_matvec_transpose_zero_float32(device float *out [[buffer(0)]],
                                  constant int &n_cols [[buffer(1)]],
                                  uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) < n_cols) {
    out[col] = 0.0f;
  }
}

[[kernel]] void
csr_matvec_transpose_zero_offsets(device int *offsets [[buffer(0)]],
                                  constant int &n_cols [[buffer(1)]],
                                  uint tid [[thread_position_in_grid]]) {
  if (tid <= static_cast<uint>(n_cols)) {
    offsets[tid] = 0;
  }
}

template <typename I>
[[kernel]] void csr_matvec_transpose_count_kernel(
    device const I *indices [[buffer(0)]], device int *offsets [[buffer(1)]],
    constant int &nnz [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  if (static_cast<int>(tid) >= nnz) {
    return;
  }

  const int col = static_cast<int>(indices[tid]);
  device atomic_int *atomic_offsets =
      reinterpret_cast<device atomic_int *>(offsets);
  atomic_fetch_add_explicit(&atomic_offsets[col + 1], 1, memory_order_relaxed);
}

[[kernel]] void csr_matvec_transpose_prefix_segments(
    device int *offsets [[buffer(0)]], device int *segments [[buffer(1)]],
    constant int &n_cols [[buffer(2)]], constant int &nnz [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }

  int running = 0;
  for (int col = 0; col < n_cols; ++col) {
    const int count = offsets[col + 1];
    offsets[col] = running;
    segments[col] = running;
    running += count;
  }
  offsets[n_cols] = running;
  segments[n_cols] = nnz;
}

template <typename I>
[[kernel]] void csr_matvec_transpose_atomic_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *x [[buffer(3)]], device float *out [[buffer(4)]],
    constant int &n_rows [[buffer(5)]], constant int &n_cols [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  device atomic_float *atomic_out =
      reinterpret_cast<device atomic_float *>(out);
  const float x_value = x[row];
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    const int col = static_cast<int>(indices[p]);
    if (col >= 0 && col < n_cols) {
      atomic_fetch_add_explicit(&atomic_out[col], data[p] * x_value,
                                memory_order_relaxed);
    }
  }
}

template <typename T, typename I, typename A>
[[kernel]] void csr_matvec_transpose_scatter_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *x [[buffer(3)]],
    device int *offsets [[buffer(4)]], device A *grouped [[buffer(5)]],
    constant int &n_rows [[buffer(6)]], constant int &n_cols [[buffer(7)]],
    uint row [[thread_position_in_grid]]) {
  if (static_cast<int>(row) >= n_rows) {
    return;
  }

  device atomic_int *atomic_offsets =
      reinterpret_cast<device atomic_int *>(offsets);
  const T x_value = x[row];
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    const int col = static_cast<int>(indices[p]);
    if (col >= 0 && col < n_cols) {
      const int dst = atomic_fetch_add_explicit(&atomic_offsets[col], 1,
                                                memory_order_relaxed);
      grouped[dst] = A(sparse_multiply<T>(data[p], x_value));
    }
  }
}

template <typename T, typename A, typename I>
[[kernel]] void csr_matvec_transpose_segmented_kernel(
    device const A *grouped [[buffer(0)]],
    device const int *segments [[buffer(1)]], device T *out [[buffer(2)]],
    constant int &n_cols [[buffer(3)]], uint col [[thread_position_in_grid]]) {
  if (static_cast<int>(col) >= n_cols) {
    return;
  }

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (int p = segments[col]; p < segments[col + 1]; ++p) {
    acc += typename sparse_accumulator<T>::type(grouped[p]);
  }
  out[col] = sparse_accumulator<T>::cast(acc);
}

template [[host_name("csr_matvec_transpose_count_int32")]] [[kernel]] void
csr_matvec_transpose_count_kernel<int>(device const int *, device int *,
                                       constant int &, uint);
template [[host_name("csr_matvec_transpose_count_int64")]] [[kernel]] void
csr_matvec_transpose_count_kernel<long>(device const long *, device int *,
                                        constant int &, uint);

template [[host_name("csr_matvec_transpose_atomic_int32")]] [[kernel]] void
csr_matvec_transpose_atomic_kernel<int>(device const float *,
                                        device const int *, device const int *,
                                        device const float *, device float *,
                                        constant int &, constant int &, uint);
template [[host_name("csr_matvec_transpose_atomic_int64")]] [[kernel]] void
csr_matvec_transpose_atomic_kernel<long>(device const float *,
                                         device const long *,
                                         device const long *,
                                         device const float *, device float *,
                                         constant int &, constant int &, uint);

#define INSTANTIATE_CSR_MATVEC_TRANSPOSE(NAME, T, I, A)                        \
  template [[host_name("csr_matvec_transpose_scatter_" #NAME)]] [[kernel]]     \
  void csr_matvec_transpose_scatter_kernel<T, I, A>(                           \
      device const T *, device const I *, device const I *, device const T *,  \
      device int *, device A *, constant int &, constant int &, uint);         \
  template [[host_name("csr_matvec_transpose_segmented_" #NAME)]] [[kernel]]   \
  void csr_matvec_transpose_segmented_kernel<T, A, I>(                         \
      device const A *, device const int *, device T *, constant int &, uint)

INSTANTIATE_CSR_MATVEC_TRANSPOSE(float16_int32, half, int, float);
INSTANTIATE_CSR_MATVEC_TRANSPOSE(float16_int64, half, long, float);
INSTANTIATE_CSR_MATVEC_TRANSPOSE(bfloat16_int32, bfloat16_t, int, float);
INSTANTIATE_CSR_MATVEC_TRANSPOSE(bfloat16_int64, bfloat16_t, long, float);

#undef INSTANTIATE_CSR_MATVEC_TRANSPOSE
