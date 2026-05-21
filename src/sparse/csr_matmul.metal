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

#include "sparse/metal_common.h"

template <typename T, typename I>
[[kernel]] void csr_matmul_kernel(device const T *data [[buffer(0)]],
                                  device const I *indices [[buffer(1)]],
                                  device const I *indptr [[buffer(2)]],
                                  device const T *rhs [[buffer(3)]],
                                  device T *out [[buffer(4)]],
                                  constant int &n_rows [[buffer(5)]],
                                  constant int &rhs_cols [[buffer(6)]],
                                  uint tid [[thread_position_in_grid]]) {
  const int total = n_rows * rhs_cols;
  if (tid >= static_cast<uint>(total)) {
    return;
  }

  const int row = static_cast<int>(tid) / rhs_cols;
  const int rhs_col = static_cast<int>(tid) - row * rhs_cols;

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    acc += sparse_multiply<T>(data[p], rhs[indices[p] * rhs_cols + rhs_col]);
  }
  out[row * rhs_cols + rhs_col] = sparse_accumulator<T>::cast(acc);
}

template [[host_name("csr_matmul_float32_int32")]] [[kernel]] void
csr_matmul_kernel<float, int>(device const float *, device const int *,
                              device const int *, device const float *,
                              device float *, constant int &, constant int &,
                              uint);
template [[host_name("csr_matmul_float32_int64")]] [[kernel]] void
csr_matmul_kernel<float, long>(device const float *, device const long *,
                               device const long *, device const float *,
                               device float *, constant int &, constant int &,
                               uint);
template [[host_name("csr_matmul_float16_int32")]] [[kernel]] void
csr_matmul_kernel<half, int>(device const half *, device const int *,
                             device const int *, device const half *,
                             device half *, constant int &, constant int &,
                             uint);
template [[host_name("csr_matmul_float16_int64")]] [[kernel]] void
csr_matmul_kernel<half, long>(device const half *, device const long *,
                              device const long *, device const half *,
                              device half *, constant int &, constant int &,
                              uint);
template [[host_name("csr_matmul_bfloat16_int32")]] [[kernel]] void
csr_matmul_kernel<bfloat16_t, int>(device const bfloat16_t *,
                                   device const int *, device const int *,
                                   device const bfloat16_t *,
                                   device bfloat16_t *, constant int &,
                                   constant int &, uint);
template [[host_name("csr_matmul_bfloat16_int64")]] [[kernel]] void
csr_matmul_kernel<bfloat16_t, long>(device const bfloat16_t *,
                                    device const long *, device const long *,
                                    device const bfloat16_t *,
                                    device bfloat16_t *, constant int &,
                                    constant int &, uint);
template [[host_name("csr_matmul_complex64_int32")]] [[kernel]] void
csr_matmul_kernel<complex64_t, int>(device const complex64_t *,
                                    device const int *, device const int *,
                                    device const complex64_t *,
                                    device complex64_t *, constant int &,
                                    constant int &, uint);
template [[host_name("csr_matmul_complex64_int64")]] [[kernel]] void
csr_matmul_kernel<complex64_t, long>(device const complex64_t *,
                                     device const long *, device const long *,
                                     device const complex64_t *,
                                     device complex64_t *, constant int &,
                                     constant int &, uint);

template <typename T, typename I>
[[kernel]] void csr_matmul_vector_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *rhs [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_rows [[buffer(5)]],
    constant int &rhs_cols [[buffer(6)]],
    uint out_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup typename sparse_accumulator<T>::type partial[128];

  const int row = static_cast<int>(out_id) / rhs_cols;
  const int rhs_col = static_cast<int>(out_id) - row * rhs_cols;
  if (row >= n_rows) {
    return;
  }

  typename sparse_accumulator<T>::type acc = sparse_accumulator<T>::zero();
  for (I p = indptr[row] + static_cast<I>(lane); p < indptr[row + 1];
       p += 128) {
    acc += sparse_multiply<T>(data[p], rhs[indices[p] * rhs_cols + rhs_col]);
  }
  partial[lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 64; stride > 0; stride >>= 1) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    out[row * rhs_cols + rhs_col] = sparse_accumulator<T>::cast(partial[0]);
  }
}

#define INSTANTIATE_CSR_MATMUL_VECTOR(NAME, T, I)                              \
  template [[host_name("csr_matmul_vector_" #NAME)]] [[kernel]] void           \
  csr_matmul_vector_kernel<T, I>(                                              \
      device const T *, device const I *, device const I *, device const T *,  \
      device T *, constant int &, constant int &, uint, uint)

INSTANTIATE_CSR_MATMUL_VECTOR(float32_int32, float, int);
INSTANTIATE_CSR_MATMUL_VECTOR(float32_int64, float, long);
INSTANTIATE_CSR_MATMUL_VECTOR(float16_int32, half, int);
INSTANTIATE_CSR_MATMUL_VECTOR(float16_int64, half, long);
INSTANTIATE_CSR_MATMUL_VECTOR(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_MATMUL_VECTOR(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_MATMUL_VECTOR(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_MATMUL_VECTOR(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_MATMUL_VECTOR

template <typename T, typename I>
[[kernel]] void csr_matmul_transpose_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device const T *rhs [[buffer(3)]],
    device T *out [[buffer(4)]], constant int &n_rows [[buffer(5)]],
    constant int &n_cols [[buffer(6)]], constant int &rhs_cols [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }

  for (int i = 0; i < n_cols * rhs_cols; ++i) {
    out[i] = T(0);
  }
  for (int row = 0; row < n_rows; ++row) {
    for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
      const int col = static_cast<int>(indices[p]);
      for (int k = 0; k < rhs_cols; ++k) {
        const int out_offset = col * rhs_cols + k;
        out[out_offset] = sparse_add_storage<T>(
            out[out_offset], sparse_accumulator<T>::cast(sparse_multiply<T>(
                                 data[p], rhs[row * rhs_cols + k])));
      }
    }
  }
}

#define INSTANTIATE_CSR_MATMUL_TRANSPOSE(NAME, T, I)                           \
  template [[host_name("csr_matmul_transpose_" #NAME)]] [[kernel]] void        \
  csr_matmul_transpose_kernel<T, I>(                                           \
      device const T *, device const I *, device const I *, device const T *,  \
      device T *, constant int &, constant int &, constant int &, uint)

INSTANTIATE_CSR_MATMUL_TRANSPOSE(float32_int32, float, int);
INSTANTIATE_CSR_MATMUL_TRANSPOSE(float32_int64, float, long);
INSTANTIATE_CSR_MATMUL_TRANSPOSE(float16_int32, half, int);
INSTANTIATE_CSR_MATMUL_TRANSPOSE(float16_int64, half, long);
INSTANTIATE_CSR_MATMUL_TRANSPOSE(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_MATMUL_TRANSPOSE(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_MATMUL_TRANSPOSE(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_MATMUL_TRANSPOSE(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_MATMUL_TRANSPOSE

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
