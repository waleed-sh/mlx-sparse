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

template <typename T>
[[kernel]] void coo_kron_data_kernel(device const T *lhs_data [[buffer(0)]],
                                     device const T *rhs_data [[buffer(1)]],
                                     device T *out [[buffer(2)]],
                                     constant int &lhs_nnz [[buffer(3)]],
                                     constant int &rhs_nnz [[buffer(4)]],
                                     uint k [[thread_position_in_grid]]) {
  const uint total = static_cast<uint>(lhs_nnz * rhs_nnz);
  if (k >= total) {
    return;
  }
  const int i = static_cast<int>(k) / rhs_nnz;
  const int j = static_cast<int>(k) - i * rhs_nnz;
  typedef typename sparse_accumulator<T>::type acc_t;
  out[k] = sparse_accumulator<T>::cast(acc_t(lhs_data[i]) * acc_t(rhs_data[j]));
}

template <>
[[host_name("coo_kron_data_complex64")]] [[kernel]]
void coo_kron_data_kernel<complex64_t>(
    device const complex64_t *lhs_data [[buffer(0)]],
    device const complex64_t *rhs_data [[buffer(1)]],
    device complex64_t *out [[buffer(2)]], constant int &lhs_nnz [[buffer(3)]],
    constant int &rhs_nnz [[buffer(4)]], uint k [[thread_position_in_grid]]) {
  const uint total = static_cast<uint>(lhs_nnz * rhs_nnz);
  if (k >= total) {
    return;
  }
  const int i = static_cast<int>(k) / rhs_nnz;
  const int j = static_cast<int>(k) - i * rhs_nnz;
  out[k] = lhs_data[i] * rhs_data[j];
}

template <typename T>
[[kernel]] void coo_kron_data_vjp_kernel(
    device const T *other_data [[buffer(0)]],
    device const T *cotangent [[buffer(1)]], device T *out [[buffer(2)]],
    constant int &lhs_nnz [[buffer(3)]], constant int &rhs_nnz [[buffer(4)]],
    constant int &lhs_grad [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const int out_size = lhs_grad != 0 ? lhs_nnz : rhs_nnz;
  if (index >= static_cast<uint>(out_size)) {
    return;
  }
  typedef typename sparse_accumulator<T>::type acc_t;
  acc_t acc = sparse_accumulator<T>::zero();
  if (lhs_grad != 0) {
    const int base = static_cast<int>(index) * rhs_nnz;
    for (int j = 0; j < rhs_nnz; ++j) {
      acc += acc_t(cotangent[base + j]) * acc_t(other_data[j]);
    }
  } else {
    for (int i = 0; i < lhs_nnz; ++i) {
      acc += acc_t(cotangent[i * rhs_nnz + static_cast<int>(index)]) *
             acc_t(other_data[i]);
    }
  }
  out[index] = sparse_accumulator<T>::cast(acc);
}

template <>
[[host_name("coo_kron_data_vjp_complex64")]] [[kernel]]
void coo_kron_data_vjp_kernel<complex64_t>(
    device const complex64_t *other_data [[buffer(0)]],
    device const complex64_t *cotangent [[buffer(1)]],
    device complex64_t *out [[buffer(2)]], constant int &lhs_nnz [[buffer(3)]],
    constant int &rhs_nnz [[buffer(4)]], constant int &lhs_grad [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const int out_size = lhs_grad != 0 ? lhs_nnz : rhs_nnz;
  if (index >= static_cast<uint>(out_size)) {
    return;
  }
  complex64_t acc = complex64_t(0.0f, 0.0f);
  if (lhs_grad != 0) {
    const int base = static_cast<int>(index) * rhs_nnz;
    for (int j = 0; j < rhs_nnz; ++j) {
      acc += cotangent[base + j] * other_data[j];
    }
  } else {
    for (int i = 0; i < lhs_nnz; ++i) {
      acc += cotangent[i * rhs_nnz + static_cast<int>(index)] * other_data[i];
    }
  }
  out[index] = acc;
}

template <typename LhsI, typename RhsI, typename OutI>
[[kernel]] void coo_kron_indices_kernel(
    device const LhsI *lhs_row [[buffer(0)]],
    device const LhsI *lhs_col [[buffer(1)]],
    device const RhsI *rhs_row [[buffer(2)]],
    device const RhsI *rhs_col [[buffer(3)]],
    device OutI *out_row [[buffer(4)]], device OutI *out_col [[buffer(5)]],
    constant int &lhs_nnz [[buffer(6)]], constant int &rhs_nnz [[buffer(7)]],
    constant int &rhs_n_rows [[buffer(8)]],
    constant int &rhs_n_cols [[buffer(9)]],
    uint k [[thread_position_in_grid]]) {
  const uint total = static_cast<uint>(lhs_nnz * rhs_nnz);
  if (k >= total) {
    return;
  }
  const int i = static_cast<int>(k) / rhs_nnz;
  const int j = static_cast<int>(k) - i * rhs_nnz;
  out_row[k] = OutI(static_cast<long>(lhs_row[i]) * rhs_n_rows +
                    static_cast<long>(rhs_row[j]));
  out_col[k] = OutI(static_cast<long>(lhs_col[i]) * rhs_n_cols +
                    static_cast<long>(rhs_col[j]));
}

#define INSTANTIATE_COO_KRON_DATA(NAME, T)                                     \
  template [[host_name("coo_kron_data_" #NAME)]] [[kernel]] void               \
  coo_kron_data_kernel<T>(device const T *, device const T *, device T *,      \
                          constant int &, constant int &, uint);               \
  template [[host_name("coo_kron_data_vjp_" #NAME)]] [[kernel]] void           \
  coo_kron_data_vjp_kernel<T>(device const T *, device const T *, device T *,  \
                              constant int &, constant int &, constant int &,  \
                              uint)

INSTANTIATE_COO_KRON_DATA(float32, float);
INSTANTIATE_COO_KRON_DATA(float16, half);
INSTANTIATE_COO_KRON_DATA(bfloat16, bfloat16_t);

#undef INSTANTIATE_COO_KRON_DATA

#define INSTANTIATE_COO_KRON_INDICES(LNAME, RNAME, ONAME, LHSI, RHSI, OUTI)    \
  template [[host_name("coo_kron_indices_" #LNAME "_" #RNAME                   \
                       "_" #ONAME)]] [[kernel]] void                           \
  coo_kron_indices_kernel<LHSI, RHSI, OUTI>(                                   \
      device const LHSI *, device const LHSI *, device const RHSI *,           \
      device const RHSI *, device OUTI *, device OUTI *, constant int &,       \
      constant int &, constant int &, constant int &, uint)

INSTANTIATE_COO_KRON_INDICES(int32, int32, int32, int, int, int);
INSTANTIATE_COO_KRON_INDICES(int32, int32, int64, int, int, long);
INSTANTIATE_COO_KRON_INDICES(int32, int64, int64, int, long, long);
INSTANTIATE_COO_KRON_INDICES(int64, int32, int64, long, int, long);
INSTANTIATE_COO_KRON_INDICES(int64, int64, int64, long, long, long);

#undef INSTANTIATE_COO_KRON_INDICES
