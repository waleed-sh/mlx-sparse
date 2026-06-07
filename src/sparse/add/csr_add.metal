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

template <typename T> inline bool csr_add_nonzero(T value) {
  return !(value == T(0));
}

template <> inline bool csr_add_nonzero<complex64_t>(complex64_t value) {
  return value.real != 0.0f || value.imag != 0.0f;
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
[[kernel]] void
csr_add_counts_kernel(device const T *lhs_data [[buffer(0)]],
                      device const LhsI *lhs_indices [[buffer(1)]],
                      device const LhsI *lhs_indptr [[buffer(2)]],
                      device const T *rhs_data [[buffer(3)]],
                      device const RhsI *rhs_indices [[buffer(4)]],
                      device const RhsI *rhs_indptr [[buffer(5)]],
                      device OutI *counts [[buffer(6)]],
                      constant int &n_rows [[buffer(7)]],
                      constant int &subtract [[buffer(8)]],
                      uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(n_rows)) {
    return;
  }

  LhsI lhs_pos = lhs_indptr[row];
  const LhsI lhs_end = lhs_indptr[row + 1];
  RhsI rhs_pos = rhs_indptr[row];
  const RhsI rhs_end = rhs_indptr[row + 1];
  OutI count = OutI(0);

  while (lhs_pos < lhs_end || rhs_pos < rhs_end) {
    long col = 0;
    if (rhs_pos >= rhs_end ||
        (lhs_pos < lhs_end && lhs_indices[lhs_pos] < rhs_indices[rhs_pos])) {
      col = static_cast<long>(lhs_indices[lhs_pos]);
    } else if (lhs_pos >= lhs_end ||
               rhs_indices[rhs_pos] < lhs_indices[lhs_pos]) {
      col = static_cast<long>(rhs_indices[rhs_pos]);
    } else {
      col = static_cast<long>(lhs_indices[lhs_pos]);
    }

    typedef typename sparse_accumulator<T>::type acc_t;
    acc_t acc = sparse_accumulator<T>::zero();
    while (lhs_pos < lhs_end &&
           static_cast<long>(lhs_indices[lhs_pos]) == col) {
      acc += acc_t(lhs_data[lhs_pos]);
      ++lhs_pos;
    }
    while (rhs_pos < rhs_end &&
           static_cast<long>(rhs_indices[rhs_pos]) == col) {
      const acc_t value = acc_t(rhs_data[rhs_pos]);
      acc = subtract != 0 ? acc - value : acc + value;
      ++rhs_pos;
    }
    if (csr_add_nonzero<T>(sparse_accumulator<T>::cast(acc))) {
      count += OutI(1);
    }
  }

  counts[row] = count;
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
[[kernel]] void csr_add_fill_kernel(
    device const T *lhs_data [[buffer(0)]],
    device const LhsI *lhs_indices [[buffer(1)]],
    device const LhsI *lhs_indptr [[buffer(2)]],
    device const T *rhs_data [[buffer(3)]],
    device const RhsI *rhs_indices [[buffer(4)]],
    device const RhsI *rhs_indptr [[buffer(5)]],
    device const OutI *out_indptr [[buffer(6)]],
    device T *out_data [[buffer(7)]], device OutI *out_indices [[buffer(8)]],
    constant int &n_rows [[buffer(9)]], constant int &subtract [[buffer(10)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(n_rows)) {
    return;
  }

  LhsI lhs_pos = lhs_indptr[row];
  const LhsI lhs_end = lhs_indptr[row + 1];
  RhsI rhs_pos = rhs_indptr[row];
  const RhsI rhs_end = rhs_indptr[row + 1];
  OutI write = out_indptr[row];

  while (lhs_pos < lhs_end || rhs_pos < rhs_end) {
    long col = 0;
    if (rhs_pos >= rhs_end ||
        (lhs_pos < lhs_end && lhs_indices[lhs_pos] < rhs_indices[rhs_pos])) {
      col = static_cast<long>(lhs_indices[lhs_pos]);
    } else if (lhs_pos >= lhs_end ||
               rhs_indices[rhs_pos] < lhs_indices[lhs_pos]) {
      col = static_cast<long>(rhs_indices[rhs_pos]);
    } else {
      col = static_cast<long>(lhs_indices[lhs_pos]);
    }

    typedef typename sparse_accumulator<T>::type acc_t;
    acc_t acc = sparse_accumulator<T>::zero();
    while (lhs_pos < lhs_end &&
           static_cast<long>(lhs_indices[lhs_pos]) == col) {
      acc += acc_t(lhs_data[lhs_pos]);
      ++lhs_pos;
    }
    while (rhs_pos < rhs_end &&
           static_cast<long>(rhs_indices[rhs_pos]) == col) {
      const acc_t value = acc_t(rhs_data[rhs_pos]);
      acc = subtract != 0 ? acc - value : acc + value;
      ++rhs_pos;
    }

    const T value = sparse_accumulator<T>::cast(acc);
    if (csr_add_nonzero<T>(value)) {
      out_indices[write] = OutI(col);
      out_data[write] = value;
      ++write;
    }
  }
}

#define INSTANTIATE_CSR_ADD_COUNTS(VNAME, LNAME, RNAME, ONAME, T, LHSI, RHSI,  \
                                   OUTI)                                       \
  template [[host_name("csr_add_counts_" #VNAME "_" #LNAME "_" #RNAME          \
                       "_" #ONAME)]] [[kernel]] void                           \
  csr_add_counts_kernel<T, LHSI, RHSI, OUTI>(                                  \
      device const T *, device const LHSI *, device const LHSI *,              \
      device const T *, device const RHSI *, device const RHSI *,              \
      device OUTI *, constant int &, constant int &, uint)

#define INSTANTIATE_CSR_ADD_FILL(VNAME, LNAME, RNAME, ONAME, T, LHSI, RHSI,    \
                                 OUTI)                                         \
  template [[host_name("csr_add_fill_" #VNAME "_" #LNAME "_" #RNAME            \
                       "_" #ONAME)]] [[kernel]] void                           \
  csr_add_fill_kernel<T, LHSI, RHSI, OUTI>(                                    \
      device const T *, device const LHSI *, device const LHSI *,              \
      device const T *, device const RHSI *, device const RHSI *,              \
      device const OUTI *, device T *, device OUTI *, constant int &,          \
      constant int &, uint)

#define INSTANTIATE_CSR_ADD_VALUE(VNAME, T)                                    \
  INSTANTIATE_CSR_ADD_COUNTS(VNAME, int32, int32, int32, T, int, int, int);    \
  INSTANTIATE_CSR_ADD_COUNTS(VNAME, int32, int32, int64, T, int, int, long);   \
  INSTANTIATE_CSR_ADD_COUNTS(VNAME, int32, int64, int64, T, int, long, long);  \
  INSTANTIATE_CSR_ADD_COUNTS(VNAME, int64, int32, int64, T, long, int, long);  \
  INSTANTIATE_CSR_ADD_COUNTS(VNAME, int64, int64, int64, T, long, long, long); \
  INSTANTIATE_CSR_ADD_FILL(VNAME, int32, int32, int32, T, int, int, int);      \
  INSTANTIATE_CSR_ADD_FILL(VNAME, int32, int32, int64, T, int, int, long);     \
  INSTANTIATE_CSR_ADD_FILL(VNAME, int32, int64, int64, T, int, long, long);    \
  INSTANTIATE_CSR_ADD_FILL(VNAME, int64, int32, int64, T, long, int, long);    \
  INSTANTIATE_CSR_ADD_FILL(VNAME, int64, int64, int64, T, long, long, long)

INSTANTIATE_CSR_ADD_VALUE(float32, float);
INSTANTIATE_CSR_ADD_VALUE(float16, half);
INSTANTIATE_CSR_ADD_VALUE(bfloat16, bfloat16_t);
INSTANTIATE_CSR_ADD_VALUE(complex64, complex64_t);

#undef INSTANTIATE_CSR_ADD_VALUE
#undef INSTANTIATE_CSR_ADD_FILL
#undef INSTANTIATE_CSR_ADD_COUNTS
