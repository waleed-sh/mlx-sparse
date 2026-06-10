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

inline bool keep_triangular(long row, long col, int k, bool upper) {
  const long diagonal = col - row;
  return upper ? diagonal >= static_cast<long>(k)
               : diagonal <= static_cast<long>(k);
}

template <typename I>
[[kernel]] void coo_triangular_counts_kernel(
    device const I *row [[buffer(0)]], device const I *col [[buffer(1)]],
    device I *counts [[buffer(2)]], constant int &nnz [[buffer(3)]],
    constant int &k [[buffer(4)]], constant int &upper_i [[buffer(5)]],
    uint p [[thread_position_in_grid]]) {
  if (p >= static_cast<uint>(nnz)) {
    return;
  }
  counts[p] = keep_triangular(static_cast<long>(row[p]),
                              static_cast<long>(col[p]), k, upper_i != 0)
                  ? I(1)
                  : I(0);
}

template <typename T, typename I>
[[kernel]] void coo_triangular_fill_kernel(
    device const T *data [[buffer(0)]], device const I *row [[buffer(1)]],
    device const I *col [[buffer(2)]], device const I *offsets [[buffer(3)]],
    device T *out_data [[buffer(4)]], device I *out_row [[buffer(5)]],
    device I *out_col [[buffer(6)]], constant int &nnz [[buffer(7)]],
    uint p [[thread_position_in_grid]]) {
  if (p >= static_cast<uint>(nnz)) {
    return;
  }
  const I write = offsets[p];
  if (offsets[p + 1] == write) {
    return;
  }
  out_data[write] = data[p];
  out_row[write] = row[p];
  out_col[write] = col[p];
}

template <typename I>
[[kernel]] void csr_triangular_counts_kernel(
    device const I *indices [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device I *counts [[buffer(2)]], constant int &n_rows [[buffer(3)]],
    constant int &k [[buffer(4)]], constant int &upper_i [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(n_rows)) {
    return;
  }
  I count = I(0);
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    if (keep_triangular(static_cast<long>(row), static_cast<long>(indices[p]),
                        k, upper_i != 0)) {
      ++count;
    }
  }
  counts[row] = count;
}

template <typename T, typename I>
[[kernel]] void csr_triangular_fill_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
    device const I *out_indptr [[buffer(3)]], device T *out_data [[buffer(4)]],
    device I *out_indices [[buffer(5)]], constant int &n_rows [[buffer(6)]],
    constant int &k [[buffer(7)]], constant int &upper_i [[buffer(8)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(n_rows)) {
    return;
  }
  I write = out_indptr[row];
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    const I col = indices[p];
    if (keep_triangular(static_cast<long>(row), static_cast<long>(col), k,
                        upper_i != 0)) {
      out_data[write] = data[p];
      out_indices[write] = col;
      ++write;
    }
  }
}

template <typename I>
[[kernel]] void csc_triangular_counts_kernel(
    device const I *indices [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device I *counts [[buffer(2)]], constant int &n_cols [[buffer(3)]],
    constant int &k [[buffer(4)]], constant int &upper_i [[buffer(5)]],
    uint col [[thread_position_in_grid]]) {
  if (col >= static_cast<uint>(n_cols)) {
    return;
  }
  I count = I(0);
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    if (keep_triangular(static_cast<long>(indices[p]), static_cast<long>(col),
                        k, upper_i != 0)) {
      ++count;
    }
  }
  counts[col] = count;
}

template <typename T, typename I>
[[kernel]] void csc_triangular_fill_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
    device const I *out_indptr [[buffer(3)]], device T *out_data [[buffer(4)]],
    device I *out_indices [[buffer(5)]], constant int &n_cols [[buffer(6)]],
    constant int &k [[buffer(7)]], constant int &upper_i [[buffer(8)]],
    uint col [[thread_position_in_grid]]) {
  if (col >= static_cast<uint>(n_cols)) {
    return;
  }
  I write = out_indptr[col];
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    const I row = indices[p];
    if (keep_triangular(static_cast<long>(row), static_cast<long>(col), k,
                        upper_i != 0)) {
      out_data[write] = data[p];
      out_indices[write] = row;
      ++write;
    }
  }
}

#define INSTANTIATE_TRI_COUNTS(NAME, I)                                        \
  template [[host_name("coo_triangular_counts_" #NAME)]] [[kernel]] void       \
  coo_triangular_counts_kernel<I>(device const I *, device const I *,          \
                                  device I *, constant int &, constant int &,  \
                                  constant int &, uint);                       \
  template [[host_name("csr_triangular_counts_" #NAME)]] [[kernel]] void       \
  csr_triangular_counts_kernel<I>(device const I *, device const I *,          \
                                  device I *, constant int &, constant int &,  \
                                  constant int &, uint);                       \
  template [[host_name("csc_triangular_counts_" #NAME)]] [[kernel]] void       \
  csc_triangular_counts_kernel<I>(device const I *, device const I *,          \
                                  device I *, constant int &, constant int &,  \
                                  constant int &, uint)

INSTANTIATE_TRI_COUNTS(int32, int);
INSTANTIATE_TRI_COUNTS(int64, long);

#undef INSTANTIATE_TRI_COUNTS

#define INSTANTIATE_TRI_FILL(VNAME, INAME, T, I)                               \
  template                                                                     \
      [[host_name("coo_triangular_fill_" #VNAME "_" #INAME)]] [[kernel]] void  \
      coo_triangular_fill_kernel<T, I>(device const T *, device const I *,     \
                                       device const I *, device const I *,     \
                                       device T *, device I *, device I *,     \
                                       constant int &, uint);                  \
  template                                                                     \
      [[host_name("csr_triangular_fill_" #VNAME "_" #INAME)]] [[kernel]] void  \
      csr_triangular_fill_kernel<T, I>(device const T *, device const I *,     \
                                       device const I *, device const I *,     \
                                       device T *, device I *, constant int &, \
                                       constant int &, constant int &, uint);  \
  template                                                                     \
      [[host_name("csc_triangular_fill_" #VNAME "_" #INAME)]] [[kernel]] void  \
      csc_triangular_fill_kernel<T, I>(device const T *, device const I *,     \
                                       device const I *, device const I *,     \
                                       device T *, device I *, constant int &, \
                                       constant int &, constant int &, uint)

INSTANTIATE_TRI_FILL(float32, int32, float, int);
INSTANTIATE_TRI_FILL(float32, int64, float, long);
INSTANTIATE_TRI_FILL(float16, int32, half, int);
INSTANTIATE_TRI_FILL(float16, int64, half, long);
INSTANTIATE_TRI_FILL(bfloat16, int32, bfloat16_t, int);
INSTANTIATE_TRI_FILL(bfloat16, int64, bfloat16_t, long);
INSTANTIATE_TRI_FILL(complex64, int32, complex64_t, int);
INSTANTIATE_TRI_FILL(complex64, int64, complex64_t, long);

#undef INSTANTIATE_TRI_FILL
