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

template <typename I>
inline int csr_source_row(device const I *indptr, int n_rows, I p) {
  for (int row = 0; row < n_rows; ++row) {
    if (p >= indptr[row] && p < indptr[row + 1]) {
      return row;
    }
  }
  return 0;
}

template <typename T, typename I>
[[kernel]] void csr_transpose_rank_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]], device T *out_data [[buffer(3)]],
    device I *out_indices [[buffer(4)]], constant int &nnz [[buffer(5)]],
    constant int &n_rows [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  if (tid >= static_cast<uint>(nnz)) {
    return;
  }

  const I p = static_cast<I>(tid);
  const int src_row = csr_source_row<I>(indptr, n_rows, p);
  const I dst_row = indices[p];

  int rank = 0;
  for (int j = 0; j < nnz; ++j) {
    const I q = static_cast<I>(j);
    const int other_src = csr_source_row<I>(indptr, n_rows, q);
    const I other_dst = indices[q];
    const bool less = other_dst < dst_row ||
                      (other_dst == dst_row &&
                       (other_src < src_row ||
                        (other_src == src_row && j < static_cast<int>(tid))));
    if (less) {
      rank += 1;
    }
  }

  out_data[rank] = data[p];
  out_indices[rank] = static_cast<I>(src_row);
}

template <typename I>
[[kernel]] void csr_transpose_indptr_kernel(
    device const I *indices [[buffer(0)]], device I *out_indptr [[buffer(1)]],
    constant int &nnz [[buffer(2)]], constant int &n_cols [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }

  for (int r = 0; r <= n_cols; ++r) {
    out_indptr[r] = I(0);
  }
  for (int p = 0; p < nnz; ++p) {
    out_indptr[static_cast<int>(indices[p]) + 1] += I(1);
  }
  for (int r = 0; r < n_cols; ++r) {
    out_indptr[r + 1] += out_indptr[r];
  }
}

#define INSTANTIATE_CSR_TRANSPOSE_RANK(NAME, T, I)                             \
  template [[host_name("csr_transpose_rank_" #NAME)]] [[kernel]] void          \
  csr_transpose_rank_kernel<T, I>(device const T *, device const I *,          \
                                  device const I *, device T *, device I *,    \
                                  constant int &, constant int &, uint)

INSTANTIATE_CSR_TRANSPOSE_RANK(float32_int32, float, int);
INSTANTIATE_CSR_TRANSPOSE_RANK(float32_int64, float, long);
INSTANTIATE_CSR_TRANSPOSE_RANK(float16_int32, half, int);
INSTANTIATE_CSR_TRANSPOSE_RANK(float16_int64, half, long);
INSTANTIATE_CSR_TRANSPOSE_RANK(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSR_TRANSPOSE_RANK(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSR_TRANSPOSE_RANK(complex64_int32, complex64_t, int);
INSTANTIATE_CSR_TRANSPOSE_RANK(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSR_TRANSPOSE_RANK

template [[host_name("csr_transpose_indptr_int32")]] [[kernel]] void
csr_transpose_indptr_kernel<int>(device const int *, device int *,
                                 constant int &, constant int &, uint);
template [[host_name("csr_transpose_indptr_int64")]] [[kernel]] void
csr_transpose_indptr_kernel<long>(device const long *, device long *,
                                  constant int &, constant int &, uint);
