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

template <typename T, typename I>
[[kernel]] void coo_tocsc_rank_kernel(device const T *data [[buffer(0)]],
                                      device const I *row [[buffer(1)]],
                                      device const I *col [[buffer(2)]],
                                      device T *out_data [[buffer(3)]],
                                      device I *out_indices [[buffer(4)]],
                                      constant int &nnz [[buffer(5)]],
                                      uint tid [[thread_position_in_grid]]) {
  if (tid >= static_cast<uint>(nnz)) {
    return;
  }

  int rank = 0;
  for (int j = 0; j < nnz; ++j) {
    const bool less = col[j] < col[tid] ||
                      (col[j] == col[tid] &&
                       (row[j] < row[tid] ||
                        (row[j] == row[tid] && j < static_cast<int>(tid))));
    if (less) {
      rank += 1;
    }
  }

  out_data[rank] = data[tid];
  out_indices[rank] = row[tid];
}

template <typename I>
[[kernel]] void coo_tocsc_indptr_kernel(device const I *col [[buffer(0)]],
                                        device I *out_indptr [[buffer(1)]],
                                        constant int &nnz [[buffer(2)]],
                                        constant int &n_cols [[buffer(3)]],
                                        uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }

  for (int c = 0; c <= n_cols; ++c) {
    out_indptr[c] = I(0);
  }
  for (int p = 0; p < nnz; ++p) {
    out_indptr[static_cast<int>(col[p]) + 1] += I(1);
  }
  for (int c = 0; c < n_cols; ++c) {
    out_indptr[c + 1] += out_indptr[c];
  }
}

#define INSTANTIATE_COO_TOCSC_RANK(NAME, T, I)                                 \
  template [[host_name("coo_tocsc_rank_" #NAME)]] [[kernel]] void              \
  coo_tocsc_rank_kernel<T, I>(device const T *, device const I *,              \
                              device const I *, device T *, device I *,        \
                              constant int &, uint)

INSTANTIATE_COO_TOCSC_RANK(float32_int32, float, int);
INSTANTIATE_COO_TOCSC_RANK(float32_int64, float, long);
INSTANTIATE_COO_TOCSC_RANK(float16_int32, half, int);
INSTANTIATE_COO_TOCSC_RANK(float16_int64, half, long);
INSTANTIATE_COO_TOCSC_RANK(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_COO_TOCSC_RANK(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_COO_TOCSC_RANK(complex64_int32, complex64_t, int);
INSTANTIATE_COO_TOCSC_RANK(complex64_int64, complex64_t, long);

#undef INSTANTIATE_COO_TOCSC_RANK

template [[host_name("coo_tocsc_indptr_int32")]] [[kernel]] void
coo_tocsc_indptr_kernel<int>(device const int *, device int *, constant int &,
                             constant int &, uint);
template [[host_name("coo_tocsc_indptr_int64")]] [[kernel]] void
coo_tocsc_indptr_kernel<long>(device const long *, device long *,
                              constant int &, constant int &, uint);
