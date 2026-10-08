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
                                      constant int &n_rows [[buffer(6)]],
                                      constant int &n_cols [[buffer(7)]],
                                      uint tid [[thread_position_in_grid]]) {
  if (tid >= static_cast<uint>(nnz)) {
    return;
  }

  const bool keep = coo_entry_in_range(row[tid], col[tid], n_rows, n_cols);

  // Ranks are taken among the entries that survive, so that they agree with the
  // pointer array the companion kernel builds from the same predicate. Entries
  // that do not survive take a slot past the last one any column points at, in
  // input order, so every slot is written exactly once.
  int rank = 0;
  int kept_total = 0;
  int dropped_before = 0;
  for (int j = 0; j < nnz; ++j) {
    if (!coo_entry_in_range(row[j], col[j], n_rows, n_cols)) {
      if (j < static_cast<int>(tid)) {
        dropped_before += 1;
      }
      continue;
    }
    kept_total += 1;
    if (!keep) {
      continue;
    }
    const bool less = col[j] < col[tid] ||
                      (col[j] == col[tid] &&
                       (row[j] < row[tid] ||
                        (row[j] == row[tid] && j < static_cast<int>(tid))));
    if (less) {
      rank += 1;
    }
  }

  if (keep) {
    out_data[rank] = data[tid];
    out_indices[rank] = row[tid];
  } else {
    out_data[kept_total + dropped_before] = T(0);
    out_indices[kept_total + dropped_before] = I(0);
  }
}

template <typename I>
[[kernel]] void coo_tocsc_indptr_kernel(device const I *row [[buffer(0)]],
                                        device const I *col [[buffer(1)]],
                                        device I *out_indptr [[buffer(2)]],
                                        constant int &nnz [[buffer(3)]],
                                        constant int &n_rows [[buffer(4)]],
                                        constant int &n_cols [[buffer(5)]],
                                        uint tid [[thread_position_in_grid]]) {
  if (tid != 0) {
    return;
  }

  for (int c = 0; c <= n_cols; ++c) {
    out_indptr[c] = I(0);
  }
  for (int p = 0; p < nnz; ++p) {
    if (!coo_entry_in_range(row[p], col[p], n_rows, n_cols)) {
      continue;
    }
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
                              constant int &, constant int &, constant int &,  \
                              uint)

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
coo_tocsc_indptr_kernel<int>(device const int *, device const int *,
                             device int *, constant int &, constant int &,
                             constant int &, uint);
template [[host_name("coo_tocsc_indptr_int64")]] [[kernel]] void
coo_tocsc_indptr_kernel<long>(device const long *, device const long *,
                              device long *, constant int &, constant int &,
                              constant int &, uint);
