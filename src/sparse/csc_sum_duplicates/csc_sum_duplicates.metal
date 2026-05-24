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

template <typename I>
[[kernel]] void csc_sum_duplicates_counts_kernel(
    device const I *indices [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device I *counts [[buffer(2)]], constant int &n_cols [[buffer(3)]],
    uint col [[thread_position_in_grid]]) {
  if (col >= static_cast<uint>(n_cols)) {
    return;
  }

  I count = I(0);
  I previous = I(0);
  bool have_previous = false;
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    const I row = indices[p];
    if (!have_previous || row != previous) {
      count += I(1);
      previous = row;
      have_previous = true;
    }
  }
  counts[col] = count;
}

template <typename T, typename I>
[[kernel]] void csc_sum_duplicates_fill_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
    device const I *out_indptr [[buffer(3)]], device T *out_data [[buffer(4)]],
    device I *out_indices [[buffer(5)]], constant int &n_cols [[buffer(6)]],
    uint col [[thread_position_in_grid]]) {
  if (col >= static_cast<uint>(n_cols)) {
    return;
  }

  typedef typename sparse_accumulator<T>::type acc_t;
  I write = out_indptr[col];
  for (I p = indptr[col]; p < indptr[col + 1];) {
    const I row = indices[p];
    acc_t acc = sparse_accumulator<T>::zero();
    do {
      acc += acc_t(data[p]);
      ++p;
    } while (p < indptr[col + 1] && indices[p] == row);

    out_indices[write] = row;
    out_data[write] = sparse_accumulator<T>::cast(acc);
    ++write;
  }
}

template [[host_name("csc_sum_duplicates_counts_int32")]] [[kernel]] void
csc_sum_duplicates_counts_kernel<int>(device const int *, device const int *,
                                      device int *, constant int &, uint);
template [[host_name("csc_sum_duplicates_counts_int64")]] [[kernel]] void
csc_sum_duplicates_counts_kernel<long>(device const long *, device const long *,
                                       device long *, constant int &, uint);

#define INSTANTIATE_CSC_SUM_DUP_FILL(NAME, T, I)                               \
  template [[host_name("csc_sum_duplicates_fill_" #NAME)]] [[kernel]] void     \
  csc_sum_duplicates_fill_kernel<T, I>(                                        \
      device const T *, device const I *, device const I *, device const I *,  \
      device T *, device I *, constant int &, uint)

INSTANTIATE_CSC_SUM_DUP_FILL(float32_int32, float, int);
INSTANTIATE_CSC_SUM_DUP_FILL(float32_int64, float, long);
INSTANTIATE_CSC_SUM_DUP_FILL(float16_int32, half, int);
INSTANTIATE_CSC_SUM_DUP_FILL(float16_int64, half, long);
INSTANTIATE_CSC_SUM_DUP_FILL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_SUM_DUP_FILL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_SUM_DUP_FILL(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_SUM_DUP_FILL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_SUM_DUP_FILL
