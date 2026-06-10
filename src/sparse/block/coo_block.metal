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
[[kernel]] void coo_block_data_copy_kernel(
    device const T *src [[buffer(0)]], device T *dst [[buffer(1)]],
    constant int &nnz [[buffer(2)]], constant int &out_offset [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= static_cast<uint>(nnz)) {
    return;
  }
  dst[out_offset + static_cast<int>(tid)] = src[tid];
}

template <typename I>
[[kernel]] void coo_block_indices_offset_kernel(
    device const I *src_row [[buffer(0)]],
    device const I *src_col [[buffer(1)]], device I *dst_row [[buffer(2)]],
    device I *dst_col [[buffer(3)]], constant int &nnz [[buffer(4)]],
    constant int &out_offset [[buffer(5)]],
    constant int &row_offset [[buffer(6)]],
    constant int &col_offset [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= static_cast<uint>(nnz)) {
    return;
  }
  const int write = out_offset + static_cast<int>(tid);
  dst_row[write] = I(static_cast<long>(src_row[tid]) + row_offset);
  dst_col[write] = I(static_cast<long>(src_col[tid]) + col_offset);
}

#define INSTANTIATE_COO_BLOCK_DATA(NAME, T)                                    \
  template [[host_name("coo_block_data_copy_" #NAME)]] [[kernel]] void         \
  coo_block_data_copy_kernel<T>(device const T *, device T *, constant int &,  \
                                constant int &, uint)

INSTANTIATE_COO_BLOCK_DATA(float32, float);
INSTANTIATE_COO_BLOCK_DATA(float16, half);
INSTANTIATE_COO_BLOCK_DATA(bfloat16, bfloat16_t);
INSTANTIATE_COO_BLOCK_DATA(complex64, complex64_t);

#undef INSTANTIATE_COO_BLOCK_DATA

#define INSTANTIATE_COO_BLOCK_INDICES(NAME, I)                                 \
  template [[host_name("coo_block_indices_offset_" #NAME)]] [[kernel]] void    \
  coo_block_indices_offset_kernel<I>(                                          \
      device const I *, device const I *, device I *, device I *,              \
      constant int &, constant int &, constant int &, constant int &, uint)

INSTANTIATE_COO_BLOCK_INDICES(int32, int);
INSTANTIATE_COO_BLOCK_INDICES(int64, long);

#undef INSTANTIATE_COO_BLOCK_INDICES
