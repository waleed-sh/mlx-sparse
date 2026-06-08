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

#include <metal_stdlib>

using namespace metal;

template <typename I>
[[kernel]] void csc_tocoo_col_kernel(device const I *indptr [[buffer(0)]],
                                     device I *col [[buffer(1)]],
                                     constant int &n_cols [[buffer(2)]],
                                     uint column [[thread_position_in_grid]]) {
  if (column >= static_cast<uint>(n_cols)) {
    return;
  }
  const I begin = indptr[column];
  const I end = indptr[column + 1];
  for (I p = begin; p < end; ++p) {
    col[p] = static_cast<I>(column);
  }
}

template [[host_name("csc_tocoo_col_int32")]] [[kernel]]
void csc_tocoo_col_kernel<int>(device const int *indptr [[buffer(0)]],
                               device int *col [[buffer(1)]],
                               constant int &n_cols [[buffer(2)]],
                               uint column [[thread_position_in_grid]]);

template [[host_name("csc_tocoo_col_int64")]] [[kernel]]
void csc_tocoo_col_kernel<long>(device const long *indptr [[buffer(0)]],
                                device long *col [[buffer(1)]],
                                constant int &n_cols [[buffer(2)]],
                                uint column [[thread_position_in_grid]]);
