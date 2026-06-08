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
[[kernel]] void csr_tocoo_row_kernel(device const I *indptr [[buffer(0)]],
                                     device I *row [[buffer(1)]],
                                     constant int &n_rows [[buffer(2)]],
                                     uint r [[thread_position_in_grid]]) {
  if (r >= static_cast<uint>(n_rows)) {
    return;
  }
  for (I p = indptr[r]; p < indptr[r + 1]; ++p) {
    row[p] = static_cast<I>(r);
  }
}

template [[host_name("csr_tocoo_row_int32")]] [[kernel]] void
csr_tocoo_row_kernel<int>(device const int *, device int *, constant int &,
                          uint);
template [[host_name("csr_tocoo_row_int64")]] [[kernel]] void
csr_tocoo_row_kernel<long>(device const long *, device long *, constant int &,
                           uint);
