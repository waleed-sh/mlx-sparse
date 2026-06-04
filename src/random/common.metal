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

#include "random/random_metal.h"

[[kernel]] void random_compressed_zero_counts(device int *counts [[buffer(0)]],
                                              constant long &segments
                                              [[buffer(1)]],
                                              uint index
                                              [[thread_position_in_grid]]) {
  if (index < static_cast<uint>(segments)) {
    counts[index] = 0;
  }
}

[[kernel]] void random_compressed_counts(
    device const uint *key [[buffer(0)]], device int *counts [[buffer(1)]],
    constant long &n_rows [[buffer(2)]], constant long &n_cols [[buffer(3)]],
    constant long &nnz [[buffer(4)]], constant int &csc [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= static_cast<uint>(nnz) || nnz == 0) {
    return;
  }

  const ulong seed = random_keyed_seed(key, n_rows, n_cols, nnz);
  const ulong total = ulong(n_rows) * ulong(n_cols);
  const ulong linear = random_linear_index(ulong(index), total, seed);
  const ulong major = csc ? linear % ulong(n_cols) : linear / ulong(n_cols);
  device atomic_int *atomic_counts =
      reinterpret_cast<device atomic_int *>(counts);
  atomic_fetch_add_explicit(&atomic_counts[major], 1, memory_order_relaxed);
}

[[kernel]] void random_structural_keys(device const uint *key [[buffer(0)]],
                                       device long *keys [[buffer(1)]],
                                       constant long &n_rows [[buffer(2)]],
                                       constant long &n_cols [[buffer(3)]],
                                       constant long &nnz [[buffer(4)]],
                                       constant int &csc [[buffer(5)]],
                                       uint index [[thread_position_in_grid]]) {
  if (index >= static_cast<uint>(nnz) || nnz == 0) {
    return;
  }

  const ulong seed = random_keyed_seed(key, n_rows, n_cols, nnz);
  const ulong total = ulong(n_rows) * ulong(n_cols);
  const ulong linear = random_linear_index(ulong(index), total, seed);
  const ulong row = linear / ulong(n_cols);
  const ulong col = linear % ulong(n_cols);
  keys[index] = long(csc ? col * ulong(n_rows) + row : linear);
}
