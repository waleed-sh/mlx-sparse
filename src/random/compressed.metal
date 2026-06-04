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

template <typename I>
[[kernel]] void random_compressed_unpack_sorted_keys_kernel(
    device const long *keys [[buffer(0)]], device I *indices [[buffer(1)]],
    constant long &minor_extent [[buffer(2)]], constant long &nnz [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= static_cast<uint>(nnz)) {
    return;
  }

  indices[index] = I(ulong(keys[index]) % ulong(minor_extent));
}

#define INSTANTIATE_RANDOM_COMPRESSED_UNPACK(NAME, I)                          \
  template [[host_name(                                                        \
      "random_compressed_unpack_sorted_keys_" #NAME)]] [[kernel]] void         \
  random_compressed_unpack_sorted_keys_kernel<I>(                              \
      device const long *, device I *, constant long &, constant long &, uint)

INSTANTIATE_RANDOM_COMPRESSED_UNPACK(int32, int);
INSTANTIATE_RANDOM_COMPRESSED_UNPACK(int64, long);

#undef INSTANTIATE_RANDOM_COMPRESSED_UNPACK
