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

#pragma once

#include <metal_stdlib>

using namespace metal;

inline ulong random_splitmix64(ulong x) {
  x += 0x9E3779B97F4A7C15ul;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ul;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBul;
  return x ^ (x >> 31);
}

inline ulong random_keyed_seed(device const uint *key, long n_rows, long n_cols,
                               long nnz) {
  ulong seed = (ulong(key[0]) << 32) | ulong(key[1]);
  seed ^= random_splitmix64(ulong(n_rows));
  seed ^= random_splitmix64(ulong(n_cols) + 0xD1B54A32D192ED03ul);
  seed ^= random_splitmix64(ulong(nnz) + 0xABC98388FB8FAC03ul);
  return seed;
}

inline int random_ceil_log2(ulong value) {
  if (value <= 1ul) {
    return 1;
  }
  int bits = 0;
  ulong current = value - 1ul;
  while (current != 0ul) {
    current >>= 1;
    bits += 1;
  }
  return bits;
}

inline ulong random_bit_mask(int bits) {
  if (bits >= 64) {
    return 0xfffffffffffffffful;
  }
  return (1ul << bits) - 1ul;
}

inline ulong random_permute_power2(ulong x, int bits, ulong seed) {
  const ulong mask = random_bit_mask(bits);
  x &= mask;
  x ^= seed & mask;
  x ^= x >> 33;
  x = (x * 0xff51afd7ed558ccdul) & mask;
  x ^= x >> 29;
  x = (x * 0xc4ceb9fe1a85ec53ul) & mask;
  x ^= x >> 32;
  return x & mask;
}

inline ulong random_linear_index(ulong k, ulong total, ulong seed) {
  const int bits = random_ceil_log2(total);
  ulong x = k;
  do {
    x = random_permute_power2(x, bits, seed);
  } while (x >= total);
  return x;
}
