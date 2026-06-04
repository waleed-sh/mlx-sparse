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

#include "random/common.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "common/common.h"

namespace mlx_sparse {

namespace {

uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

int ceil_log2_u64(uint64_t value) {
  if (value <= 1) {
    return 1;
  }
  int bits = 0;
  uint64_t current = value - 1;
  while (current != 0) {
    current >>= 1;
    ++bits;
  }
  return bits;
}

uint64_t bit_mask(int bits) {
  if (bits >= 64) {
    return std::numeric_limits<uint64_t>::max();
  }
  return (uint64_t{1} << bits) - uint64_t{1};
}

uint64_t permute_power2(uint64_t x, int bits, uint64_t seed) {
  const uint64_t mask = bit_mask(bits);
  x &= mask;
  x ^= seed & mask;
  x ^= x >> 33;
  x = (x * 0xff51afd7ed558ccdull) & mask;
  x ^= x >> 29;
  x = (x * 0xc4ceb9fe1a85ec53ull) & mask;
  x ^= x >> 32;
  return x & mask;
}

} // namespace

mx::Dtype random_index_dtype_from_bits(int index_dtype_bits) {
  if (index_dtype_bits == 32) {
    return mx::int32;
  }
  if (index_dtype_bits == 64) {
    return mx::int64;
  }
  throw std::invalid_argument(
      "random_coo_indices index dtype must be encoded as 32 or 64.");
}

void check_random_key(const mx::array &key) {
  require_rank(key, 1, "random_coo_indices key");
  require_size(key, 2, "random_coo_indices key");
  if (key.dtype() != mx::uint32) {
    throw std::invalid_argument(
        "random_coo_indices key must have dtype uint32.");
  }
}

void check_random_shape(int64_t n_rows, int64_t n_cols, int64_t nnz,
                        mx::Dtype index_dtype) {
  if (n_rows < 0 || n_cols < 0 || nnz < 0) {
    throw std::invalid_argument(
        "random_coo_indices shape and nnz must be non-negative.");
  }
  if (n_rows > std::numeric_limits<int>::max() ||
      n_cols > std::numeric_limits<int>::max() ||
      nnz > std::numeric_limits<int>::max()) {
    throw std::overflow_error(
        "random_coo_indices shape and nnz must fit MLX shape limits.");
  }
  if (n_rows != 0 && n_cols > std::numeric_limits<int64_t>::max() / n_rows) {
    throw std::overflow_error("random_coo_indices m * n overflows int64.");
  }
  const int64_t total = n_rows * n_cols;
  if (nnz > total) {
    throw std::overflow_error("random_coo_indices nnz exceeds m * n.");
  }
  const auto max_index = index_dtype == mx::int32
                             ? int64_t{std::numeric_limits<int32_t>::max()}
                             : std::numeric_limits<int64_t>::max();
  if ((n_rows > 0 && n_rows - 1 > max_index) ||
      (n_cols > 0 && n_cols - 1 > max_index) || nnz > max_index) {
    throw std::overflow_error(
        "random_coo_indices output exceeds index dtype capacity.");
  }
}

uint64_t keyed_seed(const uint32_t *key, int64_t n_rows, int64_t n_cols,
                    int64_t nnz) {
  auto seed =
      (static_cast<uint64_t>(key[0]) << 32) | static_cast<uint64_t>(key[1]);
  seed ^= splitmix64(static_cast<uint64_t>(n_rows));
  seed ^= splitmix64(static_cast<uint64_t>(n_cols) + 0xD1B54A32D192ED03ull);
  seed ^= splitmix64(static_cast<uint64_t>(nnz) + 0xABC98388FB8FAC03ull);
  return seed;
}

uint64_t random_linear_index(uint64_t k, uint64_t total, uint64_t seed) {
  const int bits = ceil_log2_u64(total);
  uint64_t x = k;
  do {
    x = permute_power2(x, bits, seed);
  } while (x >= total);
  return x;
}

} // namespace mlx_sparse
