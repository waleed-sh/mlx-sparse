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

#include <cstdint>
#include <tuple>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx_sparse {

namespace mx = mlx::core;

mx::Dtype random_index_dtype_from_bits(int index_dtype_bits);

void check_random_key(const mx::array &key);

void check_random_shape(int64_t n_rows, int64_t n_cols, int64_t nnz,
                        mx::Dtype index_dtype);

uint64_t keyed_seed(const uint32_t *key, int64_t n_rows, int64_t n_cols,
                    int64_t nnz);

uint64_t random_linear_index(uint64_t k, uint64_t total, uint64_t seed);

mx::array random_structural_keys(const mx::array &key, int64_t n_rows,
                                 int64_t n_cols, int64_t nnz, bool csc,
                                 mx::Stream stream);

mx::array random_compressed_counts(const mx::array &key, int64_t n_rows,
                                   int64_t n_cols, int64_t nnz, bool csc,
                                   mx::Stream stream);

mx::array random_compressed_unpack_sorted_keys(const mx::array &keys,
                                               int64_t minor_extent,
                                               mx::Dtype index_dtype, bool csc,
                                               mx::Stream stream);

mx::array random_compressed_indptr(const mx::array &counts,
                                   mx::Dtype index_dtype, mx::Stream stream);

} // namespace mlx_sparse
