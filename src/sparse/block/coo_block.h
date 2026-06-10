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

#include <tuple>
#include <vector>

#include "mlx/array.h"
#include "mlx/utils.h"

namespace mlx_sparse {

namespace mx = mlx::core;

mx::array coo_block_data(const std::vector<mx::array> &data_arrays,
                         mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array>
coo_block_indices(const std::vector<mx::array> &row_arrays,
                  const std::vector<mx::array> &col_arrays,
                  const std::vector<int> &row_offsets,
                  const std::vector<int> &col_offsets, int n_rows, int n_cols,
                  mx::StreamOrDevice s = {});

std::tuple<mx::array, mx::array, mx::array>
coo_block(const std::vector<mx::array> &data_arrays,
          const std::vector<mx::array> &row_arrays,
          const std::vector<mx::array> &col_arrays,
          const std::vector<int> &row_offsets,
          const std::vector<int> &col_offsets, int n_rows, int n_cols,
          mx::StreamOrDevice s = {});

} // namespace mlx_sparse
