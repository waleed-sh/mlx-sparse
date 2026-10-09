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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mlx_sparse {

struct CooperativeGrid {
  size_t groups_x;
  size_t groups_y;
};

inline bool needs_wide_dense_indices(size_t outputs, size_t rhs_elements) {
  return std::max(outputs, rhs_elements) > std::numeric_limits<uint32_t>::max();
}

// Keep each grid dimension representable by Metal's uint coordinates.
// Balance the last plane so padding is smaller than the number of planes.
inline CooperativeGrid cooperative_grid(size_t outputs) {
  constexpr size_t max_groups_x = 65536;
  outputs = std::max<size_t>(outputs, 1);
  const size_t groups_y = (outputs - 1) / max_groups_x + 1;
  if (groups_y > std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error(
        "Cooperative Metal grid exceeds the supported dimensions.");
  }
  return {(outputs - 1) / groups_y + 1, groups_y};
}

} // namespace mlx_sparse
