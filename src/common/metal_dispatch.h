// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.

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
