// Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx_sparse {

namespace mx = mlx::core;

mx::array csc_batched_matvec(const mx::array &data, const mx::array &indices,
                             const mx::array &indptr, const mx::array &rhs,
                             int n_rows, int n_cols, mx::StreamOrDevice s = {});

mx::array csc_batched_matmul(const mx::array &data, const mx::array &indices,
                             const mx::array &indptr, const mx::array &rhs,
                             int n_rows, int n_cols, mx::StreamOrDevice s = {});

} // namespace mlx_sparse
