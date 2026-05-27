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
#include <memory>
#include <string>

#include "linalg/accelerate/factorization/factorization.h"
#include "mlx/array.h"

#ifndef MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
#define MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK 0
#endif

namespace mlx_sparse {

namespace mx = mlx::core;

bool accelerate_sparse_solve_available();
bool accelerate_sparse_lu_solve_available();

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK

class AccelerateFloatSolve final {
public:
  AccelerateFloatSolve(AccelerateFloatFactorization factorization,
                       std::string method);

  const std::string &method() const noexcept { return method_; }
  int row_count() const noexcept { return factorization_.row_count(); }
  int column_count() const noexcept { return factorization_.column_count(); }
  int rhs_size() const { return factorization_.rhs_size(); }
  int solution_size() const { return factorization_.solution_size(); }

  mx::array solve(const mx::array &rhs) const;

private:
  AccelerateFloatFactorization factorization_;
  std::string method_;
};

std::unique_ptr<AccelerateFloatSolve> make_accelerate_float_solve_from_csc(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    std::int64_t n_rows, std::int64_t n_cols, const std::string &method);

std::unique_ptr<AccelerateFloatSolve> make_accelerate_float_solve_from_csr(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    std::int64_t n_rows, std::int64_t n_cols, const std::string &method);

std::unique_ptr<AccelerateFloatSolve> make_accelerate_float_solve_from_coo(
    const mx::array &data, const mx::array &row, const mx::array &col,
    std::int64_t n_rows, std::int64_t n_cols, const std::string &method);

#endif

} // namespace mlx_sparse
