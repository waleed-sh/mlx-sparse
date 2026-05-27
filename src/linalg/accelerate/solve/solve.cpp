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

#include "linalg/accelerate/solve/solve.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/common.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

namespace mlx_sparse {

bool accelerate_sparse_solve_available() {
#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
  return true;
#else
  return false;
#endif
}

bool accelerate_sparse_lu_solve_available() {
#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
  return accelerate_lu_factorization_available();
#else
  return false;
#endif
}

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&                                 \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 150500
#define MLX_SPARSE_ACCELERATE_SOLVE_HAS_LU_FACTORIZATION_TYPES 1
#else
#define MLX_SPARSE_ACCELERATE_SOLVE_HAS_LU_FACTORIZATION_TYPES 0
#endif

namespace {

SparseFactorization_t parse_factorization_method(const std::string &method) {
  if (method == "cholesky") {
    return SparseFactorizationCholesky;
  }
  if (method == "ldlt") {
    return SparseFactorizationLDLT;
  }
  if (method == "qr") {
    return SparseFactorizationQR;
  }
  if (method == "cholesky_ata") {
    return SparseFactorizationCholeskyAtA;
  }
  if (method == "lu") {
#if MLX_SPARSE_ACCELERATE_SOLVE_HAS_LU_FACTORIZATION_TYPES
    if (!accelerate_lu_factorization_available()) {
      throw std::runtime_error(
          "Accelerate LU factorization requires macOS 15.5 or newer.");
    }
    return SparseFactorizationLU;
#else
    throw std::runtime_error(
        "Accelerate LU factorization requires a macOS 15.5 SDK and runtime.");
#endif
  }
  throw std::invalid_argument(
      "Accelerate factorized solve method must be 'cholesky', 'ldlt', "
      "'qr', 'cholesky_ata', or 'lu'.");
}

bool requires_square_matrix(SparseFactorization_t type) {
  switch (type) {
  case SparseFactorizationQR:
  case SparseFactorizationCholeskyAtA:
    return false;
  default:
    return true;
  }
}

AccelerateCscAdapterOptions adapter_options_for(SparseFactorization_t type) {
  AccelerateCscAdapterOptions options;
  options.require_square = requires_square_matrix(type);
  options.require_non_empty = true;
  options.canonicalize = true;
  return options;
}

std::vector<float> copy_float_values(mx::array array) {
  auto stream = mx::default_stream(mx::default_device());
  array = mx::contiguous(array, false, stream);
  array.eval();
  if (array.size() == 0) {
    return {};
  }
  const auto *values = array.data<float>();
  return std::vector<float>(values, values + array.size());
}

std::vector<float>
row_major_to_column_major(const std::vector<float> &row_major, int row_count,
                          int column_count) {
  std::vector<float> column_major(row_major.size());
  for (int row = 0; row < row_count; ++row) {
    for (int col = 0; col < column_count; ++col) {
      column_major[static_cast<std::size_t>(col) *
                       static_cast<std::size_t>(row_count) +
                   static_cast<std::size_t>(row)] =
          row_major[static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(column_count) +
                    static_cast<std::size_t>(col)];
    }
  }
  return column_major;
}

std::vector<float>
column_major_to_row_major(const std::vector<float> &column_major, int row_count,
                          int column_count) {
  std::vector<float> row_major(column_major.size());
  for (int row = 0; row < row_count; ++row) {
    for (int col = 0; col < column_count; ++col) {
      row_major[static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(column_count) +
                static_cast<std::size_t>(col)] =
          column_major[static_cast<std::size_t>(col) *
                           static_cast<std::size_t>(row_count) +
                       static_cast<std::size_t>(row)];
    }
  }
  return row_major;
}

std::unique_ptr<AccelerateFloatSolve>
make_solve(const AccelerateCscMatrixFloat &matrix, const std::string &method) {
  const auto type = parse_factorization_method(method);
  auto factorization = make_accelerate_float_factorization(
      type, matrix, "accelerate factorized solve");
  return std::make_unique<AccelerateFloatSolve>(std::move(factorization),
                                                method);
}

} // namespace

AccelerateFloatSolve::AccelerateFloatSolve(
    AccelerateFloatFactorization factorization, std::string method)
    : factorization_(std::move(factorization)), method_(std::move(method)) {}

mx::array AccelerateFloatSolve::solve(const mx::array &rhs) const {
  require_float32(rhs, "Accelerate factorized solve rhs");
  if (rhs.ndim() == 1) {
    if (rhs.shape(0) != rhs_size()) {
      throw std::invalid_argument(
          "Accelerate factorized solve rhs has incompatible shape: expected "
          "shape (" +
          std::to_string(rhs_size()) + ",), got (" +
          std::to_string(rhs.shape(0)) + ",).");
    }
    const auto rhs_values = copy_float_values(rhs);
    const auto solution =
        factorization_.solve_vector(rhs_values, "accelerate factorized solve");
    return mx::array(solution.begin(), mx::Shape{solution_size()}, mx::float32);
  }
  if (rhs.ndim() != 2) {
    throw std::invalid_argument(
        "Accelerate factorized solve rhs must be rank-1 or rank-2.");
  }
  if (rhs.shape(0) != rhs_size()) {
    throw std::invalid_argument(
        "Accelerate factorized solve rhs has incompatible shape: expected "
        "first dimension " +
        std::to_string(rhs_size()) + ", got " + std::to_string(rhs.shape(0)) +
        ".");
  }
  const int rhs_count = rhs.shape(1);
  if (rhs_count <= 0) {
    throw std::invalid_argument(
        "Accelerate factorized solve requires at least one right-hand side.");
  }

  const auto rhs_row_major = copy_float_values(rhs);
  const auto rhs_column_major =
      row_major_to_column_major(rhs_row_major, rhs_size(), rhs_count);
  const auto solution_column_major = factorization_.solve_matrix_column_major(
      rhs_column_major, rhs_count, "accelerate factorized solve");
  const auto solution_row_major = column_major_to_row_major(
      solution_column_major, solution_size(), rhs_count);
  return mx::array(solution_row_major.begin(),
                   mx::Shape{solution_size(), rhs_count}, mx::float32);
}

std::unique_ptr<AccelerateFloatSolve> make_accelerate_float_solve_from_csc(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    std::int64_t n_rows, std::int64_t n_cols, const std::string &method) {
  const auto type = parse_factorization_method(method);
  auto matrix = make_accelerate_csc_matrix_float32(
      data, indices, indptr, n_rows, n_cols, adapter_options_for(type),
      "accelerate CSC factorized solve adapter");
  return make_solve(matrix, method);
}

std::unique_ptr<AccelerateFloatSolve> make_accelerate_float_solve_from_csr(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    std::int64_t n_rows, std::int64_t n_cols, const std::string &method) {
  const auto type = parse_factorization_method(method);
  auto matrix = make_accelerate_csc_matrix_float32_from_csr(
      data, indices, indptr, n_rows, n_cols, adapter_options_for(type),
      "accelerate CSR factorized solve adapter");
  return make_solve(matrix, method);
}

std::unique_ptr<AccelerateFloatSolve> make_accelerate_float_solve_from_coo(
    const mx::array &data, const mx::array &row, const mx::array &col,
    std::int64_t n_rows, std::int64_t n_cols, const std::string &method) {
  const auto type = parse_factorization_method(method);
  auto matrix = make_accelerate_csc_matrix_float32_from_coo(
      data, row, col, n_rows, n_cols, adapter_options_for(type),
      "accelerate COO factorized solve adapter");
  return make_solve(matrix, method);
}

#endif

} // namespace mlx_sparse
