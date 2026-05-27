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

#include "linalg/accelerate/accelerate_csc_adapter.h"

#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/common.h"

namespace mlx_sparse {
namespace {

using ColumnAccumulator = std::map<int, float>;

std::string operation_name(std::string_view operation) {
  if (operation.empty()) {
    return "accelerate_csc_adapter";
  }
  return std::string(operation);
}

std::string named(std::string_view operation, const char *suffix) {
  std::string result = operation_name(operation);
  result += ' ';
  result += suffix;
  return result;
}

void require_rank_named(const mx::array &array, int ndim,
                        const std::string &name) {
  require_rank(array, ndim, name.c_str());
}

void require_float32_named(const mx::array &array, const std::string &name) {
  require_float32(array, name.c_str());
}

void require_index_dtype_named(const mx::array &array,
                               const std::string &name) {
  require_index_dtype(array, name.c_str());
}

void require_same_index_dtype_named(const mx::array &lhs, const mx::array &rhs,
                                    const std::string &lhs_name,
                                    const std::string &rhs_name) {
  require_same_index_dtype(lhs, rhs, lhs_name.c_str(), rhs_name.c_str());
}

int checked_dimension(std::int64_t dimension, const char *name,
                      const std::string &operation) {
  if (dimension < 0) {
    throw std::invalid_argument(operation +
                                " shape dimensions must be non-negative.");
  }
  if (dimension > std::numeric_limits<int>::max()) {
    throw std::overflow_error(operation + " " + name +
                              " exceeds Accelerate's int dimension range.");
  }
  return static_cast<int>(dimension);
}

std::pair<int, int> validate_shape(std::int64_t n_rows, std::int64_t n_cols,
                                   AccelerateCscAdapterOptions options,
                                   const std::string &operation) {
  const int row_count = checked_dimension(n_rows, "n_rows", operation);
  const int column_count = checked_dimension(n_cols, "n_cols", operation);
  if (options.require_non_empty && (row_count == 0 || column_count == 0)) {
    throw std::invalid_argument(operation +
                                " shape must be non-empty for Accelerate.");
  }
  if (options.require_square && row_count != column_count) {
    throw std::invalid_argument(
        operation + " shape must be square for this Accelerate operation.");
  }
  return {row_count, column_count};
}

template <typename T> std::vector<T> copy_array(mx::array array) {
  array.eval();
  if (array.size() == 0) {
    return {};
  }
  const auto *ptr = array.data<T>();
  return std::vector<T>(ptr, ptr + array.size());
}

std::vector<float> copy_values(const mx::array &data) {
  if (data.size() >
      static_cast<std::size_t>(std::numeric_limits<long>::max())) {
    throw std::overflow_error(
        "accelerate adapter data length exceeds Accelerate's long range.");
  }
  return copy_array<float>(data);
}

long checked_column_start(std::size_t value, const std::string &operation) {
  if (value > static_cast<std::size_t>(std::numeric_limits<long>::max())) {
    throw std::overflow_error(
        operation + " nnz exceeds Accelerate's long column-start range.");
  }
  return static_cast<long>(value);
}

template <typename I>
long checked_indptr_value(I value, std::size_t nnz,
                          const std::string &operation, std::size_t position) {
  if (value < 0) {
    std::ostringstream msg;
    msg << operation << " indptr[" << position << "] must be non-negative.";
    throw std::invalid_argument(msg.str());
  }
  if (static_cast<unsigned long long>(value) >
      static_cast<unsigned long long>(std::numeric_limits<long>::max())) {
    std::ostringstream msg;
    msg << operation << " indptr[" << position
        << "] exceeds Accelerate's long column-start range.";
    throw std::overflow_error(msg.str());
  }
  if (static_cast<std::size_t>(value) > nnz) {
    std::ostringstream msg;
    msg << operation << " indptr[" << position << "] exceeds nnz " << nnz
        << ".";
    throw std::invalid_argument(msg.str());
  }
  return static_cast<long>(value);
}

template <typename I>
std::vector<long>
validate_compressed_pointers(const std::vector<I> &indptr,
                             std::size_t outer_size, std::size_t nnz,
                             const char *name, const std::string &operation) {
  const std::size_t expected_size = outer_size + 1;
  if (indptr.size() != expected_size) {
    std::ostringstream msg;
    msg << operation << ' ' << name << " must have size " << expected_size
        << ", got " << indptr.size() << ".";
    throw std::invalid_argument(msg.str());
  }

  std::vector<long> out(indptr.size());
  long previous = 0;
  for (std::size_t i = 0; i < indptr.size(); ++i) {
    const long value = checked_indptr_value(indptr[i], nnz, operation, i);
    if (i == 0 && value != 0) {
      throw std::invalid_argument(operation + " indptr[0] must be 0.");
    }
    if (i > 0 && value < previous) {
      throw std::invalid_argument(
          operation + " indptr must be monotonically non-decreasing.");
    }
    out[i] = value;
    previous = value;
  }

  if (out.back() != static_cast<long>(nnz)) {
    std::ostringstream msg;
    msg << operation << " indptr[-1] must equal nnz " << nnz << ", got "
        << out.back() << ".";
    throw std::invalid_argument(msg.str());
  }
  return out;
}

template <typename I>
int checked_row_index(I raw, int row_count, const std::string &operation,
                      std::size_t position) {
  if (raw < 0) {
    std::ostringstream msg;
    msg << operation << " row index at position " << position
        << " must be non-negative.";
    throw std::invalid_argument(msg.str());
  }
  if (static_cast<unsigned long long>(raw) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    std::ostringstream msg;
    msg << operation << " row index at position " << position
        << " exceeds Accelerate's int index range.";
    throw std::overflow_error(msg.str());
  }
  const int value = static_cast<int>(raw);
  if (value >= row_count) {
    std::ostringstream msg;
    msg << operation << " row index at position " << position
        << " is out of bounds for n_rows=" << row_count << ".";
    throw std::invalid_argument(msg.str());
  }
  return value;
}

template <typename I>
int checked_column_index(I raw, int column_count, const std::string &operation,
                         std::size_t position) {
  if (raw < 0) {
    std::ostringstream msg;
    msg << operation << " column index at position " << position
        << " must be non-negative.";
    throw std::invalid_argument(msg.str());
  }
  if (static_cast<unsigned long long>(raw) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    std::ostringstream msg;
    msg << operation << " column index at position " << position
        << " exceeds Accelerate's int index range.";
    throw std::overflow_error(msg.str());
  }
  const int value = static_cast<int>(raw);
  if (value >= column_count) {
    std::ostringstream msg;
    msg << operation << " column index at position " << position
        << " is out of bounds for n_cols=" << column_count << ".";
    throw std::invalid_argument(msg.str());
  }
  return value;
}

AccelerateCscMatrixFloat
build_from_columns(std::vector<ColumnAccumulator> columns, int row_count,
                   int column_count, const std::string &operation) {
  AccelerateCscMatrixFloat matrix;
  matrix.row_count = row_count;
  matrix.column_count = column_count;
  matrix.column_starts.resize(columns.size() + 1);

  for (std::size_t col = 0; col < columns.size(); ++col) {
    matrix.column_starts[col] =
        checked_column_start(matrix.row_indices.size(), operation);
    for (const auto &[row, value] : columns[col]) {
      matrix.row_indices.push_back(row);
      matrix.values.push_back(value);
    }
  }
  matrix.column_starts[columns.size()] =
      checked_column_start(matrix.row_indices.size(), operation);
  return matrix;
}

template <typename I>
AccelerateCscMatrixFloat
build_from_canonical_csc(const std::vector<float> &values,
                         const std::vector<I> &indices,
                         const std::vector<long> &indptr, int row_count,
                         int column_count, const std::string &operation) {
  AccelerateCscMatrixFloat matrix;
  matrix.row_count = row_count;
  matrix.column_count = column_count;
  matrix.column_starts = indptr;
  matrix.row_indices.reserve(indices.size());
  matrix.values = values;

  for (std::size_t i = 0; i < indices.size(); ++i) {
    matrix.row_indices.push_back(
        checked_row_index(indices[i], row_count, operation, i));
  }
  return matrix;
}

template <typename I>
bool csc_is_canonical(const std::vector<I> &indices,
                      const std::vector<long> &indptr, int row_count,
                      int column_count, const std::string &operation) {
  for (int col = 0; col < column_count; ++col) {
    int previous_row = -1;
    for (long p = indptr[static_cast<std::size_t>(col)];
         p < indptr[static_cast<std::size_t>(col + 1)]; ++p) {
      const int row =
          checked_row_index(indices[static_cast<std::size_t>(p)], row_count,
                            operation, static_cast<std::size_t>(p));
      if (row <= previous_row) {
        return false;
      }
      previous_row = row;
    }
  }
  return true;
}

template <typename I>
AccelerateCscMatrixFloat build_from_csc_indices(
    const std::vector<float> &values, const std::vector<I> &indices,
    const std::vector<long> &indptr, int row_count, int column_count,
    AccelerateCscAdapterOptions options, const std::string &operation) {
  if (csc_is_canonical(indices, indptr, row_count, column_count, operation)) {
    return build_from_canonical_csc(values, indices, indptr, row_count,
                                    column_count, operation);
  }

  if (!options.canonicalize) {
    throw std::invalid_argument(
        operation +
        " CSC input must have strictly sorted, duplicate-free row indices.");
  }

  std::vector<ColumnAccumulator> columns(
      static_cast<std::size_t>(column_count));
  for (int col = 0; col < column_count; ++col) {
    for (long p = indptr[static_cast<std::size_t>(col)];
         p < indptr[static_cast<std::size_t>(col + 1)]; ++p) {
      const auto position = static_cast<std::size_t>(p);
      const int row =
          checked_row_index(indices[position], row_count, operation, position);
      columns[static_cast<std::size_t>(col)][row] += values[position];
    }
  }
  return build_from_columns(std::move(columns), row_count, column_count,
                            operation);
}

template <typename I>
AccelerateCscMatrixFloat
make_csc_impl(const mx::array &data, const mx::array &indices,
              const mx::array &indptr, std::int64_t n_rows, std::int64_t n_cols,
              AccelerateCscAdapterOptions options,
              const std::string &operation) {
  const auto [row_count, column_count] =
      validate_shape(n_rows, n_cols, options, operation);
  require_rank_named(data, 1, named(operation, "data"));
  require_rank_named(indices, 1, named(operation, "indices"));
  require_rank_named(indptr, 1, named(operation, "indptr"));
  require_float32_named(data, named(operation, "data"));
  require_same_index_dtype_named(indices, indptr, named(operation, "indices"),
                                 named(operation, "indptr"));
  if (indices.size() != data.size()) {
    throw std::invalid_argument(operation +
                                " data and indices must have equal length.");
  }

  const auto values = copy_values(data);
  const auto index_values = copy_array<I>(indices);
  const auto pointer_values = copy_array<I>(indptr);
  const auto column_starts = validate_compressed_pointers(
      pointer_values, static_cast<std::size_t>(column_count), values.size(),
      "indptr", operation);
  return build_from_csc_indices(values, index_values, column_starts, row_count,
                                column_count, options, operation);
}

template <typename I>
AccelerateCscMatrixFloat
make_csr_impl(const mx::array &data, const mx::array &indices,
              const mx::array &indptr, std::int64_t n_rows, std::int64_t n_cols,
              AccelerateCscAdapterOptions options,
              const std::string &operation) {
  const auto [row_count, column_count] =
      validate_shape(n_rows, n_cols, options, operation);
  require_rank_named(data, 1, named(operation, "data"));
  require_rank_named(indices, 1, named(operation, "indices"));
  require_rank_named(indptr, 1, named(operation, "indptr"));
  require_float32_named(data, named(operation, "data"));
  require_same_index_dtype_named(indices, indptr, named(operation, "indices"),
                                 named(operation, "indptr"));
  if (indices.size() != data.size()) {
    throw std::invalid_argument(operation +
                                " data and indices must have equal length.");
  }

  const auto values = copy_values(data);
  const auto index_values = copy_array<I>(indices);
  const auto pointer_values = copy_array<I>(indptr);
  const auto row_starts = validate_compressed_pointers(
      pointer_values, static_cast<std::size_t>(row_count), values.size(),
      "indptr", operation);

  std::vector<ColumnAccumulator> columns(
      static_cast<std::size_t>(column_count));
  for (int row = 0; row < row_count; ++row) {
    for (long p = row_starts[static_cast<std::size_t>(row)];
         p < row_starts[static_cast<std::size_t>(row + 1)]; ++p) {
      const auto position = static_cast<std::size_t>(p);
      const int col = checked_column_index(index_values[position], column_count,
                                           operation, position);
      columns[static_cast<std::size_t>(col)][row] += values[position];
    }
  }
  return build_from_columns(std::move(columns), row_count, column_count,
                            operation);
}

template <typename I>
AccelerateCscMatrixFloat
make_coo_impl(const mx::array &data, const mx::array &row, const mx::array &col,
              std::int64_t n_rows, std::int64_t n_cols,
              AccelerateCscAdapterOptions options,
              const std::string &operation) {
  const auto [row_count, column_count] =
      validate_shape(n_rows, n_cols, options, operation);
  require_rank_named(data, 1, named(operation, "data"));
  require_rank_named(row, 1, named(operation, "row"));
  require_rank_named(col, 1, named(operation, "col"));
  require_float32_named(data, named(operation, "data"));
  require_same_index_dtype_named(row, col, named(operation, "row"),
                                 named(operation, "col"));
  if (row.size() != data.size() || col.size() != data.size()) {
    throw std::invalid_argument(operation +
                                " data, row, and col must have equal length.");
  }

  const auto values = copy_values(data);
  const auto row_values = copy_array<I>(row);
  const auto col_values = copy_array<I>(col);
  std::vector<ColumnAccumulator> columns(
      static_cast<std::size_t>(column_count));
  for (std::size_t i = 0; i < values.size(); ++i) {
    const int row_index =
        checked_row_index(row_values[i], row_count, operation, i);
    const int column_index =
        checked_column_index(col_values[i], column_count, operation, i);
    columns[static_cast<std::size_t>(column_index)][row_index] += values[i];
  }
  return build_from_columns(std::move(columns), row_count, column_count,
                            operation);
}

} // namespace

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
SparseMatrixStructure AccelerateCscMatrixFloat::structure() const {
  SparseMatrixStructure structure{};
  structure.rowCount = row_count;
  structure.columnCount = column_count;
  structure.columnStarts = const_cast<long *>(column_starts.data());
  structure.rowIndices = const_cast<int *>(row_indices.data());
  structure.blockSize = 1;
  return structure;
}

SparseMatrix_Float AccelerateCscMatrixFloat::matrix() const {
  SparseMatrix_Float result{};
  result.structure = structure();
  result.data = const_cast<float *>(values.data());
  return result;
}
#endif

AccelerateCscMatrixFloat make_accelerate_csc_matrix_float32(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    std::int64_t n_rows, std::int64_t n_cols,
    AccelerateCscAdapterOptions options, std::string_view operation) {
  const std::string op = operation_name(operation);
  if (is_int32(indices)) {
    return make_csc_impl<int32_t>(data, indices, indptr, n_rows, n_cols,
                                  options, op);
  }
  if (is_int64(indices)) {
    return make_csc_impl<int64_t>(data, indices, indptr, n_rows, n_cols,
                                  options, op);
  }
  require_index_dtype_named(indices, named(op, "indices"));
  return {};
}

AccelerateCscMatrixFloat make_accelerate_csc_matrix_float32_from_csr(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    std::int64_t n_rows, std::int64_t n_cols,
    AccelerateCscAdapterOptions options, std::string_view operation) {
  const std::string op = operation_name(operation);
  if (is_int32(indices)) {
    return make_csr_impl<int32_t>(data, indices, indptr, n_rows, n_cols,
                                  options, op);
  }
  if (is_int64(indices)) {
    return make_csr_impl<int64_t>(data, indices, indptr, n_rows, n_cols,
                                  options, op);
  }
  require_index_dtype_named(indices, named(op, "indices"));
  return {};
}

AccelerateCscMatrixFloat make_accelerate_csc_matrix_float32_from_coo(
    const mx::array &data, const mx::array &row, const mx::array &col,
    std::int64_t n_rows, std::int64_t n_cols,
    AccelerateCscAdapterOptions options, std::string_view operation) {
  const std::string op = operation_name(operation);
  if (is_int32(row)) {
    return make_coo_impl<int32_t>(data, row, col, n_rows, n_cols, options, op);
  }
  if (is_int64(row)) {
    return make_coo_impl<int64_t>(data, row, col, n_rows, n_cols, options, op);
  }
  require_index_dtype_named(row, named(op, "row"));
  return {};
}

} // namespace mlx_sparse
