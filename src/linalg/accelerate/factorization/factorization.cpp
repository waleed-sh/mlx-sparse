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

#include "linalg/accelerate/factorization/factorization.h"

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlx_sparse {

bool accelerate_factorization_wrappers_available() {
#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
  return true;
#else
  return false;
#endif
}

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&                                 \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= 150500
#define MLX_SPARSE_ACCELERATE_HAS_LU_FACTORIZATION_TYPES 1
#else
#define MLX_SPARSE_ACCELERATE_HAS_LU_FACTORIZATION_TYPES 0
#endif

namespace {

thread_local std::string g_last_accelerate_parameter_error;

void record_accelerate_parameter_error(const char *message) {
  g_last_accelerate_parameter_error = message ? message : "";
}

void clear_accelerate_parameter_error() {
  g_last_accelerate_parameter_error.clear();
}

std::string consume_accelerate_parameter_error() {
  std::string message = std::move(g_last_accelerate_parameter_error);
  g_last_accelerate_parameter_error.clear();
  return message;
}

std::string combine_details(std::string_view lhs, std::string_view rhs) {
  if (lhs.empty()) {
    return std::string(rhs);
  }
  if (rhs.empty()) {
    return std::string(lhs);
  }
  std::string detail(lhs);
  detail += ". ";
  detail += rhs;
  return detail;
}

bool is_symmetric_factorization(SparseFactorization_t type) {
  switch (type) {
  case SparseFactorizationCholesky:
  case SparseFactorizationLDLT:
  case SparseFactorizationLDLTUnpivoted:
  case SparseFactorizationLDLTSBK:
  case SparseFactorizationLDLTTPP:
    return true;
  default:
    return false;
  }
}

bool is_lu_factorization(SparseFactorization_t type) {
#if MLX_SPARSE_ACCELERATE_HAS_LU_FACTORIZATION_TYPES
  switch (type) {
  case SparseFactorizationLU:
  case SparseFactorizationLUUnpivoted:
  case SparseFactorizationLUSPP:
  case SparseFactorizationLUTPP:
    return true;
  default:
    return false;
  }
#else
  (void)type;
  return false;
#endif
}

void require_factorization_type_available(SparseFactorization_t type,
                                          std::string_view operation) {
  if (is_lu_factorization(type) && !accelerate_lu_factorization_available()) {
    std::string message(operation);
    if (message.empty()) {
      message = "Accelerate LU factorization";
    }
    message += " requires macOS 15.5 or newer.";
    throw std::runtime_error(message);
  }
}

int checked_size_product(int value, int multiplier, const char *name) {
  if (value < 0 || multiplier <= 0) {
    throw std::invalid_argument(std::string(name) +
                                " is not a valid Accelerate dense size.");
  }
  if (value > std::numeric_limits<int>::max() / multiplier) {
    throw std::overflow_error(std::string(name) +
                              " exceeds Accelerate dense size range.");
  }
  return value * multiplier;
}

} // namespace

const char *accelerate_factorization_name(SparseFactorization_t type) {
  switch (type) {
  case SparseFactorizationCholesky:
    return "SparseFactorizationCholesky";
  case SparseFactorizationLDLT:
    return "SparseFactorizationLDLT";
  case SparseFactorizationLDLTUnpivoted:
    return "SparseFactorizationLDLTUnpivoted";
  case SparseFactorizationLDLTSBK:
    return "SparseFactorizationLDLTSBK";
  case SparseFactorizationLDLTTPP:
    return "SparseFactorizationLDLTTPP";
  case SparseFactorizationQR:
    return "SparseFactorizationQR";
  case SparseFactorizationCholeskyAtA:
    return "SparseFactorizationCholeskyAtA";
#if MLX_SPARSE_ACCELERATE_HAS_LU_FACTORIZATION_TYPES
  case SparseFactorizationLU:
    return "SparseFactorizationLU";
  case SparseFactorizationLUUnpivoted:
    return "SparseFactorizationLUUnpivoted";
  case SparseFactorizationLUSPP:
    return "SparseFactorizationLUSPP";
  case SparseFactorizationLUTPP:
    return "SparseFactorizationLUTPP";
#endif
  default:
    return "UnknownSparseFactorization";
  }
}

const char *accelerate_subfactor_name(SparseSubfactor_t subfactor) {
  switch (subfactor) {
  case SparseSubfactorInvalid:
    return "SparseSubfactorInvalid";
  case SparseSubfactorP:
    return "SparseSubfactorP";
  case SparseSubfactorS:
    return "SparseSubfactorS";
  case SparseSubfactorL:
    return "SparseSubfactorL";
  case SparseSubfactorD:
    return "SparseSubfactorD";
  case SparseSubfactorPLPS:
    return "SparseSubfactorPLPS";
  case SparseSubfactorQ:
    return "SparseSubfactorQ";
  case SparseSubfactorR:
    return "SparseSubfactorR";
  case SparseSubfactorRP:
    return "SparseSubfactorRP";
#if MLX_SPARSE_ACCELERATE_HAS_LU_FACTORIZATION_TYPES
  case SparseSubfactorSr:
    return "SparseSubfactorSr";
  case SparseSubfactorSc:
    return "SparseSubfactorSc";
#endif
  default:
    return "UnknownSparseSubfactor";
  }
}

bool accelerate_lu_factorization_available() {
  if (__builtin_available(macOS 15.5, *)) {
    return true;
  }
  return false;
}

bool accelerate_factorization_type_available(SparseFactorization_t type) {
  return !is_lu_factorization(type) || accelerate_lu_factorization_available();
}

SparseAttributes_t
accelerate_default_attributes_for_factorization(SparseFactorization_t type) {
  SparseAttributes_t attributes{};
  attributes.triangle = SparseLowerTriangle;
  attributes.kind =
      is_symmetric_factorization(type) ? SparseSymmetric : SparseOrdinary;
  return attributes;
}

SparseSymbolicFactorOptions make_accelerate_symbolic_factor_options() {
  SparseSymbolicFactorOptions options{};
  options.control = SparseDefaultControl;
  options.orderMethod = SparseOrderDefault;
  options.order = nullptr;
  options.ignoreRowsAndColumns = nullptr;
  options.malloc = std::malloc;
  options.free = std::free;
  options.reportError = record_accelerate_parameter_error;
  return options;
}

SparseNumericFactorOptions make_accelerate_float_numeric_factor_options() {
  SparseNumericFactorOptions options{};
  options.control = SparseDefaultControl;
  options.scalingMethod = SparseScalingDefault;
  options.scaling = nullptr;
  options.pivotTolerance = 0.1;
  options.zeroTolerance = 1e-4 * FLT_EPSILON;
  return options;
}

AccelerateSymbolicFactorization::AccelerateSymbolicFactorization(
    SparseOpaqueSymbolicFactorization factorization) noexcept
    : factorization_(factorization), owns_(true) {}

AccelerateSymbolicFactorization::~AccelerateSymbolicFactorization() noexcept {
  cleanup();
}

AccelerateSymbolicFactorization::AccelerateSymbolicFactorization(
    AccelerateSymbolicFactorization &&other) noexcept
    : factorization_(other.factorization_), owns_(other.owns_) {
  other.factorization_ = {};
  other.owns_ = false;
}

AccelerateSymbolicFactorization &AccelerateSymbolicFactorization::operator=(
    AccelerateSymbolicFactorization &&other) noexcept {
  if (this != &other) {
    cleanup();
    factorization_ = other.factorization_;
    owns_ = other.owns_;
    other.factorization_ = {};
    other.owns_ = false;
  }
  return *this;
}

bool AccelerateSymbolicFactorization::has_storage() const noexcept {
  return factorization_.factorization != nullptr;
}

SparseOpaqueSymbolicFactorization
AccelerateSymbolicFactorization::release() noexcept {
  owns_ = false;
  auto out = factorization_;
  factorization_ = {};
  return out;
}

AccelerateSymbolicFactorization
AccelerateSymbolicFactorization::retained(std::string_view operation) const {
  check_ready(operation);
  clear_accelerate_parameter_error();
  AccelerateSymbolicFactorization retained(SparseRetain(factorization_));
  retained.check_ready(operation, consume_accelerate_parameter_error());
  return retained;
}

void AccelerateSymbolicFactorization::check_ready(
    std::string_view operation, std::string_view detail) const {
  check_accelerate_status(
      factorization_.status, operation,
      combine_details(detail, consume_accelerate_parameter_error()));
  if (!has_storage()) {
    throw std::runtime_error(
        std::string(operation) +
        " did not return an Accelerate symbolic factorization object.");
  }
}

void AccelerateSymbolicFactorization::cleanup() noexcept {
  if (owns_ && has_storage()) {
    SparseCleanup(factorization_);
  }
  factorization_ = {};
  owns_ = false;
}

AccelerateFloatFactorization::AccelerateFloatFactorization(
    SparseOpaqueFactorization_Float factorization) noexcept
    : factorization_(factorization), owns_(true) {}

AccelerateFloatFactorization::~AccelerateFloatFactorization() noexcept {
  cleanup();
}

AccelerateFloatFactorization::AccelerateFloatFactorization(
    AccelerateFloatFactorization &&other) noexcept
    : factorization_(other.factorization_), owns_(other.owns_) {
  other.factorization_ = {};
  other.owns_ = false;
}

AccelerateFloatFactorization &AccelerateFloatFactorization::operator=(
    AccelerateFloatFactorization &&other) noexcept {
  if (this != &other) {
    cleanup();
    factorization_ = other.factorization_;
    owns_ = other.owns_;
    other.factorization_ = {};
    other.owns_ = false;
  }
  return *this;
}

bool AccelerateFloatFactorization::has_storage() const noexcept {
  return factorization_.symbolicFactorization.factorization != nullptr ||
         factorization_.numericFactorization != nullptr;
}

SparseOpaqueFactorization_Float
AccelerateFloatFactorization::release() noexcept {
  owns_ = false;
  auto out = factorization_;
  factorization_ = {};
  return out;
}

AccelerateFloatFactorization
AccelerateFloatFactorization::retained(std::string_view operation) const {
  check_ready(operation);
  clear_accelerate_parameter_error();
  AccelerateFloatFactorization retained(SparseRetain(factorization_));
  retained.check_ready(operation, consume_accelerate_parameter_error());
  return retained;
}

void AccelerateFloatFactorization::check_ready(std::string_view operation,
                                               std::string_view detail) const {
  const auto reported =
      combine_details(detail, consume_accelerate_parameter_error());
  check_accelerate_status(factorization_.symbolicFactorization.status,
                          operation, reported);
  check_accelerate_status(factorization_.status, operation, reported);
  if (factorization_.numericFactorization == nullptr) {
    throw std::runtime_error(
        std::string(operation) +
        " did not return an Accelerate numeric factorization object.");
  }
}

std::size_t AccelerateFloatFactorization::solve_workspace_size(
    int rhs_count, std::string_view operation) const {
  if (rhs_count <= 0) {
    throw std::invalid_argument(std::string(operation) +
                                " requires at least one right-hand side.");
  }
  const auto rhs = static_cast<std::size_t>(rhs_count);
  if (factorization_.solveWorkspaceRequiredPerRHS != 0 &&
      rhs > (std::numeric_limits<std::size_t>::max() -
             factorization_.solveWorkspaceRequiredStatic) /
                factorization_.solveWorkspaceRequiredPerRHS) {
    throw std::overflow_error(std::string(operation) +
                              " workspace size overflows size_t.");
  }
  return factorization_.solveWorkspaceRequiredStatic +
         rhs * factorization_.solveWorkspaceRequiredPerRHS;
}

int AccelerateFloatFactorization::solution_size() const {
  const auto &symbolic = factorization_.symbolicFactorization;
  const int row_count = checked_size_product(
      symbolic.rowCount, symbolic.blockSize, "Accelerate row count");
  const int column_count = checked_size_product(
      symbolic.columnCount, symbolic.blockSize, "Accelerate column count");
  const bool transposed =
      symbolic.attributes.transpose ^ factorization_.attributes.transpose;
  return transposed ? row_count : column_count;
}

int AccelerateFloatFactorization::rhs_size() const {
  const auto &symbolic = factorization_.symbolicFactorization;
  if (symbolic.type != SparseFactorizationQR) {
    return solution_size();
  }
  const int row_count = checked_size_product(
      symbolic.rowCount, symbolic.blockSize, "Accelerate row count");
  const int column_count = checked_size_product(
      symbolic.columnCount, symbolic.blockSize, "Accelerate column count");
  const bool transposed =
      symbolic.attributes.transpose ^ factorization_.attributes.transpose;
  return transposed ? column_count : row_count;
}

std::vector<float>
AccelerateFloatFactorization::solve_vector(const std::vector<float> &rhs,
                                           std::string_view operation) const {
  check_ready(operation);
  const int expected_rhs_size = rhs_size();
  const int expected_solution_size = solution_size();
  if (rhs.size() != static_cast<std::size_t>(expected_rhs_size)) {
    throw std::invalid_argument(std::string(operation) +
                                " rhs has incompatible size.");
  }

  std::vector<float> solution(static_cast<std::size_t>(expected_solution_size));
  DenseVector_Float rhs_vector{expected_rhs_size,
                               const_cast<float *>(rhs.data())};
  DenseVector_Float solution_vector{expected_solution_size, solution.data()};
  std::vector<char> workspace(
      std::max<std::size_t>(solve_workspace_size(1, operation), 1));
  clear_accelerate_parameter_error();
  SparseSolve(factorization_, rhs_vector, solution_vector, workspace.data());
  const auto reported = consume_accelerate_parameter_error();
  if (!reported.empty()) {
    throw std::invalid_argument(combine_details(operation, reported));
  }
  return solution;
}

void AccelerateFloatFactorization::cleanup() noexcept {
  if (owns_ && has_storage()) {
    SparseCleanup(factorization_);
  }
  factorization_ = {};
  owns_ = false;
}

AccelerateFloatSubfactor::AccelerateFloatSubfactor(
    SparseOpaqueSubfactor_Float subfactor) noexcept
    : subfactor_(subfactor), owns_(true) {}

AccelerateFloatSubfactor::~AccelerateFloatSubfactor() noexcept { cleanup(); }

AccelerateFloatSubfactor::AccelerateFloatSubfactor(
    AccelerateFloatSubfactor &&other) noexcept
    : subfactor_(other.subfactor_), owns_(other.owns_) {
  other.subfactor_ = {};
  other.owns_ = false;
}

AccelerateFloatSubfactor &
AccelerateFloatSubfactor::operator=(AccelerateFloatSubfactor &&other) noexcept {
  if (this != &other) {
    cleanup();
    subfactor_ = other.subfactor_;
    owns_ = other.owns_;
    other.subfactor_ = {};
    other.owns_ = false;
  }
  return *this;
}

bool AccelerateFloatSubfactor::has_storage() const noexcept {
  return subfactor_.factor.symbolicFactorization.factorization != nullptr ||
         subfactor_.factor.numericFactorization != nullptr;
}

SparseOpaqueSubfactor_Float AccelerateFloatSubfactor::release() noexcept {
  owns_ = false;
  auto out = subfactor_;
  subfactor_ = {};
  return out;
}

AccelerateFloatSubfactor
AccelerateFloatSubfactor::retained(std::string_view operation) const {
  check_ready(operation);
  clear_accelerate_parameter_error();
  AccelerateFloatSubfactor retained(SparseRetain(subfactor_));
  retained.check_ready(operation, consume_accelerate_parameter_error());
  return retained;
}

void AccelerateFloatSubfactor::check_ready(std::string_view operation,
                                           std::string_view detail) const {
  const auto reported =
      combine_details(detail, consume_accelerate_parameter_error());
  check_accelerate_status(subfactor_.factor.symbolicFactorization.status,
                          operation, reported);
  check_accelerate_status(subfactor_.factor.status, operation, reported);
  if (subfactor_.contents == SparseSubfactorInvalid || !has_storage()) {
    throw std::runtime_error(std::string(operation) +
                             " did not return an Accelerate subfactor object.");
  }
}

void AccelerateFloatSubfactor::cleanup() noexcept {
  if (owns_ && has_storage()) {
    SparseCleanup(subfactor_);
  }
  subfactor_ = {};
  owns_ = false;
}

AccelerateSymbolicFactorization
make_accelerate_symbolic_factorization(SparseFactorization_t type,
                                       const AccelerateCscMatrixFloat &matrix,
                                       std::string_view operation) {
  return make_accelerate_symbolic_factorization(
      type, matrix, accelerate_default_attributes_for_factorization(type),
      operation);
}

AccelerateSymbolicFactorization make_accelerate_symbolic_factorization(
    SparseFactorization_t type, const AccelerateCscMatrixFloat &matrix,
    SparseAttributes_t attributes, std::string_view operation) {
  require_factorization_type_available(type, operation);
  auto options = make_accelerate_symbolic_factor_options();
  auto structure = matrix.structure(attributes);
  clear_accelerate_parameter_error();
  AccelerateSymbolicFactorization symbolic(
      SparseFactor(type, structure, options));
  symbolic.check_ready(operation, consume_accelerate_parameter_error());
  return symbolic;
}

AccelerateFloatFactorization
make_accelerate_float_factorization(SparseFactorization_t type,
                                    const AccelerateCscMatrixFloat &matrix,
                                    std::string_view operation) {
  return make_accelerate_float_factorization(
      type, matrix, accelerate_default_attributes_for_factorization(type),
      operation);
}

AccelerateFloatFactorization make_accelerate_float_factorization(
    SparseFactorization_t type, const AccelerateCscMatrixFloat &matrix,
    SparseAttributes_t attributes, std::string_view operation) {
  require_factorization_type_available(type, operation);
  auto symbolic_options = make_accelerate_symbolic_factor_options();
  auto numeric_options = make_accelerate_float_numeric_factor_options();
  auto sparse_matrix = matrix.matrix(attributes);
  clear_accelerate_parameter_error();
  AccelerateFloatFactorization factorization(
      SparseFactor(type, sparse_matrix, symbolic_options, numeric_options));
  factorization.check_ready(operation, consume_accelerate_parameter_error());
  return factorization;
}

AccelerateFloatFactorization make_accelerate_float_factorization(
    const AccelerateSymbolicFactorization &symbolic,
    const AccelerateCscMatrixFloat &matrix, std::string_view operation) {
  symbolic.check_ready(operation);
  auto numeric_options = make_accelerate_float_numeric_factor_options();
  auto sparse_matrix = matrix.matrix(symbolic.raw().attributes);
  clear_accelerate_parameter_error();
  AccelerateFloatFactorization factorization(
      SparseFactor(symbolic.raw(), sparse_matrix, numeric_options));
  factorization.check_ready(operation, consume_accelerate_parameter_error());
  return factorization;
}

AccelerateFloatSubfactor make_accelerate_float_subfactor(
    SparseSubfactor_t subfactor,
    const AccelerateFloatFactorization &factorization,
    std::string_view operation) {
  factorization.check_ready(operation);
  clear_accelerate_parameter_error();
  AccelerateFloatSubfactor result(
      SparseCreateSubfactor(subfactor, factorization.raw()));
  result.check_ready(operation, consume_accelerate_parameter_error());
  return result;
}

void refactor_accelerate_float_factorization(
    const AccelerateCscMatrixFloat &matrix,
    AccelerateFloatFactorization &factorization, std::string_view operation) {
  factorization.check_ready(operation);
  auto numeric_options = make_accelerate_float_numeric_factor_options();
  auto sparse_matrix =
      matrix.matrix(factorization.raw().symbolicFactorization.attributes);
  clear_accelerate_parameter_error();
  SparseRefactor(sparse_matrix, &factorization.raw(), numeric_options);
  factorization.check_ready(operation, consume_accelerate_parameter_error());
}

#endif

} // namespace mlx_sparse
