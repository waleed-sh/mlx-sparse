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

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "linalg/accelerate/adapter/csc_adapter.h"
#include "linalg/accelerate/errors/status.h"

namespace mlx_sparse {

bool accelerate_factorization_wrappers_available();

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK

const char *accelerate_factorization_name(SparseFactorization_t type);
const char *accelerate_subfactor_name(SparseSubfactor_t subfactor);
bool accelerate_lu_factorization_available();
bool accelerate_factorization_type_available(SparseFactorization_t type);

SparseAttributes_t
accelerate_default_attributes_for_factorization(SparseFactorization_t type);
SparseSymbolicFactorOptions make_accelerate_symbolic_factor_options();
SparseNumericFactorOptions make_accelerate_float_numeric_factor_options();

class AccelerateSymbolicFactorization final {
public:
  AccelerateSymbolicFactorization() noexcept = default;
  explicit AccelerateSymbolicFactorization(
      SparseOpaqueSymbolicFactorization factorization) noexcept;
  ~AccelerateSymbolicFactorization() noexcept;

  AccelerateSymbolicFactorization(const AccelerateSymbolicFactorization &) =
      delete;
  AccelerateSymbolicFactorization &
  operator=(const AccelerateSymbolicFactorization &) = delete;

  AccelerateSymbolicFactorization(
      AccelerateSymbolicFactorization &&other) noexcept;
  AccelerateSymbolicFactorization &
  operator=(AccelerateSymbolicFactorization &&other) noexcept;

  bool owns() const noexcept { return owns_; }
  bool has_storage() const noexcept;
  SparseStatus_t status() const noexcept { return factorization_.status; }
  SparseFactorization_t type() const noexcept { return factorization_.type; }
  int row_count() const noexcept { return factorization_.rowCount; }
  int column_count() const noexcept { return factorization_.columnCount; }
  std::size_t workspace_size_float() const noexcept {
    return factorization_.workspaceSize_Float;
  }
  std::size_t factor_size_float() const noexcept {
    return factorization_.factorSize_Float;
  }

  const SparseOpaqueSymbolicFactorization &raw() const noexcept {
    return factorization_;
  }
  SparseOpaqueSymbolicFactorization release() noexcept;
  AccelerateSymbolicFactorization
  retained(std::string_view operation = "accelerate symbolic retain") const;
  void check_ready(std::string_view operation,
                   std::string_view detail = {}) const;

private:
  void cleanup() noexcept;

  SparseOpaqueSymbolicFactorization factorization_{};
  bool owns_ = false;
};

class AccelerateFloatFactorization final {
public:
  AccelerateFloatFactorization() noexcept = default;
  explicit AccelerateFloatFactorization(
      SparseOpaqueFactorization_Float factorization) noexcept;
  ~AccelerateFloatFactorization() noexcept;

  AccelerateFloatFactorization(const AccelerateFloatFactorization &) = delete;
  AccelerateFloatFactorization &
  operator=(const AccelerateFloatFactorization &) = delete;

  AccelerateFloatFactorization(AccelerateFloatFactorization &&other) noexcept;
  AccelerateFloatFactorization &
  operator=(AccelerateFloatFactorization &&other) noexcept;

  bool owns() const noexcept { return owns_; }
  bool has_storage() const noexcept;
  SparseStatus_t status() const noexcept { return factorization_.status; }
  SparseStatus_t symbolic_status() const noexcept {
    return factorization_.symbolicFactorization.status;
  }
  SparseFactorization_t type() const noexcept {
    return factorization_.symbolicFactorization.type;
  }
  int row_count() const noexcept {
    return factorization_.symbolicFactorization.rowCount;
  }
  int column_count() const noexcept {
    return factorization_.symbolicFactorization.columnCount;
  }
  std::size_t solve_workspace_static() const noexcept {
    return factorization_.solveWorkspaceRequiredStatic;
  }
  std::size_t solve_workspace_per_rhs() const noexcept {
    return factorization_.solveWorkspaceRequiredPerRHS;
  }

  const SparseOpaqueFactorization_Float &raw() const noexcept {
    return factorization_;
  }
  SparseOpaqueFactorization_Float &raw() noexcept { return factorization_; }
  SparseOpaqueFactorization_Float release() noexcept;
  AccelerateFloatFactorization retained(
      std::string_view operation = "accelerate factorization retain") const;
  void check_ready(std::string_view operation,
                   std::string_view detail = {}) const;
  std::size_t solve_workspace_size(
      int rhs_count,
      std::string_view operation = "accelerate solve workspace") const;
  int solution_size() const;
  int rhs_size() const;
  std::vector<float> solve_vector(
      const std::vector<float> &rhs,
      std::string_view operation = "accelerate factorization solve") const;
  std::vector<float> solve_matrix_column_major(
      const std::vector<float> &rhs, int rhs_count,
      std::string_view operation = "accelerate factorization solve") const;

private:
  void cleanup() noexcept;

  SparseOpaqueFactorization_Float factorization_{};
  bool owns_ = false;
};

class AccelerateFloatSubfactor final {
public:
  AccelerateFloatSubfactor() noexcept = default;
  explicit AccelerateFloatSubfactor(
      SparseOpaqueSubfactor_Float subfactor) noexcept;
  ~AccelerateFloatSubfactor() noexcept;

  AccelerateFloatSubfactor(const AccelerateFloatSubfactor &) = delete;
  AccelerateFloatSubfactor &
  operator=(const AccelerateFloatSubfactor &) = delete;

  AccelerateFloatSubfactor(AccelerateFloatSubfactor &&other) noexcept;
  AccelerateFloatSubfactor &
  operator=(AccelerateFloatSubfactor &&other) noexcept;

  bool owns() const noexcept { return owns_; }
  bool has_storage() const noexcept;
  SparseStatus_t status() const noexcept { return subfactor_.factor.status; }
  SparseSubfactor_t contents() const noexcept { return subfactor_.contents; }
  std::size_t workspace_static() const noexcept {
    return subfactor_.workspaceRequiredStatic;
  }
  std::size_t workspace_per_rhs() const noexcept {
    return subfactor_.workspaceRequiredPerRHS;
  }

  const SparseOpaqueSubfactor_Float &raw() const noexcept { return subfactor_; }
  SparseOpaqueSubfactor_Float release() noexcept;
  AccelerateFloatSubfactor
  retained(std::string_view operation = "accelerate subfactor retain") const;
  void check_ready(std::string_view operation,
                   std::string_view detail = {}) const;

private:
  void cleanup() noexcept;

  SparseOpaqueSubfactor_Float subfactor_{};
  bool owns_ = false;
};

AccelerateSymbolicFactorization make_accelerate_symbolic_factorization(
    SparseFactorization_t type, const AccelerateCscMatrixFloat &matrix,
    std::string_view operation = "accelerate symbolic factorization");

AccelerateSymbolicFactorization make_accelerate_symbolic_factorization(
    SparseFactorization_t type, const AccelerateCscMatrixFloat &matrix,
    SparseAttributes_t attributes,
    std::string_view operation = "accelerate symbolic factorization");

AccelerateFloatFactorization make_accelerate_float_factorization(
    SparseFactorization_t type, const AccelerateCscMatrixFloat &matrix,
    std::string_view operation = "accelerate float factorization");

AccelerateFloatFactorization make_accelerate_float_factorization(
    SparseFactorization_t type, const AccelerateCscMatrixFloat &matrix,
    SparseAttributes_t attributes,
    std::string_view operation = "accelerate float factorization");

AccelerateFloatFactorization make_accelerate_float_factorization(
    const AccelerateSymbolicFactorization &symbolic,
    const AccelerateCscMatrixFloat &matrix,
    std::string_view operation = "accelerate float numeric factorization");

AccelerateFloatSubfactor make_accelerate_float_subfactor(
    SparseSubfactor_t subfactor,
    const AccelerateFloatFactorization &factorization,
    std::string_view operation = "accelerate float subfactor");

void refactor_accelerate_float_factorization(
    const AccelerateCscMatrixFloat &matrix,
    AccelerateFloatFactorization &factorization,
    std::string_view operation = "accelerate float refactor");

#endif

} // namespace mlx_sparse
