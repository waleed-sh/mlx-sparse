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

#include "linalg/accelerate/accelerate_errors.h"

#include <limits>
#include <stdexcept>

namespace mlx_sparse {
namespace {

constexpr int kSparseStatusReleased = -std::numeric_limits<int>::max();

AccelerateStatusDescription unknown_status() {
  return {
      AccelerateExceptionKind::runtime_error,
      "UnknownAccelerateStatus",
      "Accelerate returned an unrecognized non-success status",
  };
}

AccelerateStatusDescription describe_factorization_status(int status_code) {
  if (status_code >= 0) {
    return {
        AccelerateExceptionKind::none,
        "SparseStatusOK",
        "factorization completed successfully",
    };
  }

  switch (status_code) {
  case -1:
    return {
        AccelerateExceptionKind::runtime_error,
        "SparseFactorizationFailed",
        "factorization failed due to a numerical issue",
    };
  case -2:
    return {
        AccelerateExceptionKind::runtime_error,
        "SparseMatrixIsSingular",
        "factorization aborted because the matrix is singular",
    };
  case -3:
    return {
        AccelerateExceptionKind::runtime_error,
        "SparseInternalError",
        "factorization encountered an internal Accelerate error",
    };
  case -4:
    return {
        AccelerateExceptionKind::value_error,
        "SparseParameterError",
        "Accelerate rejected one or more user-supplied parameters",
    };
  case kSparseStatusReleased:
    return {
        AccelerateExceptionKind::runtime_error,
        "SparseStatusReleased",
        "factorization object has already been released",
    };
  default:
    return unknown_status();
  }
}

AccelerateStatusDescription describe_sparse_blas_status(int status_code) {
  switch (status_code) {
  case 0:
    return {
        AccelerateExceptionKind::none,
        "SPARSE_SUCCESS",
        "operation completed successfully",
    };
  case -1000:
    return {
        AccelerateExceptionKind::value_error,
        "SPARSE_ILLEGAL_PARAMETER",
        "Accelerate rejected one or more illegal parameters",
    };
  case -1001:
    return {
        AccelerateExceptionKind::runtime_error,
        "SPARSE_CANNOT_SET_PROPERTY",
        "matrix properties cannot be changed after values are inserted",
    };
  case -1002:
    return {
        AccelerateExceptionKind::runtime_error,
        "SPARSE_SYSTEM_ERROR",
        "Accelerate encountered an internal system error",
    };
  default:
    return unknown_status();
  }
}

AccelerateStatusDescription describe_iterative_status(int status_code) {
  switch (status_code) {
  case 0:
    return {
        AccelerateExceptionKind::none,
        "SparseIterativeConverged",
        "all solution vectors converged",
    };
  case 1:
    return {
        AccelerateExceptionKind::runtime_error,
        "SparseIterativeMaxIterations",
        "one or more solution vectors failed to converge",
    };
  case -1:
    return {
        AccelerateExceptionKind::value_error,
        "SparseIterativeParameterError",
        "Accelerate rejected one or more user-supplied parameters",
    };
  case -2:
    return {
        AccelerateExceptionKind::runtime_error,
        "SparseIterativeIllConditioned",
        "problem is ill-conditioned and convergence is unlikely",
    };
  case -99:
    return {
        AccelerateExceptionKind::runtime_error,
        "SparseIterativeInternalError",
        "iterative solver encountered an internal Accelerate error",
    };
  default:
    return unknown_status();
  }
}

const char *family_name(AccelerateStatusFamily family) {
  switch (family) {
  case AccelerateStatusFamily::factorization:
    return "factorization";
  case AccelerateStatusFamily::sparse_blas:
    return "Sparse BLAS";
  case AccelerateStatusFamily::iterative:
    return "iterative";
  }
  return "unknown";
}

std::string format_status_error(AccelerateStatusFamily family, int status_code,
                                const AccelerateStatusDescription &status,
                                std::string_view operation,
                                std::string_view detail) {
  std::string message(operation);
  if (message.empty()) {
    message = "Accelerate operation";
  }
  message += " failed with Accelerate ";
  message += family_name(family);
  message += " status ";
  message += status.name;
  message += " (";
  message += std::to_string(status_code);
  message += "): ";
  message += status.description;
  if (!detail.empty()) {
    message += ". ";
    message += detail;
  }
  return message;
}

} // namespace

AccelerateStatusFamily parse_accelerate_status_family(std::string_view family) {
  if (family == "factorization" || family == "solve") {
    return AccelerateStatusFamily::factorization;
  }
  if (family == "sparse_blas" || family == "blas" || family == "opaque") {
    return AccelerateStatusFamily::sparse_blas;
  }
  if (family == "iterative") {
    return AccelerateStatusFamily::iterative;
  }
  throw std::invalid_argument("unknown Accelerate status family '" +
                              std::string(family) + "'.");
}

AccelerateStatusDescription
describe_accelerate_status(AccelerateStatusFamily family, int status_code) {
  switch (family) {
  case AccelerateStatusFamily::factorization:
    return describe_factorization_status(status_code);
  case AccelerateStatusFamily::sparse_blas:
    return describe_sparse_blas_status(status_code);
  case AccelerateStatusFamily::iterative:
    return describe_iterative_status(status_code);
  }
  return unknown_status();
}

std::string accelerate_status_name(AccelerateStatusFamily family,
                                   int status_code) {
  return describe_accelerate_status(family, status_code).name;
}

void check_accelerate_status(AccelerateStatusFamily family, int status_code,
                             std::string_view operation,
                             std::string_view detail) {
  const auto status = describe_accelerate_status(family, status_code);
  switch (status.exception) {
  case AccelerateExceptionKind::none:
    return;
  case AccelerateExceptionKind::value_error:
    throw std::invalid_argument(
        format_status_error(family, status_code, status, operation, detail));
  case AccelerateExceptionKind::runtime_error:
    throw std::runtime_error(
        format_status_error(family, status_code, status, operation, detail));
  }
}

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
void check_accelerate_status(SparseStatus_t status, std::string_view operation,
                             std::string_view detail) {
  check_accelerate_status(AccelerateStatusFamily::factorization,
                          static_cast<int>(status), operation, detail);
}

void check_accelerate_status(sparse_status status, std::string_view operation,
                             std::string_view detail) {
  check_accelerate_status(AccelerateStatusFamily::sparse_blas,
                          static_cast<int>(status), operation, detail);
}

void check_accelerate_status(SparseIterativeStatus_t status,
                             std::string_view operation,
                             std::string_view detail) {
  check_accelerate_status(AccelerateStatusFamily::iterative,
                          static_cast<int>(status), operation, detail);
}
#endif

} // namespace mlx_sparse
