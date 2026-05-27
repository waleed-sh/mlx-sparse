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

#include <string>
#include <string_view>

#ifndef MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
#define MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK 0
#endif

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
#include <Accelerate/Accelerate.h>
#endif

namespace mlx_sparse {

enum class AccelerateStatusFamily {
  factorization,
  sparse_blas,
  iterative,
};

enum class AccelerateExceptionKind {
  none,
  value_error,
  runtime_error,
};

struct AccelerateStatusDescription {
  AccelerateExceptionKind exception;
  const char *name;
  const char *description;
};

AccelerateStatusFamily parse_accelerate_status_family(std::string_view family);

AccelerateStatusDescription
describe_accelerate_status(AccelerateStatusFamily family, int status_code);

std::string accelerate_status_name(AccelerateStatusFamily family,
                                   int status_code);

void check_accelerate_status(AccelerateStatusFamily family, int status_code,
                             std::string_view operation,
                             std::string_view detail = {});

#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
void check_accelerate_status(SparseStatus_t status, std::string_view operation,
                             std::string_view detail = {});

void check_accelerate_status(sparse_status status, std::string_view operation,
                             std::string_view detail = {});

void check_accelerate_status(SparseIterativeStatus_t status,
                             std::string_view operation,
                             std::string_view detail = {});
#endif

} // namespace mlx_sparse
