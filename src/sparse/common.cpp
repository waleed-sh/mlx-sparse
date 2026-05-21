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

#include "sparse/common.h"

#include <dlfcn.h>

#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace mlx_sparse {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get mlx_sparse current binary path.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

std::string dtype_name(mx::Dtype dtype) {
  switch (dtype.val()) {
  case mx::Dtype::Val::bool_:
    return "bool";
  case mx::Dtype::Val::uint8:
    return "uint8";
  case mx::Dtype::Val::uint16:
    return "uint16";
  case mx::Dtype::Val::uint32:
    return "uint32";
  case mx::Dtype::Val::uint64:
    return "uint64";
  case mx::Dtype::Val::int8:
    return "int8";
  case mx::Dtype::Val::int16:
    return "int16";
  case mx::Dtype::Val::int32:
    return "int32";
  case mx::Dtype::Val::int64:
    return "int64";
  case mx::Dtype::Val::float16:
    return "float16";
  case mx::Dtype::Val::float32:
    return "float32";
  case mx::Dtype::Val::float64:
    return "float64";
  case mx::Dtype::Val::bfloat16:
    return "bfloat16";
  case mx::Dtype::Val::complex64:
    return "complex64";
  }
  return "unknown";
}

std::string value_kernel_suffix(mx::Dtype dtype) {
  switch (dtype.val()) {
  case mx::Dtype::Val::float16:
    return "float16";
  case mx::Dtype::Val::float32:
    return "float32";
  case mx::Dtype::Val::bfloat16:
    return "bfloat16";
  case mx::Dtype::Val::complex64:
    return "complex64";
  default:
    throw std::invalid_argument("Unsupported sparse value dtype for Metal: " +
                                dtype_name(dtype));
  }
}

std::string index_kernel_suffix(mx::Dtype dtype) {
  switch (dtype.val()) {
  case mx::Dtype::Val::int32:
    return "int32";
  case mx::Dtype::Val::int64:
    return "int64";
  default:
    throw std::invalid_argument("Unsupported sparse index dtype for Metal: " +
                                dtype_name(dtype));
  }
}

std::string sparse_kernel_name(const std::string &op, mx::Dtype value_dtype,
                               mx::Dtype index_dtype) {
  return op + "_" + value_kernel_suffix(value_dtype) + "_" +
         index_kernel_suffix(index_dtype);
}

void require_rank(const mx::array &array, int ndim, const char *name) {
  if (static_cast<int>(array.ndim()) != ndim) {
    std::ostringstream msg;
    msg << name << " must be rank-" << ndim << ", got rank " << array.ndim()
        << ".";
    throw std::invalid_argument(msg.str());
  }
}

void require_float32(const mx::array &array, const char *name) {
  if (array.dtype() != mx::float32) {
    std::ostringstream msg;
    msg << name << " must have dtype float32, got " << dtype_name(array.dtype())
        << ".";
    throw std::invalid_argument(msg.str());
  }
}

bool is_supported_value_dtype(const mx::array &array) {
  return array.dtype() == mx::float32 || array.dtype() == mx::float16 ||
         array.dtype() == mx::bfloat16 || array.dtype() == mx::complex64;
}

void require_supported_value_dtype(const mx::array &array, const char *name) {
  if (!is_supported_value_dtype(array)) {
    std::ostringstream msg;
    msg << name
        << " must have dtype float32, float16, bfloat16, or complex64, got "
        << dtype_name(array.dtype()) << ".";
    throw std::invalid_argument(msg.str());
  }
}

void require_index_dtype(const mx::array &array, const char *name) {
  if (array.dtype() != mx::int32 && array.dtype() != mx::int64) {
    std::ostringstream msg;
    msg << name << " must have dtype int32 or int64, got "
        << dtype_name(array.dtype()) << ".";
    throw std::invalid_argument(msg.str());
  }
}

void require_same_value_dtype(const mx::array &lhs, const mx::array &rhs,
                              const char *lhs_name, const char *rhs_name) {
  require_supported_value_dtype(lhs, lhs_name);
  require_supported_value_dtype(rhs, rhs_name);
  if (lhs.dtype() != rhs.dtype()) {
    std::ostringstream msg;
    msg << lhs_name << " and " << rhs_name << " must have the same dtype, got "
        << dtype_name(lhs.dtype()) << " and " << dtype_name(rhs.dtype()) << ".";
    throw std::invalid_argument(msg.str());
  }
}

void require_same_index_dtype(const mx::array &lhs, const mx::array &rhs,
                              const char *lhs_name, const char *rhs_name) {
  require_index_dtype(lhs, lhs_name);
  require_index_dtype(rhs, rhs_name);
  if (lhs.dtype() != rhs.dtype()) {
    std::ostringstream msg;
    msg << lhs_name << " and " << rhs_name << " must have the same dtype, got "
        << dtype_name(lhs.dtype()) << " and " << dtype_name(rhs.dtype()) << ".";
    throw std::invalid_argument(msg.str());
  }
}

void require_size(const mx::array &array, int expected, const char *name) {
  if (array.size() != static_cast<size_t>(expected)) {
    std::ostringstream msg;
    msg << name << " must have size " << expected << ", got " << array.size()
        << ".";
    throw std::invalid_argument(msg.str());
  }
}

bool is_int32(const mx::array &array) { return array.dtype() == mx::int32; }

bool is_int64(const mx::array &array) { return array.dtype() == mx::int64; }

} // namespace mlx_sparse
