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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "linalg/accelerate/accelerate_csc_adapter.h"
#include "linalg/accelerate/accelerate_errors.h"
#include "linalg/linalg.h"
#include "sparse/coo_batched_matmul/coo_batched_matmul.h"
#include "sparse/coo_col_norms/coo_col_norms.h"
#include "sparse/coo_col_sums/coo_col_sums.h"
#include "sparse/coo_diagonal/coo_diagonal.h"
#include "sparse/coo_matmat/coo_matmat.h"
#include "sparse/coo_matmul/coo_matmul.h"
#include "sparse/coo_matmul_data_vjp/coo_matmul_data_vjp.h"
#include "sparse/coo_row_norms/coo_row_norms.h"
#include "sparse/coo_row_sums/coo_row_sums.h"
#include "sparse/coo_tocsc/coo_tocsc.h"
#include "sparse/coo_tocsr/coo_tocsr.h"
#include "sparse/coo_trace/coo_trace.h"
#include "sparse/csc_batched_matmul/csc_batched_matmul.h"
#include "sparse/csc_col_norms/csc_col_norms.h"
#include "sparse/csc_col_sums/csc_col_sums.h"
#include "sparse/csc_diagonal/csc_diagonal.h"
#include "sparse/csc_matmat/csc_matmat.h"
#include "sparse/csc_matmul/csc_matmul.h"
#include "sparse/csc_matmul_data_vjp/csc_matmul_data_vjp.h"
#include "sparse/csc_matmul_transpose/csc_matmul_transpose.h"
#include "sparse/csc_matvec/csc_matvec.h"
#include "sparse/csc_matvec_transpose/csc_matvec_transpose.h"
#include "sparse/csc_row_norms/csc_row_norms.h"
#include "sparse/csc_row_sums/csc_row_sums.h"
#include "sparse/csc_sort_indices/csc_sort_indices.h"
#include "sparse/csc_sum_duplicates/csc_sum_duplicates.h"
#include "sparse/csc_tocsr/csc_tocsr.h"
#include "sparse/csc_todense/csc_todense.h"
#include "sparse/csc_trace/csc_trace.h"
#include "sparse/csr_batched_matmul/csr_batched_matmul.h"
#include "sparse/csr_batched_matvec/csr_batched_matvec.h"
#include "sparse/csr_col_sums/csr_col_sums.h"
#include "sparse/csr_diagonal/csr_diagonal.h"
#include "sparse/csr_matmat/csr_matmat.h"
#include "sparse/csr_matmul/csr_matmul.h"
#include "sparse/csr_matmul_transpose/csr_matmul_transpose.h"
#include "sparse/csr_matvec/csr_matvec.h"
#include "sparse/csr_matvec_transpose/csr_matvec_transpose.h"
#include "sparse/csr_row_norms/csr_row_norms.h"
#include "sparse/csr_row_sums/csr_row_sums.h"
#include "sparse/csr_sort_indices/csr_sort_indices.h"
#include "sparse/csr_sum_duplicates/csr_sum_duplicates.h"
#include "sparse/csr_tocsc/csr_tocsc.h"
#include "sparse/csr_todense/csr_todense.h"
#include "sparse/csr_trace/csr_trace.h"
#include "sparse/csr_transpose/csr_transpose.h"
#include "sparse/fromdense/fromdense.h"
#include "sparse/identity_like/identity_like.h"

namespace nb = nanobind;
using namespace nb::literals;

#ifndef MLX_SPARSE_HAS_CPU
#define MLX_SPARSE_HAS_CPU 1
#endif

#ifndef MLX_SPARSE_HAS_METAL
#define MLX_SPARSE_HAS_METAL 0
#endif

#ifndef MLX_SPARSE_HAS_ACCELERATE
#define MLX_SPARSE_HAS_ACCELERATE 0
#endif

#ifndef MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
#define MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK 0
#endif

#ifndef MLX_SPARSE_HAS_CUDA
#define MLX_SPARSE_HAS_CUDA 0
#endif

#ifndef MLX_SPARSE_HAS_ROCM
#define MLX_SPARSE_HAS_ROCM 0
#endif

namespace {

const char *native_platform() {
#if defined(__APPLE__)
  return "darwin";
#elif defined(__linux__)
  return "linux";
#elif defined(_WIN32)
  return "windows";
#else
  return "unknown";
#endif
}

const char *native_architecture() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#else
  return "unknown";
#endif
}

mlx_sparse::AccelerateCscAdapterOptions
accelerate_csc_adapter_options(bool require_square, bool require_non_empty,
                               bool canonicalize) {
  mlx_sparse::AccelerateCscAdapterOptions options;
  options.require_square = require_square;
  options.require_non_empty = require_non_empty;
  options.canonicalize = canonicalize;
  return options;
}

nb::dict accelerate_csc_adapter_summary(
    const mlx_sparse::AccelerateCscMatrixFloat &matrix) {
  nb::dict out;
  out["n_rows"] = matrix.row_count;
  out["n_cols"] = matrix.column_count;
  out["nnz"] = static_cast<long long>(matrix.nnz());
  out["column_starts"] = matrix.column_starts;
  out["row_indices"] = matrix.row_indices;
  out["values"] = matrix.values;
#if defined(__APPLE__) && MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK
  const auto sparse = matrix.matrix();
  out["accelerate_framework"] = true;
  out["accelerate_row_count"] = sparse.structure.rowCount;
  out["accelerate_column_count"] = sparse.structure.columnCount;
  out["accelerate_block_size"] = static_cast<int>(sparse.structure.blockSize);
  out["accelerate_data_points_to_owned_values"] =
      sparse.data == matrix.values.data();
#else
  out["accelerate_framework"] = false;
#endif
  return out;
}

} // namespace

NB_MODULE(_ext, m) {
  m.doc() = "Native sparse primitives for MLX";

  m.def(
      "_compiled_capabilities",
      []() {
        nb::dict info;
        info["extension"] = true;
        info["cpu"] = static_cast<bool>(MLX_SPARSE_HAS_CPU);
        info["metal"] = static_cast<bool>(MLX_SPARSE_HAS_METAL);
        info["accelerate"] = static_cast<bool>(MLX_SPARSE_HAS_ACCELERATE);
        info["accelerate_framework"] =
            static_cast<bool>(MLX_SPARSE_HAS_ACCELERATE_FRAMEWORK);
        info["cuda"] = static_cast<bool>(MLX_SPARSE_HAS_CUDA);
        info["rocm"] = static_cast<bool>(MLX_SPARSE_HAS_ROCM);
        info["platform"] = native_platform();
        info["architecture"] = native_architecture();
        return info;
      },
      "Return compile-time native backend facts for capability reporting.");

  m.def(
      "_accelerate_status_name_for_testing",
      [](const std::string &family, int status_code) {
        return mlx_sparse::accelerate_status_name(
            mlx_sparse::parse_accelerate_status_family(family), status_code);
      },
      "family"_a, "status_code"_a,
      "Return the native Accelerate status name used by the error mapper.");

  m.def(
      "_accelerate_check_status_for_testing",
      [](const std::string &family, int status_code,
         const std::string &operation, const std::string &detail) {
        mlx_sparse::check_accelerate_status(
            mlx_sparse::parse_accelerate_status_family(family), status_code,
            operation, detail);
      },
      "family"_a, "status_code"_a, "operation"_a = "Accelerate operation",
      "detail"_a = "",
      "Raise the Python exception produced for an Accelerate status.");

  m.def(
      "_accelerate_csc_adapter_summary_for_testing",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, long long n_rows,
         long long n_cols, bool require_square, bool require_non_empty,
         bool canonicalize) {
        auto matrix = mlx_sparse::make_accelerate_csc_matrix_float32(
            data, indices, indptr, n_rows, n_cols,
            accelerate_csc_adapter_options(require_square, require_non_empty,
                                           canonicalize),
            "accelerate_csc_adapter");
        return accelerate_csc_adapter_summary(matrix);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "require_square"_a = false, "require_non_empty"_a = true,
      "canonicalize"_a = true,
      "Return the validated Accelerate CSC adapter state used by tests.");

  m.def(
      "_accelerate_csr_adapter_summary_for_testing",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, long long n_rows,
         long long n_cols, bool require_square, bool require_non_empty,
         bool canonicalize) {
        auto matrix = mlx_sparse::make_accelerate_csc_matrix_float32_from_csr(
            data, indices, indptr, n_rows, n_cols,
            accelerate_csc_adapter_options(require_square, require_non_empty,
                                           canonicalize),
            "accelerate_csr_adapter");
        return accelerate_csc_adapter_summary(matrix);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "require_square"_a = false, "require_non_empty"_a = true,
      "canonicalize"_a = true,
      "Return the validated CSR-to-Accelerate CSC adapter state used by "
      "tests.");

  m.def(
      "_accelerate_coo_adapter_summary_for_testing",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, long long n_rows, long long n_cols,
         bool require_square, bool require_non_empty, bool canonicalize) {
        auto matrix = mlx_sparse::make_accelerate_csc_matrix_float32_from_coo(
            data, row, col, n_rows, n_cols,
            accelerate_csc_adapter_options(require_square, require_non_empty,
                                           canonicalize),
            "accelerate_coo_adapter");
        return accelerate_csc_adapter_summary(matrix);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "require_square"_a = false, "require_non_empty"_a = true,
      "canonicalize"_a = true,
      "Return the validated COO-to-Accelerate CSC adapter state used by "
      "tests.");

  m.def(
      "identity_like",
      [](const mlx_sparse::mx::array &x) {
        return mlx_sparse::identity_like(x);
      },
      "x"_a, "Return a native MLX copy of x. Used as an extension smoke test.");

  m.def(
      "coo_tocsr",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, int n_rows, int n_cols) {
        return mlx_sparse::coo_tocsr(data, row, col, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "Convert COO buffers to row-sorted CSR buffers, preserving duplicates.");

  m.def(
      "coo_tocsc",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, int n_rows, int n_cols) {
        return mlx_sparse::coo_tocsc(data, row, col, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "Convert COO buffers to column-sorted CSC buffers, preserving "
      "duplicates.");

  m.def(
      "csr_todense",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_todense(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Materialize CSR buffers as a dense MLX array.");

  m.def(
      "csc_todense",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csc_todense(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Materialize CSC buffers as a dense MLX array.");

  m.def(
      "csr_sort_indices",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr) {
        return mlx_sparse::csr_sort_indices(data, indices, indptr);
      },
      "data"_a, "indices"_a, "indptr"_a,
      "Sort CSR column indices independently within each row.");

  m.def(
      "csr_sum_duplicates",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr) {
        return mlx_sparse::csr_sum_duplicates(data, indices, indptr);
      },
      "data"_a, "indices"_a, "indptr"_a,
      "Sum adjacent duplicate CSR column entries in each sorted row.");

  m.def(
      "csc_sort_indices",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr) {
        return mlx_sparse::csc_sort_indices(data, indices, indptr);
      },
      "data"_a, "indices"_a, "indptr"_a,
      "Sort CSC row indices independently within each column.");

  m.def(
      "csc_sum_duplicates",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr) {
        return mlx_sparse::csc_sum_duplicates(data, indices, indptr);
      },
      "data"_a, "indices"_a, "indptr"_a,
      "Sum adjacent duplicate CSC row entries in each sorted column.");

  m.def(
      "csr_fromdense",
      [](const mlx_sparse::mx::array &dense, int index_dtype_bits,
         float threshold) {
        return mlx_sparse::csr_fromdense(dense, index_dtype_bits, threshold);
      },
      "dense"_a, "index_dtype_bits"_a, "threshold"_a,
      "Convert a dense rank-2 MLX array into canonical CSR buffers.");

  m.def(
      "csr_transpose",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_transpose(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Transpose CSR buffers into a new row-sorted CSR representation.");

  m.def(
      "csr_tocsc",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_tocsc(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Convert CSR buffers into CSC buffers.");

  m.def(
      "csc_tocsr",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csc_tocsr(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Convert CSC buffers into CSR buffers.");

  m.def(
      "csr_row_sums",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_row_sums(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Reduce each CSR row to the sum of its stored values.");

  m.def(
      "csr_col_sums",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_col_sums(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Reduce each CSR column to the sum of its stored values.");

  m.def(
      "csr_row_norms",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_row_norms(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute the L2 norm of each CSR row.");

  m.def(
      "csr_diagonal",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_diagonal(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Extract the summed diagonal of CSR buffers.");

  m.def(
      "csr_trace",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_trace(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute the trace of CSR buffers.");

  m.def(
      "coo_row_sums",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, int n_rows, int n_cols) {
        return mlx_sparse::coo_row_sums(data, row, col, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "Reduce each COO row to the sum of its stored values.");

  m.def(
      "coo_col_sums",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, int n_rows, int n_cols) {
        return mlx_sparse::coo_col_sums(data, row, col, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "Reduce each COO column to the sum of its stored values.");

  m.def(
      "coo_row_norms",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, int n_rows, int n_cols) {
        return mlx_sparse::coo_row_norms(data, row, col, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "Compute the L2 norm of each canonical COO row.");

  m.def(
      "coo_col_norms",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, int n_rows, int n_cols) {
        return mlx_sparse::coo_col_norms(data, row, col, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "Compute the L2 norm of each canonical COO column.");

  m.def(
      "coo_diagonal",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, int n_rows, int n_cols) {
        return mlx_sparse::coo_diagonal(data, row, col, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "Extract the summed diagonal of COO buffers.");

  m.def(
      "coo_trace",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, int n_rows, int n_cols) {
        return mlx_sparse::coo_trace(data, row, col, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "n_rows"_a, "n_cols"_a,
      "Compute the trace of COO buffers.");

  m.def(
      "csc_row_sums",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csc_row_sums(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Reduce each CSC row to the sum of its stored values.");

  m.def(
      "csc_col_sums",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csc_col_sums(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Reduce each CSC column to the sum of its stored values.");

  m.def(
      "csc_row_norms",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csc_row_norms(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute the L2 norm of each canonical CSC row.");

  m.def(
      "csc_col_norms",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csc_col_norms(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute the L2 norm of each canonical CSC column.");

  m.def(
      "csc_diagonal",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csc_diagonal(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Extract the summed diagonal of CSC buffers.");

  m.def(
      "csc_trace",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csc_trace(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute the trace of CSC buffers.");

  m.def(
      "csr_matvec",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &x,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_matvec(data, indices, indptr, x, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "x"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSR buffers by a dense vector.");

  m.def(
      "coo_matvec",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, const mlx_sparse::mx::array &x,
         int n_rows, int n_cols) {
        return mlx_sparse::coo_matvec(data, row, col, x, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "x"_a, "n_rows"_a, "n_cols"_a,
      "Multiply COO buffers by a dense vector.");

  m.def(
      "csc_matvec",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &x,
         int n_rows, int n_cols) {
        return mlx_sparse::csc_matvec(data, indices, indptr, x, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "x"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSC buffers by a dense vector.");

  m.def(
      "csr_batched_matvec",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_batched_matvec(data, indices, indptr, rhs,
                                              n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSR buffers by a batch of dense vectors.");

  m.def(
      "coo_batched_matvec",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::coo_batched_matvec(data, row, col, rhs, n_rows,
                                              n_cols);
      },
      "data"_a, "row"_a, "col"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply COO buffers by a batch of dense vectors.");

  m.def(
      "csc_batched_matvec",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csc_batched_matvec(data, indices, indptr, rhs,
                                              n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSC buffers by a batch of dense vectors.");

  m.def(
      "csr_matvec_transpose",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &x,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_matvec_transpose(data, indices, indptr, x,
                                                n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "x"_a, "n_rows"_a, "n_cols"_a,
      "Multiply the transpose of CSR buffers by a dense vector.");

  m.def(
      "csc_matvec_transpose",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &x,
         int n_rows, int n_cols) {
        return mlx_sparse::csc_matvec_transpose(data, indices, indptr, x,
                                                n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "x"_a, "n_rows"_a, "n_cols"_a,
      "Multiply the transpose of CSC buffers by a dense vector.");

  m.def(
      "csr_matmul",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_matmul(data, indices, indptr, rhs, n_rows,
                                      n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSR buffers by a dense matrix.");

  m.def(
      "coo_matmul",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::coo_matmul(data, row, col, rhs, n_rows, n_cols);
      },
      "data"_a, "row"_a, "col"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply COO buffers by a dense matrix.");

  m.def(
      "csc_matmul",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csc_matmul(data, indices, indptr, rhs, n_rows,
                                      n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSC buffers by a dense matrix.");

  m.def(
      "csr_batched_matmul",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_batched_matmul(data, indices, indptr, rhs,
                                              n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSR buffers by a batch of dense matrices.");

  m.def(
      "coo_batched_matmul",
      [](const mlx_sparse::mx::array &data, const mlx_sparse::mx::array &row,
         const mlx_sparse::mx::array &col, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::coo_batched_matmul(data, row, col, rhs, n_rows,
                                              n_cols);
      },
      "data"_a, "row"_a, "col"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply COO buffers by a batch of dense matrices.");

  m.def(
      "csc_batched_matmul",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csc_batched_matmul(data, indices, indptr, rhs,
                                              n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply CSC buffers by a batch of dense matrices.");

  m.def(
      "csr_matmul_transpose",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csr_matmul_transpose(data, indices, indptr, rhs,
                                                n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply the transpose of CSR buffers by a dense matrix.");

  m.def(
      "csc_matmul_transpose",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         int n_rows, int n_cols) {
        return mlx_sparse::csc_matmul_transpose(data, indices, indptr, rhs,
                                                n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "rhs"_a, "n_rows"_a, "n_cols"_a,
      "Multiply the transpose of CSC buffers by a dense matrix.");

  m.def(
      "coo_matmul_data_vjp",
      [](const mlx_sparse::mx::array &row, const mlx_sparse::mx::array &col,
         const mlx_sparse::mx::array &rhs,
         const mlx_sparse::mx::array &cotangent, int n_rows, int n_cols) {
        return mlx_sparse::coo_matmul_data_vjp(row, col, rhs, cotangent, n_rows,
                                               n_cols);
      },
      "row"_a, "col"_a, "rhs"_a, "cotangent"_a, "n_rows"_a, "n_cols"_a,
      "Compute COO sparse-value VJP for sparse-dense products.");

  m.def(
      "csc_matmul_data_vjp",
      [](const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &rhs,
         const mlx_sparse::mx::array &cotangent, int n_rows, int n_cols) {
        return mlx_sparse::csc_matmul_data_vjp(indices, indptr, rhs, cotangent,
                                               n_rows, n_cols);
      },
      "indices"_a, "indptr"_a, "rhs"_a, "cotangent"_a, "n_rows"_a, "n_cols"_a,
      "Compute CSC sparse-value VJP for sparse-dense products.");

  m.def(
      "coo_matmat",
      [](const mlx_sparse::mx::array &lhs_data,
         const mlx_sparse::mx::array &lhs_row,
         const mlx_sparse::mx::array &lhs_col,
         const mlx_sparse::mx::array &rhs_data,
         const mlx_sparse::mx::array &rhs_row,
         const mlx_sparse::mx::array &rhs_col, int lhs_n_rows, int lhs_n_cols,
         int rhs_n_rows, int rhs_n_cols) {
        return mlx_sparse::coo_matmat(lhs_data, lhs_row, lhs_col, rhs_data,
                                      rhs_row, rhs_col, lhs_n_rows, lhs_n_cols,
                                      rhs_n_rows, rhs_n_cols);
      },
      "lhs_data"_a, "lhs_row"_a, "lhs_col"_a, "rhs_data"_a, "rhs_row"_a,
      "rhs_col"_a, "lhs_n_rows"_a, "lhs_n_cols"_a, "rhs_n_rows"_a,
      "rhs_n_cols"_a,
      "Multiply two COO matrices into a canonical COO representation.");

  m.def(
      "csc_matmat",
      [](const mlx_sparse::mx::array &lhs_data,
         const mlx_sparse::mx::array &lhs_indices,
         const mlx_sparse::mx::array &lhs_indptr,
         const mlx_sparse::mx::array &rhs_data,
         const mlx_sparse::mx::array &rhs_indices,
         const mlx_sparse::mx::array &rhs_indptr, int lhs_n_rows,
         int lhs_n_cols, int rhs_n_rows, int rhs_n_cols) {
        return mlx_sparse::csc_matmat(
            lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,
            rhs_indptr, lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols);
      },
      "lhs_data"_a, "lhs_indices"_a, "lhs_indptr"_a, "rhs_data"_a,
      "rhs_indices"_a, "rhs_indptr"_a, "lhs_n_rows"_a, "lhs_n_cols"_a,
      "rhs_n_rows"_a, "rhs_n_cols"_a,
      "Multiply two CSC matrices into a canonical CSC representation.");

  m.def(
      "csr_matmat",
      [](const mlx_sparse::mx::array &lhs_data,
         const mlx_sparse::mx::array &lhs_indices,
         const mlx_sparse::mx::array &lhs_indptr,
         const mlx_sparse::mx::array &rhs_data,
         const mlx_sparse::mx::array &rhs_indices,
         const mlx_sparse::mx::array &rhs_indptr, int lhs_n_rows,
         int lhs_n_cols, int rhs_n_rows, int rhs_n_cols) {
        return mlx_sparse::csr_matmat(
            lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices,
            rhs_indptr, lhs_n_rows, lhs_n_cols, rhs_n_rows, rhs_n_cols);
      },
      "lhs_data"_a, "lhs_indices"_a, "lhs_indptr"_a, "rhs_data"_a,
      "rhs_indices"_a, "rhs_indptr"_a, "lhs_n_rows"_a, "lhs_n_cols"_a,
      "rhs_n_rows"_a, "rhs_n_cols"_a,
      "Multiply two CSR matrices into a canonical CSR representation.");

  m.def(
      "csr_cg",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &b,
         const mlx_sparse::mx::array &x0, int n_rows, int n_cols, float rtol,
         float atol, int maxiter) {
        return mlx_sparse::csr_cg(data, indices, indptr, b, x0, n_rows, n_cols,
                                  rtol, atol, maxiter);
      },
      "data"_a, "indices"_a, "indptr"_a, "b"_a, "x0"_a, "n_rows"_a, "n_cols"_a,
      "rtol"_a, "atol"_a, "maxiter"_a,
      "Solve a float32 SPD CSR system with conjugate gradients.");

  m.def(
      "csr_lanczos",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &v0,
         int n_rows, int n_cols, int k, bool reorthogonalize) {
        return mlx_sparse::csr_lanczos(data, indices, indptr, v0, n_rows,
                                       n_cols, k, reorthogonalize);
      },
      "data"_a, "indices"_a, "indptr"_a, "v0"_a, "n_rows"_a, "n_cols"_a, "k"_a,
      "reorthogonalize"_a, "Run a float32 CSR Lanczos basis construction.");

  m.def(
      "csr_gmres",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &b,
         const mlx_sparse::mx::array &x0, int n_rows, int n_cols, float rtol,
         float atol, int restart, int maxiter) {
        return mlx_sparse::csr_gmres(data, indices, indptr, b, x0, n_rows,
                                     n_cols, rtol, atol, restart, maxiter);
      },
      "data"_a, "indices"_a, "indptr"_a, "b"_a, "x0"_a, "n_rows"_a, "n_cols"_a,
      "rtol"_a, "atol"_a, "restart"_a, "maxiter"_a,
      "Solve a float32 CSR system with restarted GMRES.");

  m.def(
      "csr_minres",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &b,
         const mlx_sparse::mx::array &x0, int n_rows, int n_cols, float rtol,
         float atol, int maxiter) {
        return mlx_sparse::csr_minres(data, indices, indptr, b, x0, n_rows,
                                      n_cols, rtol, atol, maxiter);
      },
      "data"_a, "indices"_a, "indptr"_a, "b"_a, "x0"_a, "n_rows"_a, "n_cols"_a,
      "rtol"_a, "atol"_a, "maxiter"_a,
      "Solve a float32 Hermitian CSR system with MINRES.");

  m.def(
      "csr_arnoldi",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &v0,
         int n_rows, int n_cols, int k) {
        return mlx_sparse::csr_arnoldi(data, indices, indptr, v0, n_rows,
                                       n_cols, k);
      },
      "data"_a, "indices"_a, "indptr"_a, "v0"_a, "n_rows"_a, "n_cols"_a, "k"_a,
      "Run a float32 CSR Arnoldi basis construction.");

  m.def(
      "csr_eigsh",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols, int k,
         int ncv, const std::string &which) {
        return mlx_sparse::csr_eigsh(data, indices, indptr, n_rows, n_cols, k,
                                     ncv, which);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a, "k"_a, "ncv"_a,
      "which"_a, "Compute selected Hermitian Ritz pairs from a CSR matrix.");

  m.def(
      "csr_eigs",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols, int k,
         int ncv, const std::string &which) {
        return mlx_sparse::csr_eigs(data, indices, indptr, n_rows, n_cols, k,
                                    ncv, which);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a, "k"_a, "ncv"_a,
      "which"_a, "Compute selected Arnoldi Ritz pairs from a CSR matrix.");

  m.def(
      "csr_svds",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols, int k,
         int ncv, const std::string &which) {
        return mlx_sparse::csr_svds(data, indices, indptr, n_rows, n_cols, k,
                                    ncv, which);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a, "k"_a, "ncv"_a,
      "which"_a, "Compute selected singular triplets from a CSR matrix.");

  m.def(
      "csr_normal_lanczos",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols, int k) {
        return mlx_sparse::csr_normal_lanczos(data, indices, indptr, n_rows,
                                              n_cols, k);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a, "k"_a,
      "Run Lanczos on the CSR normal operator A.T @ A.");

  m.def(
      "csr_cholesky",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_cholesky(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute a sparse left-looking Cholesky factor in CSR format.");

  m.def(
      "csr_lu",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_lu(data, indices, indptr, n_rows, n_cols);
      },
      "data"_a, "indices"_a, "indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute sparse LU factors with partial pivoting in CSR format.");

  m.def(
      "csr_triangular_solve",
      [](const mlx_sparse::mx::array &data,
         const mlx_sparse::mx::array &indices,
         const mlx_sparse::mx::array &indptr, const mlx_sparse::mx::array &b,
         int n_rows, int n_cols, bool lower, bool unit_diagonal) {
        return mlx_sparse::csr_triangular_solve(
            data, indices, indptr, b, n_rows, n_cols, lower, unit_diagonal);
      },
      "data"_a, "indices"_a, "indptr"_a, "b"_a, "n_rows"_a, "n_cols"_a,
      "lower"_a, "unit_diagonal"_a, "Solve a sparse triangular CSR system.");

  m.def(
      "csr_vdot",
      [](const mlx_sparse::mx::array &lhs_data,
         const mlx_sparse::mx::array &lhs_indices,
         const mlx_sparse::mx::array &lhs_indptr,
         const mlx_sparse::mx::array &rhs_data,
         const mlx_sparse::mx::array &rhs_indices,
         const mlx_sparse::mx::array &rhs_indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_vdot(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                                    rhs_indices, rhs_indptr, n_rows, n_cols);
      },
      "lhs_data"_a, "lhs_indices"_a, "lhs_indptr"_a, "rhs_data"_a,
      "rhs_indices"_a, "rhs_indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute the sparse Frobenius inner product of two CSR arrays.");

  m.def(
      "csr_dot",
      [](const mlx_sparse::mx::array &lhs_data,
         const mlx_sparse::mx::array &lhs_indices,
         const mlx_sparse::mx::array &lhs_indptr,
         const mlx_sparse::mx::array &rhs_data,
         const mlx_sparse::mx::array &rhs_indices,
         const mlx_sparse::mx::array &rhs_indptr, int n_rows, int n_cols) {
        return mlx_sparse::csr_dot(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                                   rhs_indices, rhs_indptr, n_rows, n_cols);
      },
      "lhs_data"_a, "lhs_indices"_a, "lhs_indptr"_a, "rhs_data"_a,
      "rhs_indices"_a, "rhs_indptr"_a, "n_rows"_a, "n_cols"_a,
      "Compute the sparse Frobenius dot product of two CSR arrays.");

  m.def(
      "csr_permute_vector",
      [](const mlx_sparse::mx::array &x, const mlx_sparse::mx::array &perm) {
        return mlx_sparse::csr_permute_vector(x, perm);
      },
      "x"_a, "perm"_a, "Apply an int32 permutation to a float32 vector.");
}
