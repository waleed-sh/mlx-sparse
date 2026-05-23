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

#include "linalg/triangular_solve/triangular_solve.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

#include "linalg/common/common.h"

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

class CSRTriangularSolve : public mx::Primitive {
public:
  CSRTriangularSolve(mx::Stream stream, int n_rows, int n_cols, bool lower,
                     bool unit_diagonal)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), lower_(lower),
        unit_diagonal_(unit_diagonal) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRTriangularSolve"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRTriangularSolve &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           lower_ == rhs.lower_ && unit_diagonal_ == rhs.unit_diagonal_;
  }

private:
  int n_rows_;
  int n_cols_;
  bool lower_;
  bool unit_diagonal_;
};

template <typename I>
void csr_triangular_solve_cpu_impl(const mx::array &data,
                                   const mx::array &indices,
                                   const mx::array &indptr, const mx::array &b,
                                   mx::array &x, int n_rows, bool lower,
                                   bool unit_diagonal, mx::Stream stream) {
  x.set_data(mx::allocator::malloc(x.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(b);
  encoder.set_output_array(x);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    b = mx::array::unsafe_weak_copy(b),
                    x = mx::array::unsafe_weak_copy(x), n_rows, lower,
                    unit_diagonal]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    auto *x_ptr = x.data<float>();

    if (lower) {
      for (int row = 0; row < n_rows; ++row) {
        float sum = b_ptr[row];
        float diag = 1.0f;
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          const int col = static_cast<int>(indices_ptr[p]);
          if (col < row) {
            sum -= data_ptr[p] * x_ptr[col];
          } else if (col == row) {
            diag = data_ptr[p];
          }
        }
        if (!unit_diagonal &&
            std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
          throw std::runtime_error(
              "csr_triangular_solve encountered a zero diagonal.");
        }
        x_ptr[row] = unit_diagonal ? sum : sum / diag;
      }
    } else {
      for (int row = n_rows - 1; row >= 0; --row) {
        float sum = b_ptr[row];
        float diag = 1.0f;
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          const int col = static_cast<int>(indices_ptr[p]);
          if (col > row) {
            sum -= data_ptr[p] * x_ptr[col];
          } else if (col == row) {
            diag = data_ptr[p];
          }
        }
        if (!unit_diagonal &&
            std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
          throw std::runtime_error(
              "csr_triangular_solve encountered a zero diagonal.");
        }
        x_ptr[row] = unit_diagonal ? sum : sum / diag;
      }
    }
  });
}

} // namespace

void CSRTriangularSolve::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];

  if (indices.dtype() == mx::int32) {
    csr_triangular_solve_cpu_impl<int32_t>(data, indices, indptr, b, outputs[0],
                                           n_rows_, lower_, unit_diagonal_,
                                           stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_triangular_solve_cpu_impl<int64_t>(data, indices, indptr, b, outputs[0],
                                           n_rows_, lower_, unit_diagonal_,
                                           stream());
    return;
  }
  throw std::runtime_error(
      "csr_triangular_solve requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRTriangularSolve::eval_gpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x = outputs[0];

  x.set_data(mx::allocator::malloc(x.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_triangular_solve", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(b, 3);
  encoder.set_output_array(x, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(n_cols_, 6);
  int lower = lower_ ? 1 : 0;
  int unit_diagonal = unit_diagonal_ ? 1 : 0;
  encoder.set_bytes(lower, 7);
  encoder.set_bytes(unit_diagonal, 8);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}
#else
void CSRTriangularSolve::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_triangular_solve has no GPU implementation in this build.");
}
#endif

mx::array csr_triangular_solve(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, const mx::array &b,
                               int n_rows, int n_cols, bool lower,
                               bool unit_diagonal, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_triangular_solve requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_triangular_solve data");
  require_rank(indices, 1, "csr_triangular_solve indices");
  require_rank(indptr, 1, "csr_triangular_solve indptr");
  require_rank(b, 1, "csr_triangular_solve b");
  require_linalg_float32(data, "csr_triangular_solve data");
  require_linalg_float32(b, "csr_triangular_solve b");
  require_same_index_dtype(indices, indptr, "csr_triangular_solve indices",
                           "csr_triangular_solve indptr");
  require_size(indptr, n_rows + 1, "csr_triangular_solve indptr");
  require_size(b, n_rows, "csr_triangular_solve b");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_triangular_solve data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);

  return mx::array(mx::Shape{n_rows}, mx::float32,
                   std::make_shared<CSRTriangularSolve>(stream, n_rows, n_cols,
                                                        lower, unit_diagonal),
                   {data_contig, indices_contig, indptr_contig, b_contig});
}

} // namespace mlx_sparse
