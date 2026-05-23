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

#include "linalg/arnoldi/arnoldi.h"

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

class CSRArnoldi : public mx::Primitive {
public:
  CSRArnoldi(mx::Stream stream, int n_rows, int n_cols, int k)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), k_(k) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRArnoldi"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRArnoldi &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ && k_ == rhs.k_;
  }

private:
  int n_rows_;
  int n_cols_;
  int k_;
};

template <typename I>
void csr_arnoldi_cpu_impl(const mx::array &data, const mx::array &indices,
                          const mx::array &indptr, const mx::array &v0,
                          mx::array &h, mx::array &basis, mx::array &actual,
                          int n_rows, int k, mx::Stream stream) {
  h.set_data(mx::allocator::malloc(h.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(v0);
  encoder.set_output_array(h);
  encoder.set_output_array(basis);
  encoder.set_output_array(actual);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    v0 = mx::array::unsafe_weak_copy(v0),
                    h = mx::array::unsafe_weak_copy(h),
                    basis = mx::array::unsafe_weak_copy(basis),
                    actual = mx::array::unsafe_weak_copy(actual), n_rows,
                    k]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *v0_ptr = v0.data<float>();
    auto *h_ptr = h.data<float>();
    auto *basis_ptr = basis.data<float>();
    auto *actual_ptr = actual.data<int32_t>();

    const int cols = k + 1;
    std::fill(h_ptr, h_ptr + static_cast<size_t>(cols) * k, 0.0f);
    std::fill(basis_ptr, basis_ptr + static_cast<size_t>(n_rows) * cols, 0.0f);

    double v_norm2 = 0.0;
    for (int i = 0; i < n_rows; ++i) {
      v_norm2 +=
          static_cast<double>(v0_ptr[i]) * static_cast<double>(v0_ptr[i]);
    }
    float v_norm = std::sqrt(std::max(v_norm2, 0.0));
    if (v_norm <= std::numeric_limits<float>::epsilon()) {
      for (int i = 0; i < n_rows; ++i) {
        basis_ptr[static_cast<size_t>(i) * cols] = i == 0 ? 1.0f : 0.0f;
      }
    } else {
      for (int i = 0; i < n_rows; ++i) {
        basis_ptr[static_cast<size_t>(i) * cols] = v0_ptr[i] / v_norm;
      }
    }

    std::vector<float> w(static_cast<size_t>(n_rows));
    std::vector<float> q(static_cast<size_t>(n_rows));
    int used = 0;
    const float eps = std::numeric_limits<float>::epsilon();
    for (int j = 0; j < k; ++j) {
      for (int i = 0; i < n_rows; ++i) {
        q[i] = basis_ptr[static_cast<size_t>(i) * cols + j];
      }
      csr_spmv_float(data_ptr, indices_ptr, indptr_ptr, q.data(), w.data(),
                     n_rows);
      for (int pass = 0; pass < 2; ++pass) {
        for (int col = 0; col <= j; ++col) {
          double coeff = 0.0;
          for (int row = 0; row < n_rows; ++row) {
            coeff += basis_ptr[static_cast<size_t>(row) * cols + col] * w[row];
          }
          h_ptr[static_cast<size_t>(col) * k + j] += static_cast<float>(coeff);
          for (int row = 0; row < n_rows; ++row) {
            w[row] -= static_cast<float>(coeff) *
                      basis_ptr[static_cast<size_t>(row) * cols + col];
          }
        }
      }
      float h_next = norm_float(w);
      h_ptr[static_cast<size_t>(j + 1) * k + j] = h_next;
      used = j + 1;
      if (h_next <= eps) {
        break;
      }
      for (int row = 0; row < n_rows; ++row) {
        basis_ptr[static_cast<size_t>(row) * cols + j + 1] = w[row] / h_next;
      }
    }
    *actual_ptr = used;
  });
}

} // namespace

void CSRArnoldi::eval_cpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &v0 = inputs[3];

  if (indices.dtype() == mx::int32) {
    csr_arnoldi_cpu_impl<int32_t>(data, indices, indptr, v0, outputs[0],
                                  outputs[1], outputs[2], n_rows_, k_,
                                  stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_arnoldi_cpu_impl<int64_t>(data, indices, indptr, v0, outputs[0],
                                  outputs[1], outputs[2], n_rows_, k_,
                                  stream());
    return;
  }
  throw std::runtime_error("csr_arnoldi requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRArnoldi::eval_gpu(const std::vector<mx::array> &inputs,
                          std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &v0 = inputs[3];
  auto &h = outputs[0];
  auto &basis = outputs[1];
  auto &actual = outputs[2];

  h.set_data(mx::allocator::malloc(h.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));
  mx::array work(
      mx::allocator::malloc(static_cast<size_t>(n_rows_) * sizeof(float)),
      mx::Shape{n_rows_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_arnoldi", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(v0, 3);
  encoder.set_output_array(h, 4);
  encoder.set_output_array(basis, 5);
  encoder.set_output_array(actual, 6);
  encoder.set_output_array(work, 7);
  encoder.set_bytes(n_rows_, 8);
  encoder.set_bytes(n_cols_, 9);
  encoder.set_bytes(k_, 10);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(work));
}
#else
void CSRArnoldi::eval_gpu(const std::vector<mx::array> &,
                          std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_arnoldi has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array>
csr_arnoldi(const mx::array &data, const mx::array &indices,
            const mx::array &indptr, const mx::array &v0, int n_rows,
            int n_cols, int k, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_arnoldi requires a non-empty square matrix.");
  }
  if (k <= 0 || k > n_rows) {
    throw std::invalid_argument("csr_arnoldi k must satisfy 0 < k <= n_rows.");
  }
  require_rank(data, 1, "csr_arnoldi data");
  require_rank(indices, 1, "csr_arnoldi indices");
  require_rank(indptr, 1, "csr_arnoldi indptr");
  require_rank(v0, 1, "csr_arnoldi v0");
  require_linalg_float32(data, "csr_arnoldi data");
  require_linalg_float32(v0, "csr_arnoldi v0");
  require_same_index_dtype(indices, indptr, "csr_arnoldi indices",
                           "csr_arnoldi indptr");
  require_size(indptr, n_rows + 1, "csr_arnoldi indptr");
  require_size(v0, n_rows, "csr_arnoldi v0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_arnoldi data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto v0_contig = mx::contiguous(v0, false, stream);

  auto primitive = std::make_shared<CSRArnoldi>(stream, n_rows, n_cols, k);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{k + 1, k}, mx::Shape{n_rows, k + 1}, mx::Shape{}},
      {mx::float32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, v0_contig});
  return {outputs[0], outputs[1], outputs[2]};
}

} // namespace mlx_sparse
