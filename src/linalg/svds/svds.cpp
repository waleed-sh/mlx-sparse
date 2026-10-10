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

#include "linalg/svds/svds.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "linalg/svds/bidiagonal.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/linalg.h"
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

class CSRNormalLanczos : public mx::Primitive {
public:
  CSRNormalLanczos(mx::Stream stream, int n_rows, int n_cols, int k,
                   bool complete_basis)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), k_(k),
        complete_basis_(complete_basis) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRNormalLanczos"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRNormalLanczos &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ && k_ == rhs.k_ &&
           complete_basis_ == rhs.complete_basis_;
  }

private:
  int n_rows_;
  int n_cols_;
  int k_;
  bool complete_basis_;
};

template <typename I>
void csr_normal_apply_fused_float(const float *data, const I *indices,
                                  const I *indptr, const float *basis,
                                  int basis_stride, int basis_col, float *out,
                                  int n_rows, int n_cols) {
  std::fill(out, out + n_cols, 0.0f);
  for (int row = 0; row < n_rows; ++row) {
    float ax = 0.0f;
    const I start = indptr[row];
    const I end = indptr[row + 1];
    for (I p = start; p < end; ++p) {
      ax += data[p] *
            basis[static_cast<size_t>(indices[p]) * basis_stride + basis_col];
    }
    if (ax == 0.0f) {
      continue;
    }
    for (I p = start; p < end; ++p) {
      out[static_cast<size_t>(indices[p])] += data[p] * ax;
    }
  }
}

template <typename I>
void csr_normal_lanczos_cpu_impl(const mx::array &data,
                                 const mx::array &indices,
                                 const mx::array &indptr, const mx::array &v0,
                                 mx::array &alphas, mx::array &betas,
                                 mx::array &basis, mx::array &actual,
                                 int n_rows, int n_cols, int k,
                                 bool complete_basis, mx::Stream stream) {
  alphas.set_data(mx::allocator::malloc(alphas.nbytes()));
  betas.set_data(mx::allocator::malloc(betas.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(v0);
  encoder.set_output_array(alphas);
  encoder.set_output_array(betas);
  encoder.set_output_array(basis);
  encoder.set_output_array(actual);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    v0 = mx::array::unsafe_weak_copy(v0),
                    alphas = mx::array::unsafe_weak_copy(alphas),
                    betas = mx::array::unsafe_weak_copy(betas),
                    basis = mx::array::unsafe_weak_copy(basis),
                    actual = mx::array::unsafe_weak_copy(actual), n_rows,
                    n_cols, k, complete_basis]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *v0_ptr = v0.data<float>();
    auto *alphas_ptr = alphas.data<float>();
    auto *betas_ptr = betas.data<float>();
    auto *basis_ptr = basis.data<float>();
    auto *actual_ptr = actual.data<int32_t>();

    std::fill(alphas_ptr, alphas_ptr + k, 0.0f);
    std::fill(betas_ptr, betas_ptr + k, 0.0f);
    std::fill(basis_ptr, basis_ptr + static_cast<size_t>(n_cols) * k, 0.0f);

    if (!initialize_krylov_basis(v0_ptr, basis_ptr, n_cols, k)) {
      *actual_ptr = 0;
      return;
    }

    std::vector<float> w(static_cast<size_t>(n_cols), 0.0f);
    float beta_prev = 0.0f;
    int used = 0;
    const float eps = std::numeric_limits<float>::epsilon();
    for (int j = 0; j < k; ++j) {
      csr_normal_apply_fused_float(data_ptr, indices_ptr, indptr_ptr, basis_ptr,
                                   k, j, w.data(), n_rows, n_cols);
      const float operator_norm = norm_float(w);
      if (j > 0) {
        for (int col = 0; col < n_cols; ++col) {
          w[static_cast<size_t>(col)] -=
              beta_prev * basis_ptr[static_cast<size_t>(col) * k + j - 1];
        }
      }

      const float alpha = dot_column_float(basis_ptr, w.data(), n_cols, k, j);
      alphas_ptr[j] = alpha;
      for (int col = 0; col < n_cols; ++col) {
        w[static_cast<size_t>(col)] -=
            alpha * basis_ptr[static_cast<size_t>(col) * k + j];
      }

      for (int pass = 0; pass < 2; ++pass) {
        for (int orth_col = 0; orth_col <= j; ++orth_col) {
          const float correction =
              dot_column_float(basis_ptr, w.data(), n_cols, k, orth_col);
          for (int col = 0; col < n_cols; ++col) {
            w[static_cast<size_t>(col)] -=
                correction * basis_ptr[static_cast<size_t>(col) * k + orth_col];
          }
        }
      }

      const float beta = norm_float(w);
      betas_ptr[j] = beta;
      used = j + 1;
      if (j + 1 == k) {
        break;
      }
      if (beta <= 8 * eps * operator_norm) {
        betas_ptr[j] = 0.0f;
        beta_prev = 0.0f;
        if (!complete_basis) {
          break;
        }
        if (!restart_krylov_basis(basis_ptr, w, n_cols, k, j + 1)) {
          used = -used;
          break;
        }
        continue;
      }
      for (int col = 0; col < n_cols; ++col) {
        basis_ptr[static_cast<size_t>(col) * k + j + 1] =
            w[static_cast<size_t>(col)] / beta;
      }
      beta_prev = beta;
    }
    *actual_ptr = used;
  });
}

std::tuple<mx::array, mx::array, mx::array>
csr_svds_impl(const mx::array &data, const mx::array &indices,
              const mx::array &indptr, int n_rows, int n_cols,
              const mx::array &v0, int k, int ncv, const std::string &which,
              bool compute_u, bool compute_v) {
  const int steps = std::min({n_rows, n_cols, std::max(ncv, k + 1)});
  auto stream = mx::default_stream(mx::default_device());
  auto cpu = mx::default_stream(mx::Device(mx::Device::cpu, 0));
  auto [left_t, right_t, images_t, actual] = csr_bidiagonal_basis(
      data, indices, indptr, v0, n_rows, n_cols, steps, stream);
  const int used = checked_krylov_dimension(actual.item<int32_t>());
  if (used < k) {
    throw std::runtime_error("Krylov basis dimension " + std::to_string(used) +
                             " is smaller than the requested number of pairs.");
  }
  // Retain all projection corrections and continuation couplings using
  // sparse products already computed while building the bases.
  auto projected = mx::matmul(left_t, mx::transpose(images_t, stream), stream);
  const bool vectors = compute_u || compute_v;
  auto small = mx::linalg::svd(projected, vectors, cpu);
  auto singular = vectors ? small[1] : small[0];
  std::vector<int32_t> order(k);
  for (int i = 0; i < k; ++i) {
    order[i] = which == "LM" ? i : steps - 1 - i;
  }
  auto selected = mx::array(order.begin(), {k}, mx::int32);
  auto s = mx::take(singular, selected, 0, stream);
  auto u = mx::zeros({n_rows, 0}, mx::float32, stream);
  auto vh = mx::zeros({0, n_cols}, mx::float32, stream);
  if (compute_u) {
    auto selected_u = mx::take(small[0], selected, 1, stream);
    u = mx::matmul(mx::transpose(left_t, stream), selected_u, stream);
  }
  if (compute_v) {
    auto selected_vh = mx::take(small[2], selected, 0, stream);
    vh = mx::matmul(selected_vh, right_t, stream);
  }
  return {u, s, vh};
}

} // namespace

void CSRNormalLanczos::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &v0 = inputs[3];

  if (indices.dtype() == mx::int32) {
    csr_normal_lanczos_cpu_impl<int32_t>(
        data, indices, indptr, v0, outputs[0], outputs[1], outputs[2],
        outputs[3], n_rows_, n_cols_, k_, complete_basis_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_normal_lanczos_cpu_impl<int64_t>(
        data, indices, indptr, v0, outputs[0], outputs[1], outputs[2],
        outputs[3], n_rows_, n_cols_, k_, complete_basis_, stream());
    return;
  }
  throw std::runtime_error(
      "csr_normal_lanczos requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRNormalLanczos::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &v0 = inputs[3];
  auto &alphas = outputs[0];
  auto &betas = outputs[1];
  auto &basis = outputs[2];
  auto &actual = outputs[3];

  alphas.set_data(mx::allocator::malloc(alphas.nbytes()));
  betas.set_data(mx::allocator::malloc(betas.nbytes()));
  basis.set_data(mx::allocator::malloc(basis.nbytes()));
  actual.set_data(mx::allocator::malloc(actual.nbytes()));
  mx::array work(
      mx::allocator::malloc(static_cast<size_t>(n_cols_) * sizeof(float)),
      mx::Shape{n_cols_}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_normal_lanczos", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(v0, 3);
  encoder.set_output_array(alphas, 4);
  encoder.set_output_array(betas, 5);
  encoder.set_output_array(basis, 6);
  encoder.set_output_array(actual, 7);
  encoder.set_output_array(work, 8);
  encoder.set_bytes(n_rows_, 9);
  encoder.set_bytes(n_cols_, 10);
  encoder.set_bytes(k_, 11);
  int complete = complete_basis_ ? 1 : 0;
  encoder.set_bytes(complete, 12);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));

  encoder.add_temporary(std::move(work));
}
#else
void CSRNormalLanczos::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_normal_lanczos has no GPU implementation in this build.");
}
#endif

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_normal_lanczos(const mx::array &data, const mx::array &indices,
                   const mx::array &indptr, const mx::array &v0, int n_rows,
                   int n_cols, int k, mx::StreamOrDevice s,
                   bool complete_basis) {
  if (n_rows <= 0 || n_cols <= 0) {
    throw std::invalid_argument(
        "csr_normal_lanczos requires a non-empty matrix.");
  }
  if (k <= 0 || k > n_cols) {
    throw std::invalid_argument(
        "csr_normal_lanczos k must satisfy 0 < k <= n_cols.");
  }
  require_rank(data, 1, "csr_normal_lanczos data");
  require_rank(indices, 1, "csr_normal_lanczos indices");
  require_rank(indptr, 1, "csr_normal_lanczos indptr");
  require_rank(v0, 1, "csr_normal_lanczos v0");
  require_linalg_float32(data, "csr_normal_lanczos data");
  require_linalg_float32(v0, "csr_normal_lanczos v0");
  require_same_index_dtype(indices, indptr, "csr_normal_lanczos indices",
                           "csr_normal_lanczos indptr");
  require_size(indptr, n_rows + 1, "csr_normal_lanczos indptr");
  require_size(v0, n_cols, "csr_normal_lanczos v0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_normal_lanczos data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto v0_contig = mx::contiguous(v0, false, stream);

  auto primitive = std::make_shared<CSRNormalLanczos>(stream, n_rows, n_cols, k,
                                                      complete_basis);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{k}, mx::Shape{k}, mx::Shape{n_cols, k}, mx::Shape{}},
      {mx::float32, mx::float32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, v0_contig});
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

std::tuple<mx::array, mx::array, mx::array>
csr_svds(const mx::array &data, const mx::array &indices,
         const mx::array &indptr, const mx::array &v0, int n_rows, int n_cols,
         int k, int ncv, const std::string &which, bool compute_u,
         bool compute_v) {
  linalg_detail::require_spectral_which(which, "csr_svds", {"LM", "SM"});
  if (n_rows <= 0 || n_cols <= 0) {
    throw std::invalid_argument("csr_svds requires a non-empty matrix.");
  }
  if (k <= 0 || k >= std::min(n_rows, n_cols)) {
    throw std::invalid_argument("csr_svds k must satisfy 0 < k < min(shape).");
  }
  require_rank(data, 1, "csr_svds data");
  require_rank(indices, 1, "csr_svds indices");
  require_rank(indptr, 1, "csr_svds indptr");
  require_rank(v0, 1, "csr_svds v0");
  require_linalg_float32(data, "csr_svds data");
  require_linalg_float32(v0, "csr_svds v0");
  require_same_index_dtype(indices, indptr, "csr_svds indices",
                           "csr_svds indptr");
  require_size(indptr, n_rows + 1, "csr_svds indptr");
  require_size(v0, n_cols, "csr_svds v0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_svds data and indices must have equal length.");
  }
  return csr_svds_impl(data, indices, indptr, n_rows, n_cols, v0, k, ncv, which,
                       compute_u, compute_v);
}

} // namespace mlx_sparse
