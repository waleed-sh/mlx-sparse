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

class CSRNormalLanczos : public mx::Primitive {
public:
  CSRNormalLanczos(mx::Stream stream, int n_rows, int n_cols, int k)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), k_(k) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRNormalLanczos"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRNormalLanczos &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ && k_ == rhs.k_;
  }

private:
  int n_rows_;
  int n_cols_;
  int k_;
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
void csr_normal_lanczos_cpu_impl(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &v0, mx::array &alphas, mx::array &betas, mx::array &basis,
    mx::array &actual, int n_rows, int n_cols, int k, mx::Stream stream) {
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
                    n_cols, k]() mutable {
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

    double norm_sq = 0.0;
    for (int col = 0; col < n_cols; ++col) {
      norm_sq += static_cast<double>(v0_ptr[col]) * v0_ptr[col];
    }
    const float norm = static_cast<float>(std::sqrt(norm_sq));
    if (norm <= std::numeric_limits<float>::epsilon()) {
      basis_ptr[0] = 1.0f;
    } else {
      const float inv_norm = 1.0f / norm;
      for (int col = 0; col < n_cols; ++col) {
        basis_ptr[static_cast<size_t>(col) * k] = v0_ptr[col] * inv_norm;
      }
    }

    std::vector<float> w(static_cast<size_t>(n_cols), 0.0f);
    float beta_prev = 0.0f;
    int used = 0;
    const float eps = std::numeric_limits<float>::epsilon();
    for (int j = 0; j < k; ++j) {
      csr_normal_apply_fused_float(data_ptr, indices_ptr, indptr_ptr, basis_ptr,
                                   k, j, w.data(), n_rows, n_cols);
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
      if (j + 1 == k || beta <= eps) {
        break;
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

template <typename I>
std::tuple<mx::array, mx::array, mx::array>
csr_svds_impl(mx::array data, mx::array indices, mx::array indptr, int n_rows,
              int n_cols, const mx::array &v0, int k, int ncv,
              const std::string &which) {
  const int steps = std::min(n_cols, std::max(ncv, k + 1));
  auto stream = mx::default_stream(mx::default_device());
  auto v0_contig = mx::contiguous(v0, false, stream);
  auto [alphas_mx, betas_mx, basis_mx, actual_k_mx] = csr_normal_lanczos(
      data, indices, indptr, v0_contig, n_rows, n_cols, steps, stream);
  mx::eval(alphas_mx, betas_mx, basis_mx, actual_k_mx);

  const int used = static_cast<int>(actual_k_mx.item<int32_t>());
  const float *alphas_ptr = alphas_mx.data<float>();
  const float *betas_ptr = betas_mx.data<float>();
  const float *basis_ptr = basis_mx.data<float>();

  std::vector<float> tridiagonal(static_cast<size_t>(used) * used, 0.0f);
  for (int i = 0; i < used; ++i) {
    tridiagonal[static_cast<size_t>(i) * used + i] = alphas_ptr[i];
    if (i > 0) {
      tridiagonal[static_cast<size_t>(i) * used + i - 1] = betas_ptr[i - 1];
      tridiagonal[static_cast<size_t>(i - 1) * used + i] = betas_ptr[i - 1];
    }
  }
  auto [evals_all, vecs_small] = jacobi_symmetric(tridiagonal, used);
  auto selected = select_ritz_indices(evals_all, k, which);
  std::vector<float> singular(static_cast<size_t>(k), 0.0f);
  std::vector<float> right(static_cast<size_t>(n_cols) * k, 0.0f);
  std::vector<float> left(static_cast<size_t>(n_rows) * k, 0.0f);

  data.eval();
  indices.eval();
  indptr.eval();
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();

  for (int out_col = 0; out_col < k; ++out_col) {
    const int eig_col = selected[static_cast<size_t>(out_col)];
    const float sigma =
        std::sqrt(std::max(evals_all[static_cast<size_t>(eig_col)], 0.0f));
    singular[static_cast<size_t>(out_col)] = sigma;
    std::vector<float> v(static_cast<size_t>(n_cols), 0.0f);
    for (int row = 0; row < n_cols; ++row) {
      double acc = 0.0;
      for (int j = 0; j < used; ++j) {
        acc += basis_ptr[static_cast<size_t>(row) * steps + j] *
               vecs_small[static_cast<size_t>(j) * used + eig_col];
      }
      v[static_cast<size_t>(row)] = static_cast<float>(acc);
      right[static_cast<size_t>(out_col) * n_cols + row] =
          static_cast<float>(acc);
    }
    auto av = host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, v, n_rows);
    for (int row = 0; row < n_rows; ++row) {
      left[static_cast<size_t>(row) * k + out_col] =
          sigma <= std::numeric_limits<float>::epsilon()
              ? 0.0f
              : av[static_cast<size_t>(row)] / sigma;
    }
  }
  return {mx::array(left.begin(), mx::Shape{n_rows, k}, mx::float32),
          mx::array(singular.begin(), mx::Shape{k}, mx::float32),
          mx::array(right.begin(), mx::Shape{k, n_cols}, mx::float32)};
}

} // namespace

void CSRNormalLanczos::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &v0 = inputs[3];

  if (indices.dtype() == mx::int32) {
    csr_normal_lanczos_cpu_impl<int32_t>(data, indices, indptr, v0, outputs[0],
                                         outputs[1], outputs[2], outputs[3],
                                         n_rows_, n_cols_, k_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    csr_normal_lanczos_cpu_impl<int64_t>(data, indices, indptr, v0, outputs[0],
                                         outputs[1], outputs[2], outputs[3],
                                         n_rows_, n_cols_, k_, stream());
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
                   int n_cols, int k, mx::StreamOrDevice s) {
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

  auto primitive =
      std::make_shared<CSRNormalLanczos>(stream, n_rows, n_cols, k);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{k}, mx::Shape{k}, mx::Shape{n_cols, k}, mx::Shape{}},
      {mx::float32, mx::float32, mx::float32, mx::int32}, primitive,
      {data_contig, indices_contig, indptr_contig, v0_contig});
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

std::tuple<mx::array, mx::array, mx::array>
csr_svds(const mx::array &data, const mx::array &indices,
         const mx::array &indptr, const mx::array &v0, int n_rows, int n_cols,
         int k, int ncv, const std::string &which) {
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
  ncv = std::min(n_cols, std::max(ncv, k + 1));
  if (indices.dtype() == mx::int32) {
    return csr_svds_impl<int32_t>(data, indices, indptr, n_rows, n_cols, v0, k,
                                  ncv, which);
  }
  if (indices.dtype() == mx::int64) {
    return csr_svds_impl<int64_t>(data, indices, indptr, n_rows, n_cols, v0, k,
                                  ncv, which);
  }
  throw std::runtime_error("csr_svds requires int32 or int64 indices.");
}

} // namespace mlx_sparse
