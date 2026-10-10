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

#include "linalg/eigs/eigs.h"

#include <algorithm>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "linalg/arnoldi/arnoldi.h"
#include "linalg/common/common.h"
#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

std::tuple<mx::array, mx::array>
csr_eigs_impl(const mx::array &data, const mx::array &indices,
              const mx::array &indptr, int n_rows, const mx::array &v0, int k,
              int ncv, const std::string &which, bool compute_vectors) {
  const int steps = std::min(n_rows, std::max(ncv, k + 1));
  auto stream = mx::default_stream(mx::default_device());
  auto cpu = mx::default_stream(mx::Device(mx::Device::cpu, 0));
  auto [h, basis, actual] = csr_arnoldi(data, indices, indptr, v0, n_rows,
                                        n_rows, steps, stream, true);
  const int used = checked_krylov_dimension(actual.item<int32_t>());
  if (used < k) {
    throw std::runtime_error("Krylov basis dimension " + std::to_string(used) +
                             " is smaller than the requested number of pairs.");
  }
  auto projected = mx::slice(h, {0, 0}, {used, used}, cpu);
  mx::array values = mx::zeros({used}, mx::complex64, cpu);
  mx::array small_vectors = mx::zeros({0, 0}, mx::complex64, cpu);
  if (compute_vectors) {
    auto eigen = mx::linalg::eig(projected, cpu);
    values = eigen.first;
    small_vectors = eigen.second;
  } else {
    values = mx::linalg::eigvals(projected, cpu);
  }
  values.eval();
  const auto *ptr = values.data<mx::complex64_t>();
  std::vector<int32_t> order(used);
  std::iota(order.begin(), order.end(), 0);
  auto score = [&](int i) {
    const auto value = ptr[i];
    return which == "LM" || which == "SM" ? std::abs(value) : value.real();
  };
  // Stable ordering keeps tied conjugate pairs in the dense solver's order.
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return which == "SM" || which == "SR" ? score(a) < score(b)
                                          : score(a) > score(b);
  });
  order.resize(k);
  auto selected = mx::array(order.begin(), {k}, mx::int32);
  auto selected_values = mx::take(values, selected, 0, stream);
  if (!compute_vectors) {
    return {selected_values, mx::zeros({n_rows, 0}, mx::complex64, stream)};
  }
  auto y = mx::take(small_vectors, selected, 1, stream);
  auto q = mx::slice(basis, {0, 0}, {n_rows, used}, stream);
  // Two real GEMMs avoid a complex copy of the entire Arnoldi basis.
  auto real = mx::matmul(q, mx::real(y, stream), stream);
  auto imag = mx::matmul(q, mx::imag(y, stream), stream);
  auto vectors = mx::add(
      real, mx::multiply(imag, mx::array(mx::complex64_t(0, 1)), stream),
      stream);
  auto norms = mx::linalg::norm(vectors, 0, true, stream);
  return {selected_values, mx::divide(vectors, norms, stream)};
}

} // namespace

std::tuple<mx::array, mx::array>
csr_eigs(const mx::array &data, const mx::array &indices,
         const mx::array &indptr, const mx::array &v0, int n_rows, int n_cols,
         int k, int ncv, const std::string &which, bool compute_vectors) {
  linalg_detail::require_spectral_which(which, "csr_eigs",
                                        {"LM", "SM", "LR", "SR"});
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_eigs requires a non-empty square matrix.");
  }
  if (k <= 0 || k >= n_rows) {
    throw std::invalid_argument("csr_eigs k must satisfy 0 < k < n_rows.");
  }
  require_rank(data, 1, "csr_eigs data");
  require_rank(indices, 1, "csr_eigs indices");
  require_rank(indptr, 1, "csr_eigs indptr");
  require_rank(v0, 1, "csr_eigs v0");
  require_linalg_float32(data, "csr_eigs data");
  require_linalg_float32(v0, "csr_eigs v0");
  require_same_index_dtype(indices, indptr, "csr_eigs indices",
                           "csr_eigs indptr");
  require_size(indptr, n_rows + 1, "csr_eigs indptr");
  require_size(v0, n_rows, "csr_eigs v0");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_eigs data and indices must have equal length.");
  }
  ncv = std::min(n_rows, std::max(ncv, k + 1));
  return csr_eigs_impl(data, indices, indptr, n_rows, v0, k, ncv, which,
                       compute_vectors);
}

} // namespace mlx_sparse
