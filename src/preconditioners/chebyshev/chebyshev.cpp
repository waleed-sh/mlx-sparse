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

#include "preconditioners/chebyshev/chebyshev.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cpu_parallel.h"
#include "linalg/common/common.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

constexpr int kDefaultEstimateSteps = 20;

class CSRChebyshevApply : public mx::Primitive {
public:
  CSRChebyshevApply(mx::Stream stream, int n_rows, int n_cols, int rhs_cols,
                    int degree, float lambda_min, float lambda_max)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols),
        rhs_cols_(rhs_cols), degree_(degree), lambda_min_(lambda_min),
        lambda_max_(lambda_max) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRChebyshevApply"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRChebyshevApply &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           rhs_cols_ == rhs.rhs_cols_ && degree_ == rhs.degree_ &&
           lambda_min_ == rhs.lambda_min_ && lambda_max_ == rhs.lambda_max_;
  }

private:
  int n_rows_;
  int n_cols_;
  int rhs_cols_;
  int degree_;
  float lambda_min_;
  float lambda_max_;
};

inline bool finite_float(float value) { return std::isfinite(value); }

void validate_chebyshev_interval(int degree, float lambda_min,
                                 float lambda_max) {
  if (degree <= 0) {
    throw std::invalid_argument("Chebyshev degree must be positive.");
  }
  if (!finite_float(lambda_min) || !finite_float(lambda_max) ||
      lambda_min <= 0.0f || lambda_max <= lambda_min) {
    throw std::invalid_argument(
        "Chebyshev spectral interval must satisfy 0 < lambda_min < "
        "lambda_max with finite bounds.");
  }
}

template <typename I>
void validate_csr_structure(mx::array data, mx::array indices, mx::array indptr,
                            int n_rows, int n_cols, const char *context) {
  data.eval();
  indices.eval();
  indptr.eval();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  const size_t nnz = data.size();

  if (indptr_ptr[0] != I{0} || indptr_ptr[n_rows] < I{0} ||
      static_cast<size_t>(indptr_ptr[n_rows]) != nnz) {
    throw std::invalid_argument(std::string(context) +
                                " has invalid CSR row offsets.");
  }
  for (int row = 0; row < n_rows; ++row) {
    const I begin = indptr_ptr[row];
    const I end = indptr_ptr[row + 1];
    if (begin > end || begin < I{0} || static_cast<size_t>(end) > nnz) {
      throw std::invalid_argument(std::string(context) +
                                  " has invalid CSR row offsets.");
    }
    for (I p = begin; p < end; ++p) {
      const I col = indices_ptr[p];
      if (col < I{0} || col >= static_cast<I>(n_cols)) {
        throw std::invalid_argument(std::string(context) +
                                    " contains an out-of-bounds column.");
      }
    }
  }
}

template <typename I>
std::tuple<float, float, float, float>
gershgorin_and_diagonal_bounds(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, int n_rows,
                               int n_cols) {
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();

  double lower = std::numeric_limits<double>::infinity();
  double upper = -std::numeric_limits<double>::infinity();
  double diag_min = std::numeric_limits<double>::infinity();
  double diag_max = -std::numeric_limits<double>::infinity();

  for (int row = 0; row < n_rows; ++row) {
    double diag = 0.0;
    double off_abs_sum = 0.0;
    bool has_diag = false;
    for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
      const float value = data_ptr[p];
      if (!finite_float(value)) {
        throw std::invalid_argument(
            "csr_chebyshev_spectral_bounds input contains a non-finite value.");
      }
      const int col = static_cast<int>(indices_ptr[p]);
      if (col == row) {
        diag += static_cast<double>(value);
        has_diag = true;
      } else {
        off_abs_sum += std::abs(static_cast<double>(value));
      }
    }
    if (!has_diag) {
      diag = 0.0;
    }
    lower = std::min(lower, diag - off_abs_sum);
    upper = std::max(upper, diag + off_abs_sum);
    diag_min = std::min(diag_min, diag);
    diag_max = std::max(diag_max, diag);
  }

  (void)n_cols;
  return {static_cast<float>(lower), static_cast<float>(upper),
          static_cast<float>(diag_min), static_cast<float>(diag_max)};
}

template <typename I>
std::tuple<float, float, int>
lanczos_ritz_bounds(const mx::array &data, const mx::array &indices,
                    const mx::array &indptr, int n_rows, int requested_steps) {
  if (requested_steps <= 0 || n_rows <= 0) {
    return {std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(), 0};
  }
  const auto *data_ptr = data.data<float>();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  const int steps = std::min(n_rows, requested_steps);
  auto [tridiagonal, basis, used] =
      host_lanczos_operator(n_rows, steps, [&](const std::vector<float> &x) {
        return host_csr_spmv(data_ptr, indices_ptr, indptr_ptr, x, n_rows);
      });
  (void)basis;
  if (used <= 0) {
    return {std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(), 0};
  }
  auto [values, vectors] = jacobi_symmetric(std::move(tridiagonal), used);
  (void)vectors;
  float ritz_min = std::numeric_limits<float>::infinity();
  float ritz_max = -std::numeric_limits<float>::infinity();
  for (float value : values) {
    if (!finite_float(value)) {
      continue;
    }
    ritz_min = std::min(ritz_min, value);
    ritz_max = std::max(ritz_max, value);
  }
  if (!finite_float(ritz_min) || !finite_float(ritz_max)) {
    return {std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(), used};
  }
  return {ritz_min, ritz_max, used};
}

template <typename I>
void csr_spmm_float_rows(const float *data_ptr, const I *indices_ptr,
                         const I *indptr_ptr, const float *rhs_ptr,
                         float *out_ptr, CpuRange range, int rhs_cols) {
  for (int row = range.begin; row < range.end; ++row) {
    const size_t row_offset = static_cast<size_t>(row) * rhs_cols;
    for (int col = 0; col < rhs_cols; ++col) {
      out_ptr[row_offset + col] = 0.0f;
    }
    for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
      const float value = data_ptr[p];
      const size_t rhs_offset = static_cast<size_t>(indices_ptr[p]) * rhs_cols;
      for (int col = 0; col < rhs_cols; ++col) {
        out_ptr[row_offset + col] += value * rhs_ptr[rhs_offset + col];
      }
    }
  }
}

template <typename I>
void chebyshev_apply_cpu_impl(const mx::array &data, const mx::array &indices,
                              const mx::array &indptr, const mx::array &rhs,
                              mx::array &out, int n_rows, int rhs_cols,
                              int degree, float lambda_min, float lambda_max,
                              mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(rhs);
  encoder.set_output_array(out);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    rhs = mx::array::unsafe_weak_copy(rhs),
                    out = mx::array::unsafe_weak_copy(out), n_rows, rhs_cols,
                    degree, lambda_min, lambda_max]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *rhs_ptr = rhs.data<float>();
    auto *out_ptr = out.data<float>();
    const int total = n_rows * rhs_cols;

    std::vector<float> x_prev(static_cast<size_t>(total), 0.0f);
    std::vector<float> x_next(static_cast<size_t>(total), 0.0f);
    std::vector<float> ax(static_cast<size_t>(total), 0.0f);

    const float scale = 2.0f / (lambda_max + lambda_min);
    const float alpha = 1.0f - scale * lambda_min;
    const float mu = 1.0f / alpha;
    const float omega_prod = 2.0f / alpha;
    float c_prev = 1.0f;
    float c_cur = mu;

    for (int i = 0; i < total; ++i) {
      out_ptr[i] = scale * rhs_ptr[i];
    }
    if (degree == 1) {
      return;
    }

    const int requested_workers = configured_cpu_worker_count();
    const auto ranges = cpu_ranges_for_compressed_segments(indptr_ptr, n_rows,
                                                           requested_workers);

    for (int it = 1; it < degree; ++it) {
      auto spmm_rows = [&](CpuRange range) {
        csr_spmm_float_rows(data_ptr, indices_ptr, indptr_ptr, out_ptr,
                            ax.data(), range, rhs_cols);
      };
      parallel_for_cpu_ranges(ranges, spmm_rows);

      const float c_next = 2.0f * mu * c_cur - c_prev;
      const float omega = omega_prod * c_cur / c_next;
      const float one_minus_omega = 1.0f - omega;
      const float omega_scale = omega * scale;
      for (int i = 0; i < total; ++i) {
        const float r = rhs_ptr[i] - ax[static_cast<size_t>(i)];
        x_next[static_cast<size_t>(i)] =
            one_minus_omega * x_prev[static_cast<size_t>(i)] +
            omega * out_ptr[i] + omega_scale * r;
      }
      std::copy(out_ptr, out_ptr + total, x_prev.begin());
      std::copy(x_next.begin(), x_next.end(), out_ptr);
      c_prev = c_cur;
      c_cur = c_next;
    }
  });
}

} // namespace

void CSRChebyshevApply::eval_cpu(const std::vector<mx::array> &inputs,
                                 std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &rhs = inputs[3];
  if (indices.dtype() == mx::int32) {
    chebyshev_apply_cpu_impl<int32_t>(data, indices, indptr, rhs, outputs[0],
                                      n_rows_, rhs_cols_, degree_, lambda_min_,
                                      lambda_max_, stream());
    return;
  }
  if (indices.dtype() == mx::int64) {
    chebyshev_apply_cpu_impl<int64_t>(data, indices, indptr, rhs, outputs[0],
                                      n_rows_, rhs_cols_, degree_, lambda_min_,
                                      lambda_max_, stream());
    return;
  }
  throw std::runtime_error(
      "csr_chebyshev_preconditioner_apply requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRChebyshevApply::eval_gpu(const std::vector<mx::array> &inputs,
                                 std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &rhs = inputs[3];
  auto &out = outputs[0];
  const int total = n_rows_ * rhs_cols_;

  out.set_data(mx::allocator::malloc(out.nbytes()));
  mx::array work(
      mx::allocator::malloc(static_cast<size_t>(3 * total) * sizeof(float)),
      mx::Shape{3 * total}, mx::float32);

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_chebyshev_apply", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(rhs, 3);
  encoder.set_output_array(out, 4);
  encoder.set_output_array(work, 5);
  encoder.set_bytes(n_rows_, 6);
  encoder.set_bytes(n_cols_, 7);
  encoder.set_bytes(rhs_cols_, 8);
  encoder.set_bytes(degree_, 9);
  encoder.set_bytes(lambda_min_, 10);
  encoder.set_bytes(lambda_max_, 11);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(work));
}
#else
void CSRChebyshevApply::eval_gpu(const std::vector<mx::array> &,
                                 std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_chebyshev_preconditioner_apply has no GPU implementation in this "
      "build.");
}
#endif

std::tuple<float, float, float, float, float, float, int>
csr_chebyshev_spectral_bounds(const mx::array &data, const mx::array &indices,
                              const mx::array &indptr, int n_rows, int n_cols,
                              bool estimate, int estimate_steps) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_chebyshev_spectral_bounds requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_chebyshev_spectral_bounds data");
  require_rank(indices, 1, "csr_chebyshev_spectral_bounds indices");
  require_rank(indptr, 1, "csr_chebyshev_spectral_bounds indptr");
  require_linalg_float32(data, "csr_chebyshev_spectral_bounds data");
  require_same_index_dtype(indices, indptr,
                           "csr_chebyshev_spectral_bounds indices",
                           "csr_chebyshev_spectral_bounds indptr");
  require_size(indptr, n_rows + 1, "csr_chebyshev_spectral_bounds indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_chebyshev_spectral_bounds data and indices must have equal "
        "length.");
  }
  if (estimate_steps < 0) {
    throw std::invalid_argument(
        "csr_chebyshev_spectral_bounds estimate_steps must be non-negative.");
  }
  const int steps =
      estimate_steps == 0 ? kDefaultEstimateSteps : estimate_steps;

  if (indices.dtype() == mx::int32) {
    validate_csr_structure<int32_t>(data, indices, indptr, n_rows, n_cols,
                                    "csr_chebyshev_spectral_bounds");
    auto [gmin, gmax, dmin, dmax] = gershgorin_and_diagonal_bounds<int32_t>(
        data, indices, indptr, n_rows, n_cols);
    auto [rmin, rmax, used] =
        estimate
            ? lanczos_ritz_bounds<int32_t>(data, indices, indptr, n_rows, steps)
            : std::tuple<float, float, int>{
                  std::numeric_limits<float>::quiet_NaN(),
                  std::numeric_limits<float>::quiet_NaN(), 0};
    return {gmin, gmax, rmin, rmax, dmin, dmax, used};
  }
  if (indices.dtype() == mx::int64) {
    validate_csr_structure<int64_t>(data, indices, indptr, n_rows, n_cols,
                                    "csr_chebyshev_spectral_bounds");
    auto [gmin, gmax, dmin, dmax] = gershgorin_and_diagonal_bounds<int64_t>(
        data, indices, indptr, n_rows, n_cols);
    auto [rmin, rmax, used] =
        estimate
            ? lanczos_ritz_bounds<int64_t>(data, indices, indptr, n_rows, steps)
            : std::tuple<float, float, int>{
                  std::numeric_limits<float>::quiet_NaN(),
                  std::numeric_limits<float>::quiet_NaN(), 0};
    return {gmin, gmax, rmin, rmax, dmin, dmax, used};
  }
  throw std::runtime_error(
      "csr_chebyshev_spectral_bounds requires int32 or int64 indices.");
}

mx::array csr_chebyshev_preconditioner_apply(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &rhs, int n_rows, int n_cols, int degree, float lambda_min,
    float lambda_max, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_chebyshev_preconditioner_apply requires a non-empty square "
        "matrix.");
  }
  validate_chebyshev_interval(degree, lambda_min, lambda_max);
  require_rank(data, 1, "csr_chebyshev_preconditioner_apply data");
  require_rank(indices, 1, "csr_chebyshev_preconditioner_apply indices");
  require_rank(indptr, 1, "csr_chebyshev_preconditioner_apply indptr");
  if (rhs.ndim() != 1 && rhs.ndim() != 2) {
    throw std::invalid_argument(
        "csr_chebyshev_preconditioner_apply rhs must be rank-1 or rank-2.");
  }
  require_linalg_float32(data, "csr_chebyshev_preconditioner_apply data");
  require_linalg_float32(rhs, "csr_chebyshev_preconditioner_apply rhs");
  require_same_index_dtype(indices, indptr,
                           "csr_chebyshev_preconditioner_apply indices",
                           "csr_chebyshev_preconditioner_apply indptr");
  require_size(indptr, n_rows + 1, "csr_chebyshev_preconditioner_apply indptr");
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_chebyshev_preconditioner_apply data and indices must have equal "
        "length.");
  }
  if (rhs.shape(0) != n_cols) {
    throw std::invalid_argument(
        "csr_chebyshev_preconditioner_apply rhs leading dimension must equal "
        "the sparse matrix column count.");
  }
  const int rhs_cols = rhs.ndim() == 2 ? rhs.shape(1) : 1;
  if (rhs_cols <= 0) {
    throw std::invalid_argument(
        "csr_chebyshev_preconditioner_apply rhs must have at least one "
        "column.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  return mx::array(rhs.shape(), mx::float32,
                   std::make_shared<CSRChebyshevApply>(stream, n_rows, n_cols,
                                                       rhs_cols, degree,
                                                       lambda_min, lambda_max),
                   {data_contig, indices_contig, indptr_contig, rhs_contig});
}

} // namespace mlx_sparse
