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

#include "linalg/svds/bidiagonal.h"

#include "linalg/common/common.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/random.h"
#include "mlx/transforms.h"
#include "sparse/csr_matvec/csr_matvec.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {
namespace {

using namespace linalg_detail;

#ifdef _METAL_
class CSRBidiagonalAdjoint : public mx::Primitive {
public:
  CSRBidiagonalAdjoint(mx::Stream stream, int rows, int cols)
      : Primitive(stream), rows_(rows), cols_(cols) {}
  const char *name() const override { return "CSRBidiagonalAdjoint"; }
  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRBidiagonalAdjoint &>(other);
    return rows_ == rhs.rows_ && cols_ == rhs.cols_;
  }
  void eval_cpu(const std::vector<mx::array> &,
                std::vector<mx::array> &) override {
    throw std::runtime_error("CSRBidiagonalAdjoint requires a GPU stream.");
  }
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override {
    auto &out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto &s = stream();
    auto &device = mx::metal::device(s.device);
    auto *library = device.get_library("mlx_sparse", current_binary_dir());
    auto &encoder = mx::metal::get_command_encoder(s);
    auto *zero = device.get_kernel("csr_bidiagonal_zero", library);
    encoder.set_compute_pipeline_state(zero);
    encoder.set_output_array(out, 0);
    encoder.set_bytes(cols_, 1);
    encoder.dispatch_threads(MTL::Size(cols_, 1, 1),
                             MTL::Size(kSolverThreads, 1, 1));
    auto *apply =
        device.get_kernel(sparse_kernel_name("csr_bidiagonal_adjoint",
                                             mx::float32, inputs[1].dtype()),
                          library);
    encoder.set_compute_pipeline_state(apply);
    for (int i = 0; i < 4; ++i) {
      encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 4);
    encoder.set_bytes(rows_, 5);
    encoder.dispatch_threads(MTL::Size(rows_, 1, 1),
                             MTL::Size(kSolverThreads, 1, 1));
  }

private:
  int rows_, cols_;
};

mx::array device_norm(const mx::array &work, mx::Stream stream) {
  auto scale = mx::max(mx::abs(work, stream), stream);
  auto divisor = mx::where(mx::greater(scale, mx::array(0.0f), stream), scale,
                           mx::array(1.0f), stream);
  auto scaled = mx::divide(work, divisor, stream);
  return mx::multiply(
      scale, mx::sqrt(mx::sum(mx::square(scaled, stream), stream), stream),
      stream);
}

mx::array device_direction(const mx::array &basis_t, mx::array work, int used,
                           mx::Stream stream) {
  const int n = work.shape(0);
  auto q = mx::slice(basis_t, {0, 0}, {used, n}, stream);
  for (uint32_t attempt = 0; attempt < 4; ++attempt) {
    if (attempt != 0) {
      const uint32_t seed =
          0x9e3779b9u * uint32_t(used) + 0x85ebca6bu * attempt;
      work = mx::random::uniform(-1.0f, 1.0f, {n}, mx::float32,
                                 mx::random::key(seed), stream);
    }
    auto initial = device_norm(work, stream);
    if (used != 0) {
      // CGS2 uses device GEMVs for the same two-pass orthogonality contract.
      for (int pass = 0; pass < 2; ++pass) {
        auto coefficients = mx::matmul(q, work, stream);
        work = mx::subtract(
            work, mx::matmul(mx::transpose(q, stream), coefficients, stream),
            stream);
      }
    }
    auto norm = device_norm(work, stream);
    mx::eval(initial, norm);
    const float threshold = (attempt == 0 ? 8 : 16) *
                            std::numeric_limits<float>::epsilon() *
                            initial.item<float>();
    if (norm.item<float>() > threshold) {
      return mx::divide(work, norm, stream);
    }
  }
  throw std::runtime_error("Krylov basis continuation failed. Could not "
                           "generate an independent start.");
}

std::tuple<mx::array, mx::array, mx::array, mx::array>
device_wide_basis(const mx::array &data, const mx::array &indices,
                  const mx::array &indptr, const mx::array &v0, int m, int n,
                  int p, mx::Stream stream) {
  const int q = std::min(n, p + 1);
  auto left = mx::zeros({p, m}, mx::float32, stream);
  auto right = mx::zeros({q, n}, mx::float32, stream);
  auto images = mx::zeros({q, m}, mx::float32, stream);
  auto scale = mx::max(mx::abs(v0, stream), stream);
  const float scale_value = scale.item<float>();
  if (scale_value == 0 || !std::isfinite(scale_value)) {
    return {left, right, images, mx::array(int32_t(0))};
  }
  auto scaled = mx::divide(v0, scale, stream);
  auto start = mx::divide(scaled, device_norm(scaled, stream), stream);
  right = mx::slice_update(right, mx::reshape(start, {1, n}, stream), {0, 0},
                           {1, n}, stream);
  for (int j = 0; j < p; ++j) {
    auto v =
        mx::reshape(mx::slice(right, {j, 0}, {j + 1, n}, stream), {n}, stream);
    auto av = csr_matvec(data, indices, indptr, v, m, n, stream);
    images = mx::slice_update(images, mx::reshape(av, {1, m}, stream), {j, 0},
                              {j + 1, m}, stream);
    auto u = device_direction(left, av, j, stream);
    left = mx::slice_update(left, mx::reshape(u, {1, m}, stream), {j, 0},
                            {j + 1, m}, stream);
    if (j + 1 == q) {
      break;
    }
    auto primitive = std::make_shared<CSRBidiagonalAdjoint>(stream, m, n);
    auto atu =
        mx::array({n}, mx::float32, primitive,
                  {data, indices, indptr, mx::contiguous(u, false, stream)});
    auto next = device_direction(right, atu, j + 1, stream);
    right = mx::slice_update(right, mx::reshape(next, {1, n}, stream),
                             {j + 1, 0}, {j + 2, n}, stream);
  }
  if (q > p) {
    auto v =
        mx::reshape(mx::slice(right, {p, 0}, {p + 1, n}, stream), {n}, stream);
    auto av = csr_matvec(data, indices, indptr, v, m, n, stream);
    images = mx::slice_update(images, mx::reshape(av, {1, m}, stream), {p, 0},
                              {p + 1, m}, stream);
  }
  return {left, right, images, mx::array(int32_t(p))};
}
#endif

class CSRBidiagonalBasis : public mx::Primitive {
public:
  CSRBidiagonalBasis(mx::Stream stream, int rows, int cols, int steps)
      : Primitive(stream), rows_(rows), cols_(cols), steps_(steps) {}
  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  const char *name() const override { return "CSRBidiagonalBasis"; }
  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRBidiagonalBasis &>(other);
    return rows_ == rhs.rows_ && cols_ == rhs.cols_ && steps_ == rhs.steps_;
  }

private:
  int rows_, cols_, steps_;
};

bool append_direction(float *basis, std::vector<float> &work, int n, int used) {
  for (uint32_t attempt = 0; attempt < 4; ++attempt) {
    if (attempt != 0) {
      const uint32_t seed =
          0x9e3779b9u * uint32_t(used) + 0x85ebca6bu * attempt;
      for (int row = 0; row < n; ++row) {
        work[row] = krylov_random_component(uint32_t(row), seed);
      }
    }
    const float initial = norm_float(work);
    for (int pass = 0; pass < 2; ++pass) {
      for (int col = 0; col < used; ++col) {
        const auto *q = basis + static_cast<size_t>(col) * n;
        double sum = 0;
        for (int row = 0; row < n; ++row) {
          sum += static_cast<double>(q[row]) * work[row];
        }
        const float coefficient = static_cast<float>(sum);
        for (int row = 0; row < n; ++row) {
          work[row] -= coefficient * q[row];
        }
      }
    }
    const float residual = norm_float(work);
    const float threshold = (attempt == 0 ? 8 : 16) *
                            std::numeric_limits<float>::epsilon() * initial;
    if (residual > threshold) {
      auto *q = basis + static_cast<size_t>(used) * n;
      for (int row = 0; row < n; ++row) {
        q[row] = work[row] / residual;
      }
      return true;
    }
  }
  return false;
}

template <typename I>
void build_basis(const std::vector<mx::array> &inputs,
                 std::vector<mx::array> &outputs, int m, int n, int p,
                 mx::Stream stream) {
  for (auto &out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  auto &encoder = mx::cpu::get_command_encoder(stream);
  for (const auto &in : inputs) {
    encoder.set_input_array(in);
  }
  for (auto &out : outputs) {
    encoder.set_output_array(out);
  }
  encoder.dispatch([data = mx::array::unsafe_weak_copy(inputs[0]),
                    indices = mx::array::unsafe_weak_copy(inputs[1]),
                    indptr = mx::array::unsafe_weak_copy(inputs[2]),
                    start = mx::array::unsafe_weak_copy(inputs[3]),
                    left = mx::array::unsafe_weak_copy(outputs[0]),
                    right = mx::array::unsafe_weak_copy(outputs[1]),
                    images = mx::array::unsafe_weak_copy(outputs[2]),
                    actual = mx::array::unsafe_weak_copy(outputs[3]), m, n,
                    p]() mutable {
    const auto *a = data.data<float>();
    const auto *ix = indices.data<I>();
    const auto *ip = indptr.data<I>();
    auto *u = left.data<float>();
    auto *v = right.data<float>();
    auto *used = actual.data<int32_t>();
    auto *av_basis = images.data<float>();
    const int q = std::min(n, p + 1);
    std::fill(u, u + static_cast<size_t>(m) * p, 0.0f);
    std::fill(v, v + static_cast<size_t>(n) * q, 0.0f);
    *used = 0;
    if (!initialize_krylov_basis(start.data<float>(), v, n, 1)) {
      return;
    }
    std::vector<float> av(m), atu(n);
    std::vector<double> sum(n);
    for (int j = 0; j < p; ++j) {
      for (int row = 0; row < m; ++row) {
        double acc = 0;
        for (I t = ip[row]; t < ip[row + 1]; ++t) {
          acc +=
              static_cast<double>(a[t]) * v[static_cast<size_t>(j) * n + ix[t]];
        }
        av[row] = static_cast<float>(acc);
        av_basis[static_cast<size_t>(j) * m + row] = av[row];
      }
      if (!append_direction(u, av, m, j)) {
        *used = -(j + 1);
        return;
      }
      *used = j + 1;
      if (j + 1 == q) {
        break;
      }
      std::fill(sum.begin(), sum.end(), 0.0);
      for (int row = 0; row < m; ++row) {
        const float value = u[static_cast<size_t>(j) * m + row];
        for (I t = ip[row]; t < ip[row + 1]; ++t) {
          sum[ix[t]] += static_cast<double>(a[t]) * value;
        }
      }
      for (int col = 0; col < n; ++col) {
        atu[col] = static_cast<float>(sum[col]);
      }
      if (!append_direction(v, atu, n, j + 1)) {
        *used = -(j + 1);
        return;
      }
    }
    if (q > p) {
      for (int row = 0; row < m; ++row) {
        double acc = 0;
        for (I t = ip[row]; t < ip[row + 1]; ++t) {
          acc +=
              static_cast<double>(a[t]) * v[static_cast<size_t>(p) * n + ix[t]];
        }
        av_basis[static_cast<size_t>(p) * m + row] = static_cast<float>(acc);
      }
    }
  });
}

void CSRBidiagonalBasis::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  if (inputs[1].dtype() == mx::int32) {
    build_basis<int32_t>(inputs, outputs, rows_, cols_, steps_, stream());
  } else {
    build_basis<int64_t>(inputs, outputs, rows_, cols_, steps_, stream());
  }
}

#ifdef _METAL_
void CSRBidiagonalBasis::eval_gpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  for (auto &out : outputs) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  mx::array left_work(
      mx::allocator::malloc(static_cast<size_t>(rows_) * sizeof(float)),
      mx::Shape{rows_}, mx::float32);
  mx::array right_work(
      mx::allocator::malloc(static_cast<size_t>(cols_) * sizeof(float)),
      mx::Shape{cols_}, mx::float32);
  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *library = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel =
      device.get_kernel(sparse_kernel_name("csr_bidiagonal_basis", mx::float32,
                                           inputs[1].dtype()),
                        library);
  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  for (int i = 0; i < 4; ++i) {
    encoder.set_input_array(inputs[i], i);
  }
  for (int i = 0; i < 4; ++i) {
    encoder.set_output_array(outputs[i], i + 4);
  }
  encoder.set_output_array(left_work, 8);
  encoder.set_output_array(right_work, 9);
  encoder.set_bytes(rows_, 10);
  encoder.set_bytes(cols_, 11);
  encoder.set_bytes(steps_, 12);
  encoder.dispatch_threads(MTL::Size(kSolverThreads, 1, 1),
                           MTL::Size(kSolverThreads, 1, 1));
  encoder.add_temporary(std::move(left_work));
  encoder.add_temporary(std::move(right_work));
}
#else
void CSRBidiagonalBasis::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_bidiagonal_basis has no GPU implementation in this build.");
}
#endif

} // namespace

std::tuple<mx::array, mx::array, mx::array, mx::array>
csr_bidiagonal_basis(const mx::array &data, const mx::array &indices,
                     const mx::array &indptr, const mx::array &v0, int rows,
                     int cols, int steps, mx::Stream stream) {
#ifdef _METAL_
  if (stream.device == mx::Device(mx::Device::gpu, 0) &&
      std::max(rows, cols) >= 8192) {
    return device_wide_basis(mx::contiguous(data, false, stream),
                             mx::contiguous(indices, false, stream),
                             mx::contiguous(indptr, false, stream),
                             mx::contiguous(v0, false, stream), rows, cols,
                             steps, stream);
  }
#endif
  auto primitive =
      std::make_shared<CSRBidiagonalBasis>(stream, rows, cols, steps);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{steps, rows}, mx::Shape{std::min(cols, steps + 1), cols},
       mx::Shape{std::min(cols, steps + 1), rows}, mx::Shape{}},
      {mx::float32, mx::float32, mx::float32, mx::int32}, primitive,
      {mx::contiguous(data, false, stream),
       mx::contiguous(indices, false, stream),
       mx::contiguous(indptr, false, stream),
       mx::contiguous(v0, false, stream)});
  return {outputs[0], outputs[1], outputs[2], outputs[3]};
}

} // namespace mlx_sparse
