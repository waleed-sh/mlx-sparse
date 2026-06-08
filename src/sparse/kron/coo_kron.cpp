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

#include "sparse/kron/coo_kron.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "common/common.h"
#include "common/cpu_parallel.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

namespace mlx_sparse {

namespace {

template <typename T> typename Accumulator<T>::Type accumulator_value(T value) {
  using AccT = typename Accumulator<T>::Type;
  if constexpr (std::is_same_v<T, mx::float16_t> ||
                std::is_same_v<T, mx::bfloat16_t>) {
    return static_cast<float>(value);
  } else {
    return static_cast<AccT>(value);
  }
}

int64_t checked_product(int64_t lhs, int64_t rhs, const char *name) {
  if (lhs < 0 || rhs < 0) {
    throw std::invalid_argument(std::string(name) +
                                " inputs must be non-negative.");
  }
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
    throw std::overflow_error(std::string(name) + " product overflows int64.");
  }
  const int64_t product = lhs * rhs;
  if (product > std::numeric_limits<int>::max()) {
    throw std::overflow_error(std::string(name) + " exceeds MLX shape limits.");
  }
  return product;
}

mx::Dtype promoted_index_dtype(mx::Dtype lhs, mx::Dtype rhs) {
  if (lhs != mx::int32 && lhs != mx::int64) {
    throw std::invalid_argument("coo_kron lhs indices must be int32 or int64.");
  }
  if (rhs != mx::int32 && rhs != mx::int64) {
    throw std::invalid_argument("coo_kron rhs indices must be int32 or int64.");
  }
  return lhs == rhs ? lhs : mx::int64;
}

void check_index_capacity(int64_t n_rows, int64_t n_cols,
                          mx::Dtype index_dtype) {
  if (index_dtype == mx::int32) {
    if (n_rows > std::numeric_limits<int32_t>::max() ||
        n_cols > std::numeric_limits<int32_t>::max()) {
      throw std::overflow_error(
          "coo_kron output shape exceeds int32 index capacity.");
    }
  }
}

std::string kron_data_kernel_name(const std::string &prefix,
                                  mx::Dtype value_dtype) {
  return prefix + "_" + value_kernel_suffix(value_dtype);
}

std::string kron_index_kernel_name(const std::string &prefix,
                                   mx::Dtype lhs_index_dtype,
                                   mx::Dtype rhs_index_dtype,
                                   mx::Dtype out_index_dtype) {
  return prefix + "_" + index_kernel_suffix(lhs_index_dtype) + "_" +
         index_kernel_suffix(rhs_index_dtype) + "_" +
         index_kernel_suffix(out_index_dtype);
}

class COOKronData : public mx::Primitive {
public:
  COOKronData(mx::Stream stream, int lhs_nnz, int rhs_nnz)
      : Primitive(stream), lhs_nnz_(lhs_nnz), rhs_nnz_(rhs_nnz) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  std::vector<mx::array> jvp(const std::vector<mx::array> &primals,
                             const std::vector<mx::array> &tangents,
                             const std::vector<int> &argnums) override;

  std::vector<mx::array> vjp(const std::vector<mx::array> &primals,
                             const std::vector<mx::array> &cotangents,
                             const std::vector<int> &argnums,
                             const std::vector<mx::array> &) override;

  const char *name() const override { return "COOKronData"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOKronData &>(other);
    return lhs_nnz_ == rhs.lhs_nnz_ && rhs_nnz_ == rhs.rhs_nnz_;
  }

private:
  int lhs_nnz_;
  int rhs_nnz_;
};

class COOKronDataVJP : public mx::Primitive {
public:
  COOKronDataVJP(mx::Stream stream, int lhs_nnz, int rhs_nnz, bool lhs_grad)
      : Primitive(stream), lhs_nnz_(lhs_nnz), rhs_nnz_(rhs_nnz),
        lhs_grad_(lhs_grad) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOKronDataVJP"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOKronDataVJP &>(other);
    return lhs_nnz_ == rhs.lhs_nnz_ && rhs_nnz_ == rhs.rhs_nnz_ &&
           lhs_grad_ == rhs.lhs_grad_;
  }

private:
  int lhs_nnz_;
  int rhs_nnz_;
  bool lhs_grad_;
};

class COOKronIndices : public mx::Primitive {
public:
  COOKronIndices(mx::Stream stream, int lhs_nnz, int rhs_nnz, int rhs_n_rows,
                 int rhs_n_cols)
      : Primitive(stream), lhs_nnz_(lhs_nnz), rhs_nnz_(rhs_nnz),
        rhs_n_rows_(rhs_n_rows), rhs_n_cols_(rhs_n_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "COOKronIndices"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const COOKronIndices &>(other);
    return lhs_nnz_ == rhs.lhs_nnz_ && rhs_nnz_ == rhs.rhs_nnz_ &&
           rhs_n_rows_ == rhs.rhs_n_rows_ && rhs_n_cols_ == rhs.rhs_n_cols_;
  }

private:
  int lhs_nnz_;
  int rhs_nnz_;
  int rhs_n_rows_;
  int rhs_n_cols_;
};

template <typename T>
void kron_data_cpu_impl(const mx::array &lhs_data, const mx::array &rhs_data,
                        mx::array &out, int lhs_nnz, int rhs_nnz,
                        mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(lhs_data);
  encoder.set_input_array(rhs_data);
  encoder.set_output_array(out);

  encoder.dispatch([lhs_data = mx::array::unsafe_weak_copy(lhs_data),
                    rhs_data = mx::array::unsafe_weak_copy(rhs_data),
                    out = mx::array::unsafe_weak_copy(out), lhs_nnz,
                    rhs_nnz]() mutable {
    const auto *lhs = lhs_data.data<T>();
    const auto *rhs = rhs_data.data<T>();
    auto *dst = out.data<T>();
    const int64_t total = static_cast<int64_t>(lhs_nnz) * rhs_nnz;

    auto fill_range = [&](CpuRange range) {
      for (int64_t k = range.begin; k < range.end; ++k) {
        const int64_t i = k / rhs_nnz;
        const int64_t j = k - i * rhs_nnz;
        dst[k] = Accumulator<T>::cast(accumulator_value<T>(lhs[i]) *
                                      accumulator_value<T>(rhs[j]));
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || total <= 0) {
      fill_range({0, static_cast<int>(total)});
      return;
    }
    parallel_for_cpu_ranges(equal_cpu_ranges(static_cast<int>(total), workers),
                            fill_range);
  });
}

template <typename T>
void kron_data_vjp_cpu_impl(const mx::array &other_data,
                            const mx::array &cotangent, mx::array &out,
                            int lhs_nnz, int rhs_nnz, bool lhs_grad,
                            mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(other_data);
  encoder.set_input_array(cotangent);
  encoder.set_output_array(out);

  encoder.dispatch([other_data = mx::array::unsafe_weak_copy(other_data),
                    cotangent = mx::array::unsafe_weak_copy(cotangent),
                    out = mx::array::unsafe_weak_copy(out), lhs_nnz, rhs_nnz,
                    lhs_grad]() mutable {
    using AccT = typename Accumulator<T>::Type;
    const auto *other = other_data.data<T>();
    const auto *cot = cotangent.data<T>();
    auto *dst = out.data<T>();
    const int out_size = lhs_grad ? lhs_nnz : rhs_nnz;

    auto fill_range = [&](CpuRange range) {
      for (int index = range.begin; index < range.end; ++index) {
        AccT acc = Accumulator<T>::zero();
        if (lhs_grad) {
          const int64_t base = static_cast<int64_t>(index) * rhs_nnz;
          for (int j = 0; j < rhs_nnz; ++j) {
            acc += accumulator_value<T>(cot[base + j]) *
                   accumulator_value<T>(other[j]);
          }
        } else {
          for (int i = 0; i < lhs_nnz; ++i) {
            acc += accumulator_value<T>(
                       cot[static_cast<int64_t>(i) * rhs_nnz + index]) *
                   accumulator_value<T>(other[i]);
          }
        }
        dst[index] = Accumulator<T>::cast(acc);
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || out_size <= 0) {
      fill_range({0, out_size});
      return;
    }
    parallel_for_cpu_ranges(equal_cpu_ranges(out_size, workers), fill_range);
  });
}

template <typename LhsI, typename RhsI, typename OutI>
void kron_indices_cpu_impl(const mx::array &lhs_row, const mx::array &lhs_col,
                           const mx::array &rhs_row, const mx::array &rhs_col,
                           mx::array &out_row, mx::array &out_col, int lhs_nnz,
                           int rhs_nnz, int rhs_n_rows, int rhs_n_cols,
                           mx::Stream stream) {
  out_row.set_data(mx::allocator::malloc(out_row.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(lhs_row);
  encoder.set_input_array(lhs_col);
  encoder.set_input_array(rhs_row);
  encoder.set_input_array(rhs_col);
  encoder.set_output_array(out_row);
  encoder.set_output_array(out_col);

  encoder.dispatch([lhs_row = mx::array::unsafe_weak_copy(lhs_row),
                    lhs_col = mx::array::unsafe_weak_copy(lhs_col),
                    rhs_row = mx::array::unsafe_weak_copy(rhs_row),
                    rhs_col = mx::array::unsafe_weak_copy(rhs_col),
                    out_row = mx::array::unsafe_weak_copy(out_row),
                    out_col = mx::array::unsafe_weak_copy(out_col), lhs_nnz,
                    rhs_nnz, rhs_n_rows, rhs_n_cols]() mutable {
    const auto *lhs_r = lhs_row.data<LhsI>();
    const auto *lhs_c = lhs_col.data<LhsI>();
    const auto *rhs_r = rhs_row.data<RhsI>();
    const auto *rhs_c = rhs_col.data<RhsI>();
    auto *row = out_row.data<OutI>();
    auto *col = out_col.data<OutI>();
    const int64_t total = static_cast<int64_t>(lhs_nnz) * rhs_nnz;

    auto fill_range = [&](CpuRange range) {
      for (int64_t k = range.begin; k < range.end; ++k) {
        const int64_t i = k / rhs_nnz;
        const int64_t j = k - i * rhs_nnz;
        row[k] = static_cast<OutI>(static_cast<int64_t>(lhs_r[i]) * rhs_n_rows +
                                   rhs_r[j]);
        col[k] = static_cast<OutI>(static_cast<int64_t>(lhs_c[i]) * rhs_n_cols +
                                   rhs_c[j]);
      }
    };

    const int workers = configured_cpu_worker_count();
    if (workers <= 1 || total <= 0) {
      fill_range({0, static_cast<int>(total)});
      return;
    }
    parallel_for_cpu_ranges(equal_cpu_ranges(static_cast<int>(total), workers),
                            fill_range);
  });
}

mx::array kron_data_vjp(const mx::array &other_data, const mx::array &cotangent,
                        int lhs_nnz, int rhs_nnz, bool lhs_grad,
                        mx::Stream stream) {
  const int out_size = lhs_grad ? lhs_nnz : rhs_nnz;
  auto primitive =
      std::make_shared<COOKronDataVJP>(stream, lhs_nnz, rhs_nnz, lhs_grad);
  return mx::array(mx::Shape{out_size}, cotangent.dtype(), primitive,
                   {other_data, cotangent});
}

} // namespace

void COOKronData::eval_cpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &rhs_data = inputs[1];

#define DISPATCH_COO_KRON_DATA_CPU(DTYPE, TYPE)                                \
  if (lhs_data.dtype() == DTYPE) {                                             \
    kron_data_cpu_impl<TYPE>(lhs_data, rhs_data, outputs[0], lhs_nnz_,         \
                             rhs_nnz_, stream());                              \
    return;                                                                    \
  }

  DISPATCH_COO_KRON_DATA_CPU(mx::float32, float)
  DISPATCH_COO_KRON_DATA_CPU(mx::float16, mx::float16_t)
  DISPATCH_COO_KRON_DATA_CPU(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_KRON_DATA_CPU(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_KRON_DATA_CPU

  throw std::runtime_error("coo_kron_data unsupported value dtype.");
}

#ifdef _METAL_
void COOKronData::eval_gpu(const std::vector<mx::array> &inputs,
                           std::vector<mx::array> &outputs) {
  const auto &lhs_data = inputs[0];
  const auto &rhs_data = inputs[1];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      kron_data_kernel_name("coo_kron_data", lhs_data.dtype()), lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(lhs_data, 0);
  encoder.set_input_array(rhs_data, 1);
  encoder.set_output_array(out, 2);
  encoder.set_bytes(lhs_nnz_, 3);
  encoder.set_bytes(rhs_nnz_, 4);

  const auto threads =
      std::max<size_t>(static_cast<size_t>(lhs_nnz_) * rhs_nnz_, 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOKronData::eval_gpu(const std::vector<mx::array> &,
                           std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_kron_data has no GPU implementation in this build.");
}
#endif

std::vector<mx::array> COOKronData::jvp(const std::vector<mx::array> &primals,
                                        const std::vector<mx::array> &tangents,
                                        const std::vector<int> &argnums) {
  std::vector<mx::array> terms;
  for (size_t i = 0; i < argnums.size(); ++i) {
    if (argnums[i] == 0) {
      terms.push_back(coo_kron_data(tangents[i], primals[1], stream()));
    } else if (argnums[i] == 1) {
      terms.push_back(coo_kron_data(primals[0], tangents[i], stream()));
    } else {
      throw std::runtime_error(
          "COOKronData JVP is implemented only for sparse data arrays.");
    }
  }
  if (terms.empty()) {
    throw std::runtime_error("COOKronData JVP requires at least one tangent.");
  }
  auto result = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    result = mx::add(result, terms[i], stream());
  }
  return {result};
}

std::vector<mx::array>
COOKronData::vjp(const std::vector<mx::array> &primals,
                 const std::vector<mx::array> &cotangents,
                 const std::vector<int> &argnums,
                 const std::vector<mx::array> &) {
  std::vector<mx::array> vjps;
  for (int argnum : argnums) {
    if (argnum == 0) {
      auto rhs = primals[1].dtype() == mx::complex64
                     ? mx::conjugate(primals[1], stream())
                     : primals[1];
      vjps.push_back(kron_data_vjp(rhs, cotangents[0], lhs_nnz_, rhs_nnz_, true,
                                   stream()));
    } else if (argnum == 1) {
      auto lhs = primals[0].dtype() == mx::complex64
                     ? mx::conjugate(primals[0], stream())
                     : primals[0];
      vjps.push_back(kron_data_vjp(lhs, cotangents[0], lhs_nnz_, rhs_nnz_,
                                   false, stream()));
    } else {
      throw std::runtime_error(
          "COOKronData VJP is implemented only for sparse data arrays.");
    }
  }
  return vjps;
}

void COOKronDataVJP::eval_cpu(const std::vector<mx::array> &inputs,
                              std::vector<mx::array> &outputs) {
  const auto &other_data = inputs[0];
  const auto &cotangent = inputs[1];

#define DISPATCH_COO_KRON_VJP_CPU(DTYPE, TYPE)                                 \
  if (cotangent.dtype() == DTYPE) {                                            \
    kron_data_vjp_cpu_impl<TYPE>(other_data, cotangent, outputs[0], lhs_nnz_,  \
                                 rhs_nnz_, lhs_grad_, stream());               \
    return;                                                                    \
  }

  DISPATCH_COO_KRON_VJP_CPU(mx::float32, float)
  DISPATCH_COO_KRON_VJP_CPU(mx::float16, mx::float16_t)
  DISPATCH_COO_KRON_VJP_CPU(mx::bfloat16, mx::bfloat16_t)
  DISPATCH_COO_KRON_VJP_CPU(mx::complex64, mx::complex64_t)
#undef DISPATCH_COO_KRON_VJP_CPU

  throw std::runtime_error("coo_kron_data VJP unsupported value dtype.");
}

#ifdef _METAL_
void COOKronDataVJP::eval_gpu(const std::vector<mx::array> &inputs,
                              std::vector<mx::array> &outputs) {
  const auto &other_data = inputs[0];
  const auto &cotangent = inputs[1];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      kron_data_kernel_name("coo_kron_data_vjp", cotangent.dtype()), lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(other_data, 0);
  encoder.set_input_array(cotangent, 1);
  encoder.set_output_array(out, 2);
  encoder.set_bytes(lhs_nnz_, 3);
  encoder.set_bytes(rhs_nnz_, 4);
  const int lhs_grad = lhs_grad_ ? 1 : 0;
  encoder.set_bytes(lhs_grad, 5);

  const auto threads =
      std::max<size_t>(static_cast<size_t>(lhs_grad_ ? lhs_nnz_ : rhs_nnz_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOKronDataVJP::eval_gpu(const std::vector<mx::array> &,
                              std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_kron_data VJP has no GPU implementation in this build.");
}
#endif

void COOKronIndices::eval_cpu(const std::vector<mx::array> &inputs,
                              std::vector<mx::array> &outputs) {
  const auto &lhs_row = inputs[0];
  const auto &lhs_col = inputs[1];
  const auto &rhs_row = inputs[2];
  const auto &rhs_col = inputs[3];

  if (lhs_row.dtype() == mx::int32 && rhs_row.dtype() == mx::int32 &&
      outputs[0].dtype() == mx::int32) {
    kron_indices_cpu_impl<int32_t, int32_t, int32_t>(
        lhs_row, lhs_col, rhs_row, rhs_col, outputs[0], outputs[1], lhs_nnz_,
        rhs_nnz_, rhs_n_rows_, rhs_n_cols_, stream());
    return;
  }
  if (lhs_row.dtype() == mx::int32 && rhs_row.dtype() == mx::int32) {
    kron_indices_cpu_impl<int32_t, int32_t, int64_t>(
        lhs_row, lhs_col, rhs_row, rhs_col, outputs[0], outputs[1], lhs_nnz_,
        rhs_nnz_, rhs_n_rows_, rhs_n_cols_, stream());
    return;
  }
  if (lhs_row.dtype() == mx::int32 && rhs_row.dtype() == mx::int64) {
    kron_indices_cpu_impl<int32_t, int64_t, int64_t>(
        lhs_row, lhs_col, rhs_row, rhs_col, outputs[0], outputs[1], lhs_nnz_,
        rhs_nnz_, rhs_n_rows_, rhs_n_cols_, stream());
    return;
  }
  if (lhs_row.dtype() == mx::int64 && rhs_row.dtype() == mx::int32) {
    kron_indices_cpu_impl<int64_t, int32_t, int64_t>(
        lhs_row, lhs_col, rhs_row, rhs_col, outputs[0], outputs[1], lhs_nnz_,
        rhs_nnz_, rhs_n_rows_, rhs_n_cols_, stream());
    return;
  }
  kron_indices_cpu_impl<int64_t, int64_t, int64_t>(
      lhs_row, lhs_col, rhs_row, rhs_col, outputs[0], outputs[1], lhs_nnz_,
      rhs_nnz_, rhs_n_rows_, rhs_n_cols_, stream());
}

#ifdef _METAL_
void COOKronIndices::eval_gpu(const std::vector<mx::array> &inputs,
                              std::vector<mx::array> &outputs) {
  const auto &lhs_row = inputs[0];
  const auto &lhs_col = inputs[1];
  const auto &rhs_row = inputs[2];
  const auto &rhs_col = inputs[3];
  auto &out_row = outputs[0];
  auto &out_col = outputs[1];

  out_row.set_data(mx::allocator::malloc(out_row.nbytes()));
  out_col.set_data(mx::allocator::malloc(out_col.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel(
      kron_index_kernel_name("coo_kron_indices", lhs_row.dtype(),
                             rhs_row.dtype(), out_row.dtype()),
      lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(lhs_row, 0);
  encoder.set_input_array(lhs_col, 1);
  encoder.set_input_array(rhs_row, 2);
  encoder.set_input_array(rhs_col, 3);
  encoder.set_output_array(out_row, 4);
  encoder.set_output_array(out_col, 5);
  encoder.set_bytes(lhs_nnz_, 6);
  encoder.set_bytes(rhs_nnz_, 7);
  encoder.set_bytes(rhs_n_rows_, 8);
  encoder.set_bytes(rhs_n_cols_, 9);

  const auto threads =
      std::max<size_t>(static_cast<size_t>(lhs_nnz_) * rhs_nnz_, 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void COOKronIndices::eval_gpu(const std::vector<mx::array> &,
                              std::vector<mx::array> &) {
  throw std::runtime_error(
      "coo_kron_indices has no GPU implementation in this build.");
}
#endif

mx::array coo_kron_data(const mx::array &lhs_data, const mx::array &rhs_data,
                        mx::StreamOrDevice s) {
  require_rank(lhs_data, 1, "coo_kron lhs_data");
  require_rank(rhs_data, 1, "coo_kron rhs_data");
  require_same_value_dtype(lhs_data, rhs_data, "coo_kron lhs_data",
                           "coo_kron rhs_data");
  const auto lhs_nnz = static_cast<int64_t>(lhs_data.size());
  const auto rhs_nnz = static_cast<int64_t>(rhs_data.size());
  const int out_nnz =
      static_cast<int>(checked_product(lhs_nnz, rhs_nnz, "coo_kron nnz"));

  auto stream = mx::to_stream(s);
  auto lhs_data_contig = mx::contiguous(lhs_data, false, stream);
  auto rhs_data_contig = mx::contiguous(rhs_data, false, stream);
  auto primitive = std::make_shared<COOKronData>(
      stream, static_cast<int>(lhs_nnz), static_cast<int>(rhs_nnz));
  return mx::array(mx::Shape{out_nnz}, lhs_data.dtype(), primitive,
                   {lhs_data_contig, rhs_data_contig});
}

std::tuple<mx::array, mx::array>
coo_kron_indices(const mx::array &lhs_row, const mx::array &lhs_col,
                 const mx::array &rhs_row, const mx::array &rhs_col,
                 int lhs_n_rows, int lhs_n_cols, int rhs_n_rows, int rhs_n_cols,
                 mx::StreamOrDevice s) {
  if (lhs_n_rows < 0 || lhs_n_cols < 0 || rhs_n_rows < 0 || rhs_n_cols < 0) {
    throw std::invalid_argument(
        "coo_kron shape dimensions must be non-negative.");
  }
  require_rank(lhs_row, 1, "coo_kron lhs_row");
  require_rank(lhs_col, 1, "coo_kron lhs_col");
  require_rank(rhs_row, 1, "coo_kron rhs_row");
  require_rank(rhs_col, 1, "coo_kron rhs_col");
  require_same_index_dtype(lhs_row, lhs_col, "coo_kron lhs_row",
                           "coo_kron lhs_col");
  require_same_index_dtype(rhs_row, rhs_col, "coo_kron rhs_row",
                           "coo_kron rhs_col");
  if (lhs_row.size() != lhs_col.size() || rhs_row.size() != rhs_col.size()) {
    throw std::invalid_argument(
        "coo_kron row and col arrays must have equal lengths per operand.");
  }

  const auto lhs_nnz = static_cast<int64_t>(lhs_row.size());
  const auto rhs_nnz = static_cast<int64_t>(rhs_row.size());
  const int out_nnz =
      static_cast<int>(checked_product(lhs_nnz, rhs_nnz, "coo_kron nnz"));
  const auto out_n_rows =
      checked_product(lhs_n_rows, rhs_n_rows, "coo_kron n_rows");
  const auto out_n_cols =
      checked_product(lhs_n_cols, rhs_n_cols, "coo_kron n_cols");
  const auto out_index_dtype =
      promoted_index_dtype(lhs_row.dtype(), rhs_row.dtype());
  check_index_capacity(out_n_rows, out_n_cols, out_index_dtype);

  auto stream = mx::to_stream(s);
  auto lhs_row_contig = mx::contiguous(lhs_row, false, stream);
  auto lhs_col_contig = mx::contiguous(lhs_col, false, stream);
  auto rhs_row_contig = mx::contiguous(rhs_row, false, stream);
  auto rhs_col_contig = mx::contiguous(rhs_col, false, stream);
  auto primitive = std::make_shared<COOKronIndices>(
      stream, static_cast<int>(lhs_nnz), static_cast<int>(rhs_nnz), rhs_n_rows,
      rhs_n_cols);
  auto outputs = mx::array::make_arrays(
      {mx::Shape{out_nnz}, mx::Shape{out_nnz}},
      {out_index_dtype, out_index_dtype}, primitive,
      {lhs_row_contig, lhs_col_contig, rhs_row_contig, rhs_col_contig});
  return {outputs[0], outputs[1]};
}

std::tuple<mx::array, mx::array, mx::array>
coo_kron(const mx::array &lhs_data, const mx::array &lhs_row,
         const mx::array &lhs_col, const mx::array &rhs_data,
         const mx::array &rhs_row, const mx::array &rhs_col, int lhs_n_rows,
         int lhs_n_cols, int rhs_n_rows, int rhs_n_cols, mx::StreamOrDevice s) {
  if (lhs_data.size() != lhs_row.size() || rhs_data.size() != rhs_row.size()) {
    throw std::invalid_argument(
        "coo_kron data and coordinate arrays must have equal lengths.");
  }
  auto stream = mx::to_stream(s);
  auto data = coo_kron_data(lhs_data, rhs_data, stream);
  auto [row, col] =
      coo_kron_indices(lhs_row, lhs_col, rhs_row, rhs_col, lhs_n_rows,
                       lhs_n_cols, rhs_n_rows, rhs_n_cols, stream);
  return {data, row, col};
}

} // namespace mlx_sparse
