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

#include "linalg/inner_product/inner_product.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "common/cpu_parallel.h"
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

class CSRVdot : public mx::Primitive {
public:
  CSRVdot(mx::Stream stream, int n_rows, int n_cols, bool conjugate_lhs)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols),
        conjugate_lhs_(conjugate_lhs) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRVdot"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRVdot &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           conjugate_lhs_ == rhs.conjugate_lhs_;
  }

private:
  int n_rows_;
  int n_cols_;
  bool conjugate_lhs_;
};

template <typename T> T sparse_inner_product_value(T lhs, T rhs, bool) {
  return lhs * rhs;
}

template <>
mx::complex64_t sparse_inner_product_value(mx::complex64_t lhs,
                                           mx::complex64_t rhs,
                                           bool conjugate_lhs) {
  const std::complex<float> lhs_value(lhs);
  const std::complex<float> rhs_value(rhs);
  return mx::complex64_t((conjugate_lhs ? std::conj(lhs_value) : lhs_value) *
                         rhs_value);
}

template <typename T, typename I>
void csr_vdot_cpu_impl(const mx::array &lhs_data, const mx::array &lhs_indices,
                       const mx::array &lhs_indptr, const mx::array &rhs_data,
                       const mx::array &rhs_indices,
                       const mx::array &rhs_indptr, mx::array &out, int n_rows,
                       bool conjugate_lhs, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(lhs_data);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(lhs_indptr);
  encoder.set_input_array(rhs_data);
  encoder.set_input_array(rhs_indices);
  encoder.set_input_array(rhs_indptr);
  encoder.set_output_array(out);

  encoder.dispatch([lhs_data = mx::array::unsafe_weak_copy(lhs_data),
                    lhs_indices = mx::array::unsafe_weak_copy(lhs_indices),
                    lhs_indptr = mx::array::unsafe_weak_copy(lhs_indptr),
                    rhs_data = mx::array::unsafe_weak_copy(rhs_data),
                    rhs_indices = mx::array::unsafe_weak_copy(rhs_indices),
                    rhs_indptr = mx::array::unsafe_weak_copy(rhs_indptr),
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    conjugate_lhs]() mutable {
    const auto *lhs_data_ptr = lhs_data.data<T>();
    const auto *lhs_indices_ptr = lhs_indices.data<I>();
    const auto *lhs_indptr_ptr = lhs_indptr.data<I>();
    const auto *rhs_data_ptr = rhs_data.data<T>();
    const auto *rhs_indices_ptr = rhs_indices.data<I>();
    const auto *rhs_indptr_ptr = rhs_indptr.data<I>();
    using AccT = std::conditional_t<std::is_same_v<T, mx::complex64_t>,
                                    std::complex<double>, double>;

    auto compute_rows = [&](CpuRange range) -> AccT {
      AccT local{};
      for (int row = range.begin; row < range.end; ++row) {
        I lp = lhs_indptr_ptr[row];
        I rp = rhs_indptr_ptr[row];
        const I lend = lhs_indptr_ptr[row + 1];
        const I rend = rhs_indptr_ptr[row + 1];
        while (lp < lend && rp < rend) {
          const I lc = lhs_indices_ptr[lp];
          const I rc = rhs_indices_ptr[rp];
          if (lc == rc) {
            if constexpr (std::is_same_v<T, mx::complex64_t>) {
              const std::complex<float> lhs_value(lhs_data_ptr[lp]);
              const std::complex<float> rhs_value(rhs_data_ptr[rp]);
              local += static_cast<std::complex<double>>(
                  conjugate_lhs ? std::conj(lhs_value) * rhs_value
                                : lhs_value * rhs_value);
            } else {
              local += static_cast<double>(sparse_inner_product_value<T>(
                  lhs_data_ptr[lp], rhs_data_ptr[rp], conjugate_lhs));
            }
            ++lp;
            ++rp;
          } else if (lc < rc) {
            ++lp;
          } else {
            ++rp;
          }
        }
      }
      return local;
    };

    AccT acc{};
    const int workers = configured_cpu_worker_count();
    const auto estimated_work =
        static_cast<int64_t>(lhs_data.size() + rhs_data.size());
    if (!should_parallelize_cpu_tree_reduction(workers, estimated_work) ||
        n_rows <= 0) {
      acc = compute_rows({0, n_rows});
    } else {
      std::vector<int64_t> row_work(static_cast<size_t>(n_rows));
      for (int row = 0; row < n_rows; ++row) {
        row_work[static_cast<size_t>(row)] =
            static_cast<int64_t>(lhs_indptr_ptr[row + 1] -
                                 lhs_indptr_ptr[row]) +
            static_cast<int64_t>(rhs_indptr_ptr[row + 1] - rhs_indptr_ptr[row]);
      }
      const auto ranges = cpu_ranges_for_output_work(row_work, workers);
      acc = parallel_reduce_cpu_ranges<AccT>(ranges, compute_rows);
    }
    if constexpr (std::is_same_v<T, mx::complex64_t>) {
      *out.data<T>() = mx::complex64_t(static_cast<float>(acc.real()),
                                       static_cast<float>(acc.imag()));
    } else {
      *out.data<T>() = static_cast<T>(acc);
    }
  });
}

} // namespace

void CSRVdot::eval_cpu(const std::vector<mx::array> &inputs,
                       std::vector<mx::array> &outputs) {
  auto &lhs_data = inputs[0];
  auto &lhs_indices = inputs[1];
  auto &lhs_indptr = inputs[2];
  auto &rhs_data = inputs[3];
  auto &rhs_indices = inputs[4];
  auto &rhs_indptr = inputs[5];

  if (lhs_data.dtype() == mx::float32 && lhs_indices.dtype() == mx::int32) {
    csr_vdot_cpu_impl<float, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        outputs[0], n_rows_, conjugate_lhs_, stream());
    return;
  }
  if (lhs_data.dtype() == mx::float32 && lhs_indices.dtype() == mx::int64) {
    csr_vdot_cpu_impl<float, int64_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        outputs[0], n_rows_, conjugate_lhs_, stream());
    return;
  }
  if (lhs_data.dtype() == mx::complex64 && lhs_indices.dtype() == mx::int32) {
    csr_vdot_cpu_impl<mx::complex64_t, int32_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        outputs[0], n_rows_, conjugate_lhs_, stream());
    return;
  }
  if (lhs_data.dtype() == mx::complex64 && lhs_indices.dtype() == mx::int64) {
    csr_vdot_cpu_impl<mx::complex64_t, int64_t>(
        lhs_data, lhs_indices, lhs_indptr, rhs_data, rhs_indices, rhs_indptr,
        outputs[0], n_rows_, conjugate_lhs_, stream());
    return;
  }
  throw std::runtime_error("csr_vdot requires float32 or complex64 data with "
                           "int32 or int64 indices.");
}

#ifdef _METAL_
void CSRVdot::eval_gpu(const std::vector<mx::array> &inputs,
                       std::vector<mx::array> &outputs) {
  auto &lhs_data = inputs[0];
  auto &lhs_indices = inputs[1];
  auto &lhs_indptr = inputs[2];
  auto &rhs_data = inputs[3];
  auto &rhs_indices = inputs[4];
  auto &rhs_indptr = inputs[5];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name = sparse_kernel_name(conjugate_lhs_ ? "csr_vdot" : "csr_dot",
                                        lhs_data.dtype(), lhs_indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(lhs_data, 0);
  encoder.set_input_array(lhs_indices, 1);
  encoder.set_input_array(lhs_indptr, 2);
  encoder.set_input_array(rhs_data, 3);
  encoder.set_input_array(rhs_indices, 4);
  encoder.set_input_array(rhs_indptr, 5);
  encoder.set_output_array(out, 6);
  encoder.set_bytes(n_rows_, 7);
  encoder.set_bytes(n_cols_, 8);
  encoder.dispatch_threads(MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
}
#else
void CSRVdot::eval_gpu(const std::vector<mx::array> &,
                       std::vector<mx::array> &) {
  throw std::runtime_error("csr_vdot has no GPU implementation in this build.");
}
#endif

mx::array csr_inner_product(const mx::array &lhs_data,
                            const mx::array &lhs_indices,
                            const mx::array &lhs_indptr,
                            const mx::array &rhs_data,
                            const mx::array &rhs_indices,
                            const mx::array &rhs_indptr, int n_rows, int n_cols,
                            bool conjugate_lhs, mx::StreamOrDevice s) {
  if (n_rows < 0 || n_cols < 0) {
    throw std::invalid_argument(
        "csr sparse inner product shape dimensions must be non-negative.");
  }
  require_rank(lhs_data, 1, "csr inner product lhs_data");
  require_rank(lhs_indices, 1, "csr inner product lhs_indices");
  require_rank(lhs_indptr, 1, "csr inner product lhs_indptr");
  require_rank(rhs_data, 1, "csr inner product rhs_data");
  require_rank(rhs_indices, 1, "csr inner product rhs_indices");
  require_rank(rhs_indptr, 1, "csr inner product rhs_indptr");
  require_inner_product_dtype(lhs_data, "csr inner product lhs_data");
  require_inner_product_dtype(rhs_data, "csr inner product rhs_data");
  if (lhs_data.dtype() != rhs_data.dtype()) {
    throw std::invalid_argument(
        "csr sparse inner product operands must use the same value dtype.");
  }
  require_same_index_dtype(lhs_indices, lhs_indptr,
                           "csr inner product lhs_indices",
                           "csr inner product lhs_indptr");
  require_same_index_dtype(rhs_indices, rhs_indptr,
                           "csr inner product rhs_indices",
                           "csr inner product rhs_indptr");
  if (lhs_indices.dtype() != rhs_indices.dtype()) {
    throw std::invalid_argument(
        "csr sparse inner product operands must use the same index dtype.");
  }
  require_size(lhs_indptr, n_rows + 1, "csr inner product lhs_indptr");
  require_size(rhs_indptr, n_rows + 1, "csr inner product rhs_indptr");
  if (lhs_indices.size() != lhs_data.size() ||
      rhs_indices.size() != rhs_data.size()) {
    throw std::invalid_argument(
        "csr sparse inner product data and indices must have equal lengths.");
  }

  auto stream = mx::to_stream(s);
  auto lhs_data_contig = mx::contiguous(lhs_data, false, stream);
  auto lhs_indices_contig = mx::contiguous(lhs_indices, false, stream);
  auto lhs_indptr_contig = mx::contiguous(lhs_indptr, false, stream);
  auto rhs_data_contig = mx::contiguous(rhs_data, false, stream);
  auto rhs_indices_contig = mx::contiguous(rhs_indices, false, stream);
  auto rhs_indptr_contig = mx::contiguous(rhs_indptr, false, stream);

  return mx::array(
      mx::Shape{}, lhs_data.dtype(),
      std::make_shared<CSRVdot>(stream, n_rows, n_cols, conjugate_lhs),
      {lhs_data_contig, lhs_indices_contig, lhs_indptr_contig, rhs_data_contig,
       rhs_indices_contig, rhs_indptr_contig});
}

mx::array csr_vdot(const mx::array &lhs_data, const mx::array &lhs_indices,
                   const mx::array &lhs_indptr, const mx::array &rhs_data,
                   const mx::array &rhs_indices, const mx::array &rhs_indptr,
                   int n_rows, int n_cols, mx::StreamOrDevice s) {
  return csr_inner_product(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                           rhs_indices, rhs_indptr, n_rows, n_cols, true, s);
}

mx::array csr_dot(const mx::array &lhs_data, const mx::array &lhs_indices,
                  const mx::array &lhs_indptr, const mx::array &rhs_data,
                  const mx::array &rhs_indices, const mx::array &rhs_indptr,
                  int n_rows, int n_cols, mx::StreamOrDevice s) {
  return csr_inner_product(lhs_data, lhs_indices, lhs_indptr, rhs_data,
                           rhs_indices, rhs_indptr, n_rows, n_cols, false, s);
}

} // namespace mlx_sparse
