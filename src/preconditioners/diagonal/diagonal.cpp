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

#include "preconditioners/diagonal/diagonal.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#endif

#include "linalg/common/common.h"

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

class DiagonalPreconditionerApply : public mx::Primitive {
public:
  DiagonalPreconditionerApply(mx::Stream stream, int n_rows, int rhs_cols)
      : Primitive(stream), n_rows_(n_rows), rhs_cols_(rhs_cols) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "DiagonalPreconditionerApply"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const DiagonalPreconditionerApply &>(other);
    return n_rows_ == rhs.n_rows_ && rhs_cols_ == rhs.rhs_cols_;
  }

private:
  int n_rows_;
  int rhs_cols_;
};

void diagonal_apply_cpu_impl(const mx::array &inv_diag, const mx::array &rhs,
                             mx::array &out, int n_rows, int rhs_cols,
                             mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(inv_diag);
  encoder.set_input_array(rhs);
  encoder.set_output_array(out);

  encoder.dispatch([inv_diag = mx::array::unsafe_weak_copy(inv_diag),
                    rhs = mx::array::unsafe_weak_copy(rhs),
                    out = mx::array::unsafe_weak_copy(out), n_rows,
                    rhs_cols]() mutable {
    const auto *inv_diag_ptr = inv_diag.data<float>();
    const auto *rhs_ptr = rhs.data<float>();
    auto *out_ptr = out.data<float>();
    if (rhs_cols == 1) {
      for (int row = 0; row < n_rows; ++row) {
        out_ptr[row] = inv_diag_ptr[row] * rhs_ptr[row];
      }
      return;
    }
    for (int row = 0; row < n_rows; ++row) {
      const float scale = inv_diag_ptr[row];
      const int offset = row * rhs_cols;
      for (int col = 0; col < rhs_cols; ++col) {
        out_ptr[offset + col] = scale * rhs_ptr[offset + col];
      }
    }
  });
}

} // namespace

void DiagonalPreconditionerApply::eval_cpu(const std::vector<mx::array> &inputs,
                                           std::vector<mx::array> &outputs) {
  diagonal_apply_cpu_impl(inputs[0], inputs[1], outputs[0], n_rows_, rhs_cols_,
                          stream());
}

#ifdef _METAL_
void DiagonalPreconditionerApply::eval_gpu(const std::vector<mx::array> &inputs,
                                           std::vector<mx::array> &outputs) {
  auto &inv_diag = inputs[0];
  auto &rhs = inputs[1];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel =
      device.get_kernel("diagonal_preconditioner_apply_float32", lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(inv_diag, 0);
  encoder.set_input_array(rhs, 1);
  encoder.set_output_array(out, 2);
  encoder.set_bytes(n_rows_, 3);
  encoder.set_bytes(rhs_cols_, 4);
  const size_t total = static_cast<size_t>(std::max(n_rows_, 1)) *
                       static_cast<size_t>(rhs_cols_);
  const auto group = std::min(total, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(total, 1, 1), MTL::Size(group, 1, 1));
}
#else
void DiagonalPreconditionerApply::eval_gpu(const std::vector<mx::array> &,
                                           std::vector<mx::array> &) {
  throw std::runtime_error(
      "diagonal_preconditioner_apply has no GPU implementation in this build.");
}
#endif

mx::array diagonal_preconditioner_apply(const mx::array &inv_diag,
                                        const mx::array &rhs,
                                        mx::StreamOrDevice s) {
  require_rank(inv_diag, 1, "diagonal_preconditioner_apply inv_diag");
  if (rhs.ndim() != 1 && rhs.ndim() != 2) {
    throw std::invalid_argument(
        "diagonal_preconditioner_apply rhs must be rank-1 or rank-2.");
  }
  require_linalg_float32(inv_diag, "diagonal_preconditioner_apply inv_diag");
  require_linalg_float32(rhs, "diagonal_preconditioner_apply rhs");
  const int n_rows = static_cast<int>(inv_diag.size());
  if (rhs.shape(0) != n_rows) {
    throw std::invalid_argument(
        "diagonal_preconditioner_apply rhs leading dimension must match "
        "inv_diag length.");
  }
  const int rhs_cols = rhs.ndim() == 2 ? rhs.shape(1) : 1;
  if (rhs_cols <= 0) {
    throw std::invalid_argument(
        "diagonal_preconditioner_apply rhs must have at least one column.");
  }

  auto stream = mx::to_stream(s);
  auto inv_diag_contig = mx::contiguous(inv_diag, false, stream);
  auto rhs_contig = mx::contiguous(rhs, false, stream);
  return mx::array(
      rhs.shape(), mx::float32,
      std::make_shared<DiagonalPreconditionerApply>(stream, n_rows, rhs_cols),
      {inv_diag_contig, rhs_contig});
}

} // namespace mlx_sparse
