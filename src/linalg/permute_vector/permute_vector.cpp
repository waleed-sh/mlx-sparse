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

#include "linalg/permute_vector/permute_vector.h"

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

class CSRPermuteVector : public mx::Primitive {
public:
  explicit CSRPermuteVector(mx::Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRPermuteVector"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

void csr_permute_vector_cpu_impl(const mx::array &x, const mx::array &perm,
                                 mx::array &out, mx::Stream stream) {
  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(x);
  encoder.set_input_array(perm);
  encoder.set_output_array(out);

  encoder.dispatch([x = mx::array::unsafe_weak_copy(x),
                    perm = mx::array::unsafe_weak_copy(perm),
                    out = mx::array::unsafe_weak_copy(out)]() mutable {
    const auto *x_ptr = x.data<float>();
    const auto *perm_ptr = perm.data<int32_t>();
    auto *out_ptr = out.data<float>();
    for (size_t i = 0; i < out.size(); ++i) {
      out_ptr[i] = x_ptr[perm_ptr[i]];
    }
  });
}

} // namespace

void CSRPermuteVector::eval_cpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  csr_permute_vector_cpu_impl(inputs[0], inputs[1], outputs[0], stream());
}

#ifdef _METAL_
void CSRPermuteVector::eval_gpu(const std::vector<mx::array> &inputs,
                                std::vector<mx::array> &outputs) {
  auto &x = inputs[0];
  auto &perm = inputs[1];
  auto &out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto *kernel = device.get_kernel("csr_permute_vector_float32", lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(perm, 1);
  encoder.set_output_array(out, 2);
  int size = static_cast<int>(out.size());
  encoder.set_bytes(size, 3);
  auto threads = std::max<size_t>(out.size(), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRPermuteVector::eval_gpu(const std::vector<mx::array> &,
                                std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_permute_vector has no GPU implementation in this build.");
}
#endif

mx::array csr_permute_vector(const mx::array &x, const mx::array &perm,
                             mx::StreamOrDevice s) {
  require_rank(x, 1, "csr_permute_vector x");
  require_rank(perm, 1, "csr_permute_vector perm");
  require_linalg_float32(x, "csr_permute_vector x");
  if (perm.dtype() != mx::int32) {
    throw std::invalid_argument(
        "csr_permute_vector perm must have dtype int32.");
  }
  require_size(perm, static_cast<int>(x.size()), "csr_permute_vector perm");

  auto stream = mx::to_stream(s);
  auto x_contig = mx::contiguous(x, false, stream);
  auto perm_contig = mx::contiguous(perm, false, stream);
  return mx::array(mx::Shape{static_cast<int>(x.size())}, mx::float32,
                   std::make_shared<CSRPermuteVector>(stream),
                   {x_contig, perm_contig});
}

} // namespace mlx_sparse
