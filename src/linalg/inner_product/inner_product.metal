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

#include "linalg/common/metal_common.h"

template <typename T, typename I, bool ConjugateLhs>
[[kernel]] void csr_vdot_kernel(device const T *lhs_data [[buffer(0)]],
                                device const I *lhs_indices [[buffer(1)]],
                                device const I *lhs_indptr [[buffer(2)]],
                                device const T *rhs_data [[buffer(3)]],
                                device const I *rhs_indices [[buffer(4)]],
                                device const I *rhs_indptr [[buffer(5)]],
                                device T *out [[buffer(6)]],
                                constant int &n_rows [[buffer(7)]],
                                constant int &n_cols [[buffer(8)]],
                                uint lane [[thread_index_in_threadgroup]]) {
  (void)n_cols;
  typedef typename sparse_accumulator<T>::type acc_t;
  threadgroup acc_t scratch[256];
  acc_t local = sparse_accumulator<T>::zero();
  for (int row = static_cast<int>(lane); row < n_rows;
       row += static_cast<int>(k_linalg_threads)) {
    I lp = lhs_indptr[row];
    I rp = rhs_indptr[row];
    const I lend = lhs_indptr[row + 1];
    const I rend = rhs_indptr[row + 1];
    while (lp < lend && rp < rend) {
      const I lc = lhs_indices[lp];
      const I rc = rhs_indices[rp];
      if (lc == rc) {
        const T lhs =
            ConjugateLhs ? sparse_conjugate(lhs_data[lp]) : lhs_data[lp];
        local += sparse_multiply<T>(lhs, rhs_data[rp]);
        ++lp;
        ++rp;
      } else if (lc < rc) {
        ++lp;
      } else {
        ++rp;
      }
    }
  }
  const acc_t reduced = reduce_sum_256_any<acc_t>(local, scratch, lane);
  if (lane == 0) {
    out[0] = sparse_accumulator<T>::cast(reduced);
  }
}

#define INSTANTIATE_CSR_INNER(OP, NAME, T, I, CONJ)                            \
  template [[host_name(#OP "_" #NAME)]] [[kernel]] void                        \
  csr_vdot_kernel<T, I, CONJ>(device const T *, device const I *,              \
                              device const I *, device const T *,              \
                              device const I *, device const I *, device T *,  \
                              constant int &, constant int &, uint)

INSTANTIATE_CSR_INNER(csr_vdot, float32_int32, float, int, true);
INSTANTIATE_CSR_INNER(csr_vdot, float32_int64, float, long, true);
INSTANTIATE_CSR_INNER(csr_vdot, complex64_int32, complex64_t, int, true);
INSTANTIATE_CSR_INNER(csr_vdot, complex64_int64, complex64_t, long, true);
INSTANTIATE_CSR_INNER(csr_dot, float32_int32, float, int, false);
INSTANTIATE_CSR_INNER(csr_dot, float32_int64, float, long, false);
INSTANTIATE_CSR_INNER(csr_dot, complex64_int32, complex64_t, int, false);
INSTANTIATE_CSR_INNER(csr_dot, complex64_int64, complex64_t, long, false);

#undef INSTANTIATE_CSR_INNER
