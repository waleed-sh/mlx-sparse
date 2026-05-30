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

template <typename I>
[[kernel]] void csr_triangular_solve_kernel(
    device const float *data [[buffer(0)]],
    device const I *indices [[buffer(1)]], device const I *indptr [[buffer(2)]],
    device const float *b [[buffer(3)]], device float *x [[buffer(4)]],
    constant int &n_rows [[buffer(5)]], constant int &n_cols [[buffer(6)]],
    constant int &lower [[buffer(7)]],
    constant int &unit_diagonal [[buffer(8)]],
    constant int &rhs_cols [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  (void)n_cols;
  const int rhs = static_cast<int>(tid);
  if (rhs >= rhs_cols) {
    return;
  }
  if (lower != 0) {
    for (int row = 0; row < n_rows; ++row) {
      const int row_base = row * rhs_cols;
      float sum = b[row_base + rhs];
      float diag = 1.0f;
      for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices[p]);
        if (col < row) {
          sum -= data[p] * x[col * rhs_cols + rhs];
        } else if (col == row) {
          diag = data[p];
        }
      }
      x[row_base + rhs] = unit_diagonal != 0 ? sum : sum / diag;
    }
  } else {
    for (int row = n_rows - 1; row >= 0; --row) {
      const int row_base = row * rhs_cols;
      float sum = b[row_base + rhs];
      float diag = 1.0f;
      for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices[p]);
        if (col > row) {
          sum -= data[p] * x[col * rhs_cols + rhs];
        } else if (col == row) {
          diag = data[p];
        }
      }
      x[row_base + rhs] = unit_diagonal != 0 ? sum : sum / diag;
    }
  }
}

template [[host_name("csr_triangular_solve_float32_int32")]] [[kernel]] void
csr_triangular_solve_kernel<int>(device const float *, device const int *,
                                 device const int *, device const float *,
                                 device float *, constant int &, constant int &,
                                 constant int &, constant int &, constant int &,
                                 uint);

template [[host_name("csr_triangular_solve_float32_int64")]] [[kernel]] void
csr_triangular_solve_kernel<long>(device const float *, device const long *,
                                  device const long *, device const float *,
                                  device float *, constant int &,
                                  constant int &, constant int &,
                                  constant int &, constant int &, uint);
