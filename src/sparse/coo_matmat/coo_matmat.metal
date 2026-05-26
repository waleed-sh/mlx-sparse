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

#include "common/metal_common.h"

template <typename T> inline bool coo_matmat_nonzero(T value) {
  return !(value == T(0));
}

template <> inline bool coo_matmat_nonzero<complex64_t>(complex64_t value) {
  return value.real != 0.0f || value.imag != 0.0f;
}

template <typename LhsI, typename RhsI>
inline bool coo_matmat_first_candidate(device const LhsI *lhs_indices,
                                       device const LhsI *lhs_indptr,
                                       device const RhsI *rhs_indices,
                                       device const RhsI *rhs_indptr, int row,
                                       LhsI lhs_pos, RhsI rhs_pos, RhsI col) {
  for (LhsI prev_lhs = lhs_indptr[row]; prev_lhs <= lhs_pos; ++prev_lhs) {
    const int prev_rhs_row = static_cast<int>(lhs_indices[prev_lhs]);
    const RhsI prev_end =
        prev_lhs == lhs_pos ? rhs_pos : rhs_indptr[prev_rhs_row + 1];
    for (RhsI prev_rhs = rhs_indptr[prev_rhs_row]; prev_rhs < prev_end;
         ++prev_rhs) {
      if (rhs_indices[prev_rhs] == col) {
        return false;
      }
    }
  }
  return true;
}

template <typename LhsI, typename RhsI, typename OutI>
[[kernel]] void
coo_matmat_symbolic_kernel(device const LhsI *lhs_indices [[buffer(0)]],
                           device const LhsI *lhs_indptr [[buffer(1)]],
                           device const RhsI *rhs_indices [[buffer(2)]],
                           device const RhsI *rhs_indptr [[buffer(3)]],
                           device OutI *counts [[buffer(4)]],
                           constant int &lhs_n_rows [[buffer(5)]],
                           constant int &rhs_n_rows [[buffer(6)]],
                           constant int &rhs_n_cols [[buffer(7)]],
                           uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(lhs_n_rows)) {
    return;
  }

  OutI count = OutI(0);
  for (LhsI lhs_pos = lhs_indptr[row]; lhs_pos < lhs_indptr[row + 1];
       ++lhs_pos) {
    const int rhs_row = static_cast<int>(lhs_indices[lhs_pos]);
    if (rhs_row < 0 || rhs_row >= rhs_n_rows) {
      continue;
    }
    for (RhsI rhs_pos = rhs_indptr[rhs_row]; rhs_pos < rhs_indptr[rhs_row + 1];
         ++rhs_pos) {
      const RhsI col = rhs_indices[rhs_pos];
      if (col < RhsI(0) || col >= RhsI(rhs_n_cols)) {
        continue;
      }
      if (coo_matmat_first_candidate<LhsI, RhsI>(
              lhs_indices, lhs_indptr, rhs_indices, rhs_indptr,
              static_cast<int>(row), lhs_pos, rhs_pos, col)) {
        count += OutI(1);
      }
    }
  }
  counts[row] = count;
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
[[kernel]] void
coo_matmat_numeric_kernel(device const T *lhs_data [[buffer(0)]],
                          device const LhsI *lhs_indices [[buffer(1)]],
                          device const LhsI *lhs_indptr [[buffer(2)]],
                          device const T *rhs_data [[buffer(3)]],
                          device const RhsI *rhs_indices [[buffer(4)]],
                          device const RhsI *rhs_indptr [[buffer(5)]],
                          device const OutI *out_indptr [[buffer(6)]],
                          device T *out_data [[buffer(7)]],
                          device OutI *out_col [[buffer(8)]],
                          constant int &lhs_n_rows [[buffer(9)]],
                          constant int &rhs_n_rows [[buffer(10)]],
                          constant int &rhs_n_cols [[buffer(11)]],
                          uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(lhs_n_rows)) {
    return;
  }

  typedef typename sparse_accumulator<T>::type acc_t;
  for (LhsI lhs_pos = lhs_indptr[row]; lhs_pos < lhs_indptr[row + 1];
       ++lhs_pos) {
    const int rhs_row = static_cast<int>(lhs_indices[lhs_pos]);
    if (rhs_row < 0 || rhs_row >= rhs_n_rows) {
      continue;
    }
    for (RhsI rhs_pos = rhs_indptr[rhs_row]; rhs_pos < rhs_indptr[rhs_row + 1];
         ++rhs_pos) {
      const RhsI col = rhs_indices[rhs_pos];
      if (col < RhsI(0) || col >= RhsI(rhs_n_cols)) {
        continue;
      }
      if (!coo_matmat_first_candidate<LhsI, RhsI>(
              lhs_indices, lhs_indptr, rhs_indices, rhs_indptr,
              static_cast<int>(row), lhs_pos, rhs_pos, col)) {
        continue;
      }

      OutI rank = OutI(0);
      for (LhsI rank_lhs = lhs_indptr[row]; rank_lhs < lhs_indptr[row + 1];
           ++rank_lhs) {
        const int rank_rhs_row = static_cast<int>(lhs_indices[rank_lhs]);
        if (rank_rhs_row < 0 || rank_rhs_row >= rhs_n_rows) {
          continue;
        }
        for (RhsI rank_rhs = rhs_indptr[rank_rhs_row];
             rank_rhs < rhs_indptr[rank_rhs_row + 1]; ++rank_rhs) {
          const RhsI rank_col = rhs_indices[rank_rhs];
          if (rank_col < col &&
              coo_matmat_first_candidate<LhsI, RhsI>(
                  lhs_indices, lhs_indptr, rhs_indices, rhs_indptr,
                  static_cast<int>(row), rank_lhs, rank_rhs, rank_col)) {
            rank += OutI(1);
          }
        }
      }

      acc_t acc = sparse_accumulator<T>::zero();
      for (LhsI sum_lhs = lhs_indptr[row]; sum_lhs < lhs_indptr[row + 1];
           ++sum_lhs) {
        const int sum_rhs_row = static_cast<int>(lhs_indices[sum_lhs]);
        if (sum_rhs_row < 0 || sum_rhs_row >= rhs_n_rows) {
          continue;
        }
        const T lhs_value = lhs_data[sum_lhs];
        for (RhsI sum_rhs = rhs_indptr[sum_rhs_row];
             sum_rhs < rhs_indptr[sum_rhs_row + 1]; ++sum_rhs) {
          if (rhs_indices[sum_rhs] == col) {
            acc += sparse_multiply<T>(lhs_value, rhs_data[sum_rhs]);
          }
        }
      }

      const OutI dst = out_indptr[row] + rank;
      out_col[dst] = OutI(col);
      out_data[dst] = sparse_accumulator<T>::cast(acc);
    }
  }
}

template <typename T, typename I>
[[kernel]] void coo_matmat_prune_counts_kernel(
    device const T *data [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device I *counts [[buffer(2)]], constant int &n_rows [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(n_rows)) {
    return;
  }

  I count = I(0);
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    if (coo_matmat_nonzero<T>(data[p])) {
      count += I(1);
    }
  }
  counts[row] = count;
}

template <typename T, typename I>
[[kernel]] void coo_matmat_prune_fill_kernel(
    device const T *data [[buffer(0)]], device const I *col [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
    device const I *out_indptr [[buffer(3)]], device T *out_data [[buffer(4)]],
    device I *out_row [[buffer(5)]], device I *out_col [[buffer(6)]],
    constant int &n_rows [[buffer(7)]], uint row [[thread_position_in_grid]]) {
  if (row >= static_cast<uint>(n_rows)) {
    return;
  }

  I write = out_indptr[row];
  for (I p = indptr[row]; p < indptr[row + 1]; ++p) {
    const T value = data[p];
    if (coo_matmat_nonzero<T>(value)) {
      out_data[write] = value;
      out_row[write] = I(row);
      out_col[write] = col[p];
      ++write;
    }
  }
}

#define INSTANTIATE_COO_MATMAT_SYMBOLIC(LNAME, RNAME, ONAME, LHSI, RHSI, OUTI) \
  template [[host_name("coo_matmat_symbolic_" #LNAME "_" #RNAME                \
                       "_" #ONAME)]] [[kernel]] void                           \
  coo_matmat_symbolic_kernel<LHSI, RHSI, OUTI>(                                \
      device const LHSI *, device const LHSI *, device const RHSI *,           \
      device const RHSI *, device OUTI *, constant int &, constant int &,      \
      constant int &, uint)

INSTANTIATE_COO_MATMAT_SYMBOLIC(int32, int32, int32, int, int, int);
INSTANTIATE_COO_MATMAT_SYMBOLIC(int32, int32, int64, int, int, long);
INSTANTIATE_COO_MATMAT_SYMBOLIC(int32, int64, int64, int, long, long);
INSTANTIATE_COO_MATMAT_SYMBOLIC(int64, int32, int64, long, int, long);
INSTANTIATE_COO_MATMAT_SYMBOLIC(int64, int64, int64, long, long, long);

#undef INSTANTIATE_COO_MATMAT_SYMBOLIC

#define INSTANTIATE_COO_MATMAT_NUMERIC(VNAME, LNAME, RNAME, ONAME, T, LHSI,    \
                                       RHSI, OUTI)                             \
  template [[host_name("coo_matmat_numeric_" #VNAME "_" #LNAME "_" #RNAME      \
                       "_" #ONAME)]] [[kernel]] void                           \
  coo_matmat_numeric_kernel<T, LHSI, RHSI, OUTI>(                              \
      device const T *, device const LHSI *, device const LHSI *,              \
      device const T *, device const RHSI *, device const RHSI *,              \
      device const OUTI *, device T *, device OUTI *, constant int &,          \
      constant int &, constant int &, uint)

#define INSTANTIATE_COO_MATMAT_NUMERIC_INDEXES(VNAME, T)                       \
  INSTANTIATE_COO_MATMAT_NUMERIC(VNAME, int32, int32, int32, T, int, int,      \
                                 int);                                         \
  INSTANTIATE_COO_MATMAT_NUMERIC(VNAME, int32, int32, int64, T, int, int,      \
                                 long);                                        \
  INSTANTIATE_COO_MATMAT_NUMERIC(VNAME, int32, int64, int64, T, int, long,     \
                                 long);                                        \
  INSTANTIATE_COO_MATMAT_NUMERIC(VNAME, int64, int32, int64, T, long, int,     \
                                 long);                                        \
  INSTANTIATE_COO_MATMAT_NUMERIC(VNAME, int64, int64, int64, T, long, long,    \
                                 long)

INSTANTIATE_COO_MATMAT_NUMERIC_INDEXES(float32, float);
INSTANTIATE_COO_MATMAT_NUMERIC_INDEXES(float16, half);
INSTANTIATE_COO_MATMAT_NUMERIC_INDEXES(bfloat16, bfloat16_t);
INSTANTIATE_COO_MATMAT_NUMERIC_INDEXES(complex64, complex64_t);

#undef INSTANTIATE_COO_MATMAT_NUMERIC_INDEXES
#undef INSTANTIATE_COO_MATMAT_NUMERIC

#define INSTANTIATE_COO_MATMAT_PRUNE_COUNTS(NAME, T, I)                        \
  template [[host_name("coo_matmat_prune_counts_" #NAME)]] [[kernel]] void     \
  coo_matmat_prune_counts_kernel<T, I>(device const T *, device const I *,     \
                                       device I *, constant int &, uint)

INSTANTIATE_COO_MATMAT_PRUNE_COUNTS(float32_int32, float, int);
INSTANTIATE_COO_MATMAT_PRUNE_COUNTS(float32_int64, float, long);
INSTANTIATE_COO_MATMAT_PRUNE_COUNTS(float16_int32, half, int);
INSTANTIATE_COO_MATMAT_PRUNE_COUNTS(float16_int64, half, long);
INSTANTIATE_COO_MATMAT_PRUNE_COUNTS(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_COO_MATMAT_PRUNE_COUNTS(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_COO_MATMAT_PRUNE_COUNTS(complex64_int32, complex64_t, int);
INSTANTIATE_COO_MATMAT_PRUNE_COUNTS(complex64_int64, complex64_t, long);

#undef INSTANTIATE_COO_MATMAT_PRUNE_COUNTS

#define INSTANTIATE_COO_MATMAT_PRUNE_FILL(NAME, T, I)                          \
  template [[host_name("coo_matmat_prune_fill_" #NAME)]] [[kernel]] void       \
  coo_matmat_prune_fill_kernel<T, I>(                                          \
      device const T *, device const I *, device const I *, device const I *,  \
      device T *, device I *, device I *, constant int &, uint)

INSTANTIATE_COO_MATMAT_PRUNE_FILL(float32_int32, float, int);
INSTANTIATE_COO_MATMAT_PRUNE_FILL(float32_int64, float, long);
INSTANTIATE_COO_MATMAT_PRUNE_FILL(float16_int32, half, int);
INSTANTIATE_COO_MATMAT_PRUNE_FILL(float16_int64, half, long);
INSTANTIATE_COO_MATMAT_PRUNE_FILL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_COO_MATMAT_PRUNE_FILL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_COO_MATMAT_PRUNE_FILL(complex64_int32, complex64_t, int);
INSTANTIATE_COO_MATMAT_PRUNE_FILL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_COO_MATMAT_PRUNE_FILL
