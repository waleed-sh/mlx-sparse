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

template <typename T> inline bool csc_matmat_nonzero(T value) {
  return !(value == T(0));
}

template <> inline bool csc_matmat_nonzero<complex64_t>(complex64_t value) {
  return value.real != 0.0f || value.imag != 0.0f;
}

template <typename LhsI, typename RhsI>
inline bool csc_matmat_first_candidate(device const LhsI *lhs_indices,
                                       device const LhsI *lhs_indptr,
                                       device const RhsI *rhs_indices,
                                       device const RhsI *rhs_indptr, int col,
                                       RhsI rhs_pos, LhsI lhs_pos, LhsI row,
                                       int lhs_n_cols) {
  for (RhsI prev_rhs = rhs_indptr[col]; prev_rhs <= rhs_pos; ++prev_rhs) {
    const int prev_lhs_col = static_cast<int>(rhs_indices[prev_rhs]);
    if (prev_lhs_col < 0 || prev_lhs_col >= lhs_n_cols) {
      continue;
    }
    const LhsI prev_end =
        prev_rhs == rhs_pos ? lhs_pos : lhs_indptr[prev_lhs_col + 1];
    for (LhsI prev_lhs = lhs_indptr[prev_lhs_col]; prev_lhs < prev_end;
         ++prev_lhs) {
      if (lhs_indices[prev_lhs] == row) {
        return false;
      }
    }
  }
  return true;
}

template <typename LhsI, typename RhsI, typename OutI>
[[kernel]] void
csc_matmat_symbolic_kernel(device const LhsI *lhs_indices [[buffer(0)]],
                           device const LhsI *lhs_indptr [[buffer(1)]],
                           device const RhsI *rhs_indices [[buffer(2)]],
                           device const RhsI *rhs_indptr [[buffer(3)]],
                           device OutI *counts [[buffer(4)]],
                           constant int &lhs_n_rows [[buffer(5)]],
                           constant int &lhs_n_cols [[buffer(6)]],
                           constant int &rhs_n_cols [[buffer(7)]],
                           uint col [[thread_position_in_grid]]) {
  if (col >= static_cast<uint>(rhs_n_cols)) {
    return;
  }

  OutI count = OutI(0);
  for (RhsI rhs_pos = rhs_indptr[col]; rhs_pos < rhs_indptr[col + 1];
       ++rhs_pos) {
    const int lhs_col = static_cast<int>(rhs_indices[rhs_pos]);
    if (lhs_col < 0 || lhs_col >= lhs_n_cols) {
      continue;
    }
    for (LhsI lhs_pos = lhs_indptr[lhs_col]; lhs_pos < lhs_indptr[lhs_col + 1];
         ++lhs_pos) {
      const LhsI row = lhs_indices[lhs_pos];
      if (row < LhsI(0) || row >= LhsI(lhs_n_rows)) {
        continue;
      }
      if (csc_matmat_first_candidate<LhsI, RhsI>(
              lhs_indices, lhs_indptr, rhs_indices, rhs_indptr,
              static_cast<int>(col), rhs_pos, lhs_pos, row, lhs_n_cols)) {
        count += OutI(1);
      }
    }
  }
  counts[col] = count;
}

template <typename T, typename LhsI, typename RhsI, typename OutI>
[[kernel]] void
csc_matmat_numeric_kernel(device const T *lhs_data [[buffer(0)]],
                          device const LhsI *lhs_indices [[buffer(1)]],
                          device const LhsI *lhs_indptr [[buffer(2)]],
                          device const T *rhs_data [[buffer(3)]],
                          device const RhsI *rhs_indices [[buffer(4)]],
                          device const RhsI *rhs_indptr [[buffer(5)]],
                          device const OutI *out_indptr [[buffer(6)]],
                          device T *out_data [[buffer(7)]],
                          device OutI *out_indices [[buffer(8)]],
                          constant int &lhs_n_rows [[buffer(9)]],
                          constant int &lhs_n_cols [[buffer(10)]],
                          constant int &rhs_n_cols [[buffer(11)]],
                          uint col [[thread_position_in_grid]]) {
  if (col >= static_cast<uint>(rhs_n_cols)) {
    return;
  }

  typedef typename sparse_accumulator<T>::type acc_t;
  for (RhsI rhs_pos = rhs_indptr[col]; rhs_pos < rhs_indptr[col + 1];
       ++rhs_pos) {
    const int lhs_col = static_cast<int>(rhs_indices[rhs_pos]);
    if (lhs_col < 0 || lhs_col >= lhs_n_cols) {
      continue;
    }
    for (LhsI lhs_pos = lhs_indptr[lhs_col]; lhs_pos < lhs_indptr[lhs_col + 1];
         ++lhs_pos) {
      const LhsI row = lhs_indices[lhs_pos];
      if (row < LhsI(0) || row >= LhsI(lhs_n_rows)) {
        continue;
      }
      if (!csc_matmat_first_candidate<LhsI, RhsI>(
              lhs_indices, lhs_indptr, rhs_indices, rhs_indptr,
              static_cast<int>(col), rhs_pos, lhs_pos, row, lhs_n_cols)) {
        continue;
      }

      OutI rank = OutI(0);
      for (RhsI rank_rhs = rhs_indptr[col]; rank_rhs < rhs_indptr[col + 1];
           ++rank_rhs) {
        const int rank_lhs_col = static_cast<int>(rhs_indices[rank_rhs]);
        if (rank_lhs_col < 0 || rank_lhs_col >= lhs_n_cols) {
          continue;
        }
        for (LhsI rank_lhs = lhs_indptr[rank_lhs_col];
             rank_lhs < lhs_indptr[rank_lhs_col + 1]; ++rank_lhs) {
          const LhsI rank_row = lhs_indices[rank_lhs];
          if (rank_row < row && csc_matmat_first_candidate<LhsI, RhsI>(
                                    lhs_indices, lhs_indptr, rhs_indices,
                                    rhs_indptr, static_cast<int>(col), rank_rhs,
                                    rank_lhs, rank_row, lhs_n_cols)) {
            rank += OutI(1);
          }
        }
      }

      acc_t acc = sparse_accumulator<T>::zero();
      for (RhsI sum_rhs = rhs_indptr[col]; sum_rhs < rhs_indptr[col + 1];
           ++sum_rhs) {
        const int sum_lhs_col = static_cast<int>(rhs_indices[sum_rhs]);
        if (sum_lhs_col < 0 || sum_lhs_col >= lhs_n_cols) {
          continue;
        }
        const T rhs_value = rhs_data[sum_rhs];
        for (LhsI sum_lhs = lhs_indptr[sum_lhs_col];
             sum_lhs < lhs_indptr[sum_lhs_col + 1]; ++sum_lhs) {
          if (lhs_indices[sum_lhs] == row) {
            acc += sparse_multiply<T>(lhs_data[sum_lhs], rhs_value);
          }
        }
      }

      const OutI dst = out_indptr[col] + rank;
      out_indices[dst] = OutI(row);
      out_data[dst] = sparse_accumulator<T>::cast(acc);
    }
  }
}

template <typename T, typename I>
[[kernel]] void csc_matmat_prune_counts_kernel(
    device const T *data [[buffer(0)]], device const I *indptr [[buffer(1)]],
    device I *counts [[buffer(2)]], constant int &n_cols [[buffer(3)]],
    uint col [[thread_position_in_grid]]) {
  if (col >= static_cast<uint>(n_cols)) {
    return;
  }

  I count = I(0);
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    if (csc_matmat_nonzero<T>(data[p])) {
      count += I(1);
    }
  }
  counts[col] = count;
}

template <typename T, typename I>
[[kernel]] void csc_matmat_prune_fill_kernel(
    device const T *data [[buffer(0)]], device const I *indices [[buffer(1)]],
    device const I *indptr [[buffer(2)]],
    device const I *out_indptr [[buffer(3)]], device T *out_data [[buffer(4)]],
    device I *out_indices [[buffer(5)]], constant int &n_cols [[buffer(6)]],
    uint col [[thread_position_in_grid]]) {
  if (col >= static_cast<uint>(n_cols)) {
    return;
  }

  I write = out_indptr[col];
  for (I p = indptr[col]; p < indptr[col + 1]; ++p) {
    const T value = data[p];
    if (csc_matmat_nonzero<T>(value)) {
      out_data[write] = value;
      out_indices[write] = indices[p];
      ++write;
    }
  }
}

#define INSTANTIATE_CSC_MATMAT_SYMBOLIC(LNAME, RNAME, ONAME, LHSI, RHSI, OUTI) \
  template [[host_name("csc_matmat_symbolic_" #LNAME "_" #RNAME                \
                       "_" #ONAME)]] [[kernel]] void                           \
  csc_matmat_symbolic_kernel<LHSI, RHSI, OUTI>(                                \
      device const LHSI *, device const LHSI *, device const RHSI *,           \
      device const RHSI *, device OUTI *, constant int &, constant int &,      \
      constant int &, uint)

INSTANTIATE_CSC_MATMAT_SYMBOLIC(int32, int32, int32, int, int, int);
INSTANTIATE_CSC_MATMAT_SYMBOLIC(int32, int32, int64, int, int, long);
INSTANTIATE_CSC_MATMAT_SYMBOLIC(int32, int64, int64, int, long, long);
INSTANTIATE_CSC_MATMAT_SYMBOLIC(int64, int32, int64, long, int, long);
INSTANTIATE_CSC_MATMAT_SYMBOLIC(int64, int64, int64, long, long, long);

#undef INSTANTIATE_CSC_MATMAT_SYMBOLIC

#define INSTANTIATE_CSC_MATMAT_NUMERIC(VNAME, LNAME, RNAME, ONAME, T, LHSI,    \
                                       RHSI, OUTI)                             \
  template [[host_name("csc_matmat_numeric_" #VNAME "_" #LNAME "_" #RNAME      \
                       "_" #ONAME)]] [[kernel]] void                           \
  csc_matmat_numeric_kernel<T, LHSI, RHSI, OUTI>(                              \
      device const T *, device const LHSI *, device const LHSI *,              \
      device const T *, device const RHSI *, device const RHSI *,              \
      device const OUTI *, device T *, device OUTI *, constant int &,          \
      constant int &, constant int &, uint)

#define INSTANTIATE_CSC_MATMAT_NUMERIC_INDEXES(VNAME, T)                       \
  INSTANTIATE_CSC_MATMAT_NUMERIC(VNAME, int32, int32, int32, T, int, int,      \
                                 int);                                         \
  INSTANTIATE_CSC_MATMAT_NUMERIC(VNAME, int32, int32, int64, T, int, int,      \
                                 long);                                        \
  INSTANTIATE_CSC_MATMAT_NUMERIC(VNAME, int32, int64, int64, T, int, long,     \
                                 long);                                        \
  INSTANTIATE_CSC_MATMAT_NUMERIC(VNAME, int64, int32, int64, T, long, int,     \
                                 long);                                        \
  INSTANTIATE_CSC_MATMAT_NUMERIC(VNAME, int64, int64, int64, T, long, long,    \
                                 long)

INSTANTIATE_CSC_MATMAT_NUMERIC_INDEXES(float32, float);
INSTANTIATE_CSC_MATMAT_NUMERIC_INDEXES(float16, half);
INSTANTIATE_CSC_MATMAT_NUMERIC_INDEXES(bfloat16, bfloat16_t);
INSTANTIATE_CSC_MATMAT_NUMERIC_INDEXES(complex64, complex64_t);

#undef INSTANTIATE_CSC_MATMAT_NUMERIC_INDEXES
#undef INSTANTIATE_CSC_MATMAT_NUMERIC

#define INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS(NAME, T, I)                        \
  template [[host_name("csc_matmat_prune_counts_" #NAME)]] [[kernel]] void     \
  csc_matmat_prune_counts_kernel<T, I>(device const T *, device const I *,     \
                                       device I *, constant int &, uint)

INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS(float32_int32, float, int);
INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS(float32_int64, float, long);
INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS(float16_int32, half, int);
INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS(float16_int64, half, long);
INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_MATMAT_PRUNE_COUNTS

#define INSTANTIATE_CSC_MATMAT_PRUNE_FILL(NAME, T, I)                          \
  template [[host_name("csc_matmat_prune_fill_" #NAME)]] [[kernel]] void       \
  csc_matmat_prune_fill_kernel<T, I>(                                          \
      device const T *, device const I *, device const I *, device const I *,  \
      device T *, device I *, constant int &, uint)

INSTANTIATE_CSC_MATMAT_PRUNE_FILL(float32_int32, float, int);
INSTANTIATE_CSC_MATMAT_PRUNE_FILL(float32_int64, float, long);
INSTANTIATE_CSC_MATMAT_PRUNE_FILL(float16_int32, half, int);
INSTANTIATE_CSC_MATMAT_PRUNE_FILL(float16_int64, half, long);
INSTANTIATE_CSC_MATMAT_PRUNE_FILL(bfloat16_int32, bfloat16_t, int);
INSTANTIATE_CSC_MATMAT_PRUNE_FILL(bfloat16_int64, bfloat16_t, long);
INSTANTIATE_CSC_MATMAT_PRUNE_FILL(complex64_int32, complex64_t, int);
INSTANTIATE_CSC_MATMAT_PRUNE_FILL(complex64_int64, complex64_t, long);

#undef INSTANTIATE_CSC_MATMAT_PRUNE_FILL
