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

#include "linalg/triangular_solve/triangular_solve.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
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

#include "common/cpu_parallel.h"
#include "linalg/common/common.h"

namespace mlx_sparse {

namespace {

using namespace linalg_detail;

class CSRTriangularSolve : public mx::Primitive {
public:
  CSRTriangularSolve(mx::Stream stream, int n_rows, int n_cols, bool lower,
                     bool unit_diagonal, int rhs_cols,
                     bool has_analysis = false)
      : Primitive(stream), n_rows_(n_rows), n_cols_(n_cols), lower_(lower),
        unit_diagonal_(unit_diagonal), rhs_cols_(rhs_cols),
        has_analysis_(has_analysis) {}

  void eval_cpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;
  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override;

  const char *name() const override { return "CSRTriangularSolve"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &rhs = static_cast<const CSRTriangularSolve &>(other);
    return n_rows_ == rhs.n_rows_ && n_cols_ == rhs.n_cols_ &&
           lower_ == rhs.lower_ && unit_diagonal_ == rhs.unit_diagonal_ &&
           rhs_cols_ == rhs.rhs_cols_ && has_analysis_ == rhs.has_analysis_;
  }

private:
  int n_rows_;
  int n_cols_;
  bool lower_;
  bool unit_diagonal_;
  int rhs_cols_;
  bool has_analysis_;
};

class LevelBarrier {
public:
  explicit LevelBarrier(int parties) : parties_(std::max(parties, 1)) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const int generation = generation_;
    if (++arrived_ == parties_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return generation != generation_; });
  }

private:
  int parties_;
  int arrived_{0};
  int generation_{0};
  std::mutex mutex_;
  std::condition_variable cv_;
};

template <typename Fn>
void parallel_for_triangular_levels(int workers, int n_levels, Fn &&fn) {
  if (n_levels <= 0) {
    return;
  }
  const int worker_count = std::max(workers, 1);
  if (worker_count == 1) {
    for (int level = 0; level < n_levels; ++level) {
      fn(0, 1, level);
    }
    return;
  }

  LevelBarrier barrier(worker_count);
  std::atomic<bool> failed{false};
  std::exception_ptr first_exception = nullptr;
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(worker_count - 1));

  auto run_worker = [&](int worker_id) {
    for (int level = 0; level < n_levels; ++level) {
      if (!failed.load(std::memory_order_relaxed)) {
        try {
          fn(worker_id, worker_count, level);
        } catch (...) {
          bool expected = false;
          if (failed.compare_exchange_strong(expected, true,
                                             std::memory_order_relaxed)) {
            first_exception = std::current_exception();
          }
        }
      }
      barrier.wait();
    }
  };

  for (int worker = 1; worker < worker_count; ++worker) {
    threads.emplace_back(run_worker, worker);
  }
  run_worker(0);
  for (auto &thread : threads) {
    thread.join();
  }
  if (first_exception != nullptr) {
    std::rethrow_exception(first_exception);
  }
}

template <typename I>
float cached_diagonal_value(const float *data_ptr, const I *indices_ptr,
                            const I *indptr_ptr,
                            const I *diagonal_positions_ptr, int row,
                            bool unit_diagonal) {
  if (unit_diagonal) {
    return 1.0f;
  }
  const I position = diagonal_positions_ptr[row];
  if (position < indptr_ptr[row] || position >= indptr_ptr[row + 1] ||
      indices_ptr[position] != static_cast<I>(row)) {
    throw std::runtime_error(
        "csr_triangular_solve encountered a missing diagonal.");
  }
  const float diag = data_ptr[position];
  if (std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
    throw std::runtime_error(
        "csr_triangular_solve encountered a zero diagonal.");
  }
  return diag;
}

template <typename I>
void csr_triangular_solve_vector_cached_row(
    const float *data_ptr, const I *indices_ptr, const I *indptr_ptr,
    const I *diagonal_positions_ptr, const float *b_ptr, float *x_ptr, int row,
    bool lower, bool unit_diagonal) {
  float sum = b_ptr[row];
  float diag = 1.0f;
  if (diagonal_positions_ptr != nullptr) {
    diag = cached_diagonal_value(data_ptr, indices_ptr, indptr_ptr,
                                 diagonal_positions_ptr, row, unit_diagonal);
  }
  for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
    const int col = static_cast<int>(indices_ptr[p]);
    if (lower ? col < row : col > row) {
      sum -= data_ptr[p] * x_ptr[col];
    } else if (diagonal_positions_ptr == nullptr && col == row) {
      diag = data_ptr[p];
    }
  }
  if (diagonal_positions_ptr == nullptr && !unit_diagonal &&
      std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
    throw std::runtime_error(
        "csr_triangular_solve encountered a zero diagonal.");
  }
  x_ptr[row] = unit_diagonal ? sum : sum / diag;
}

template <typename I>
void csr_triangular_solve_matrix_cached_row(
    const float *data_ptr, const I *indices_ptr, const I *indptr_ptr,
    const I *diagonal_positions_ptr, const float *b_ptr, float *x_ptr, int row,
    int rhs_cols, bool lower, bool unit_diagonal, float *sum) {
  const size_t row_base = static_cast<size_t>(row) * rhs_cols;
  for (int rhs = 0; rhs < rhs_cols; ++rhs) {
    sum[rhs] = b_ptr[row_base + rhs];
  }
  float diag = 1.0f;
  if (diagonal_positions_ptr != nullptr) {
    diag = cached_diagonal_value(data_ptr, indices_ptr, indptr_ptr,
                                 diagonal_positions_ptr, row, unit_diagonal);
  }
  for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
    const int col = static_cast<int>(indices_ptr[p]);
    if (lower ? col < row : col > row) {
      const float value = data_ptr[p];
      const size_t col_base = static_cast<size_t>(col) * rhs_cols;
      for (int rhs = 0; rhs < rhs_cols; ++rhs) {
        sum[rhs] -= value * x_ptr[col_base + rhs];
      }
    } else if (diagonal_positions_ptr == nullptr && col == row) {
      diag = data_ptr[p];
    }
  }
  if (diagonal_positions_ptr == nullptr && !unit_diagonal &&
      std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
    throw std::runtime_error(
        "csr_triangular_solve encountered a zero diagonal.");
  }
  if (unit_diagonal) {
    for (int rhs = 0; rhs < rhs_cols; ++rhs) {
      x_ptr[row_base + rhs] = sum[rhs];
    }
  } else {
    const float inv_diag = 1.0f / diag;
    for (int rhs = 0; rhs < rhs_cols; ++rhs) {
      x_ptr[row_base + rhs] = sum[rhs] * inv_diag;
    }
  }
}

template <typename I>
void csr_triangular_solve_vector_cpu_impl(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, mx::array &x, int n_rows, bool lower,
    bool unit_diagonal, mx::Stream stream) {
  x.set_data(mx::allocator::malloc(x.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(b);
  encoder.set_output_array(x);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    b = mx::array::unsafe_weak_copy(b),
                    x = mx::array::unsafe_weak_copy(x), n_rows, lower,
                    unit_diagonal]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    auto *x_ptr = x.data<float>();

    if (lower) {
      for (int row = 0; row < n_rows; ++row) {
        float sum = b_ptr[row];
        float diag = 1.0f;
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          const int col = static_cast<int>(indices_ptr[p]);
          if (col < row) {
            sum -= data_ptr[p] * x_ptr[col];
          } else if (col == row) {
            diag = data_ptr[p];
          }
        }
        if (!unit_diagonal &&
            std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
          throw std::runtime_error(
              "csr_triangular_solve encountered a zero diagonal.");
        }
        x_ptr[row] = unit_diagonal ? sum : sum / diag;
      }
    } else {
      for (int row = n_rows - 1; row >= 0; --row) {
        float sum = b_ptr[row];
        float diag = 1.0f;
        for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
          const int col = static_cast<int>(indices_ptr[p]);
          if (col > row) {
            sum -= data_ptr[p] * x_ptr[col];
          } else if (col == row) {
            diag = data_ptr[p];
          }
        }
        if (!unit_diagonal &&
            std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
          throw std::runtime_error(
              "csr_triangular_solve encountered a zero diagonal.");
        }
        x_ptr[row] = unit_diagonal ? sum : sum / diag;
      }
    }
  });
}

template <typename I, int kRhsCols>
void csr_triangular_solve_matrix_fixed_cols(
    const float *data_ptr, const I *indices_ptr, const I *indptr_ptr,
    const float *b_ptr, float *x_ptr, int n_rows, bool lower,
    bool unit_diagonal, const I *diagonal_positions_ptr = nullptr) {
  std::array<float, static_cast<size_t>(kRhsCols)> sum{};
  if (lower) {
    for (int row = 0; row < n_rows; ++row) {
      for (int rhs = 0; rhs < kRhsCols; ++rhs) {
        sum[static_cast<size_t>(rhs)] =
            b_ptr[static_cast<size_t>(row) * kRhsCols + rhs];
      }
      float diag = 1.0f;
      if (diagonal_positions_ptr != nullptr) {
        diag =
            cached_diagonal_value(data_ptr, indices_ptr, indptr_ptr,
                                  diagonal_positions_ptr, row, unit_diagonal);
      }
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices_ptr[p]);
        if (col < row) {
          const float value = data_ptr[p];
          const float *x_col = x_ptr + static_cast<size_t>(col) * kRhsCols;
          for (int rhs = 0; rhs < kRhsCols; ++rhs) {
            sum[static_cast<size_t>(rhs)] -= value * x_col[rhs];
          }
        } else if (diagonal_positions_ptr == nullptr && col == row) {
          diag = data_ptr[p];
        }
      }
      if (diagonal_positions_ptr == nullptr && !unit_diagonal &&
          std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error(
            "csr_triangular_solve encountered a zero diagonal.");
      }
      float *x_row = x_ptr + static_cast<size_t>(row) * kRhsCols;
      if (unit_diagonal) {
        for (int rhs = 0; rhs < kRhsCols; ++rhs) {
          x_row[rhs] = sum[static_cast<size_t>(rhs)];
        }
      } else {
        const float inv_diag = 1.0f / diag;
        for (int rhs = 0; rhs < kRhsCols; ++rhs) {
          x_row[rhs] = sum[static_cast<size_t>(rhs)] * inv_diag;
        }
      }
    }
  } else {
    for (int row = n_rows - 1; row >= 0; --row) {
      for (int rhs = 0; rhs < kRhsCols; ++rhs) {
        sum[static_cast<size_t>(rhs)] =
            b_ptr[static_cast<size_t>(row) * kRhsCols + rhs];
      }
      float diag = 1.0f;
      if (diagonal_positions_ptr != nullptr) {
        diag =
            cached_diagonal_value(data_ptr, indices_ptr, indptr_ptr,
                                  diagonal_positions_ptr, row, unit_diagonal);
      }
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices_ptr[p]);
        if (col > row) {
          const float value = data_ptr[p];
          const float *x_col = x_ptr + static_cast<size_t>(col) * kRhsCols;
          for (int rhs = 0; rhs < kRhsCols; ++rhs) {
            sum[static_cast<size_t>(rhs)] -= value * x_col[rhs];
          }
        } else if (diagonal_positions_ptr == nullptr && col == row) {
          diag = data_ptr[p];
        }
      }
      if (diagonal_positions_ptr == nullptr && !unit_diagonal &&
          std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error(
            "csr_triangular_solve encountered a zero diagonal.");
      }
      float *x_row = x_ptr + static_cast<size_t>(row) * kRhsCols;
      if (unit_diagonal) {
        for (int rhs = 0; rhs < kRhsCols; ++rhs) {
          x_row[rhs] = sum[static_cast<size_t>(rhs)];
        }
      } else {
        const float inv_diag = 1.0f / diag;
        for (int rhs = 0; rhs < kRhsCols; ++rhs) {
          x_row[rhs] = sum[static_cast<size_t>(rhs)] * inv_diag;
        }
      }
    }
  }
}

template <typename I>
void csr_triangular_solve_matrix_rhs_range(
    const float *data_ptr, const I *indices_ptr, const I *indptr_ptr,
    const float *b_ptr, float *x_ptr, int n_rows, int rhs_cols, CpuRange cols,
    bool lower, bool unit_diagonal, const I *diagonal_positions_ptr = nullptr) {
  const int width = cols.end - cols.begin;
  std::vector<float> sum(static_cast<size_t>(width));
  if (lower) {
    for (int row = 0; row < n_rows; ++row) {
      const size_t row_base = static_cast<size_t>(row) * rhs_cols;
      for (int rhs = cols.begin; rhs < cols.end; ++rhs) {
        sum[static_cast<size_t>(rhs - cols.begin)] = b_ptr[row_base + rhs];
      }
      float diag = 1.0f;
      if (diagonal_positions_ptr != nullptr) {
        diag =
            cached_diagonal_value(data_ptr, indices_ptr, indptr_ptr,
                                  diagonal_positions_ptr, row, unit_diagonal);
      }
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices_ptr[p]);
        if (col < row) {
          const float value = data_ptr[p];
          const size_t col_base = static_cast<size_t>(col) * rhs_cols;
          for (int rhs = cols.begin; rhs < cols.end; ++rhs) {
            sum[static_cast<size_t>(rhs - cols.begin)] -=
                value * x_ptr[col_base + rhs];
          }
        } else if (diagonal_positions_ptr == nullptr && col == row) {
          diag = data_ptr[p];
        }
      }
      if (diagonal_positions_ptr == nullptr && !unit_diagonal &&
          std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error(
            "csr_triangular_solve encountered a zero diagonal.");
      }
      if (unit_diagonal) {
        for (int rhs = cols.begin; rhs < cols.end; ++rhs) {
          x_ptr[row_base + rhs] = sum[static_cast<size_t>(rhs - cols.begin)];
        }
      } else {
        const float inv_diag = 1.0f / diag;
        for (int rhs = cols.begin; rhs < cols.end; ++rhs) {
          x_ptr[row_base + rhs] =
              sum[static_cast<size_t>(rhs - cols.begin)] * inv_diag;
        }
      }
    }
  } else {
    for (int row = n_rows - 1; row >= 0; --row) {
      const size_t row_base = static_cast<size_t>(row) * rhs_cols;
      for (int rhs = cols.begin; rhs < cols.end; ++rhs) {
        sum[static_cast<size_t>(rhs - cols.begin)] = b_ptr[row_base + rhs];
      }
      float diag = 1.0f;
      if (diagonal_positions_ptr != nullptr) {
        diag =
            cached_diagonal_value(data_ptr, indices_ptr, indptr_ptr,
                                  diagonal_positions_ptr, row, unit_diagonal);
      }
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices_ptr[p]);
        if (col > row) {
          const float value = data_ptr[p];
          const size_t col_base = static_cast<size_t>(col) * rhs_cols;
          for (int rhs = cols.begin; rhs < cols.end; ++rhs) {
            sum[static_cast<size_t>(rhs - cols.begin)] -=
                value * x_ptr[col_base + rhs];
          }
        } else if (diagonal_positions_ptr == nullptr && col == row) {
          diag = data_ptr[p];
        }
      }
      if (diagonal_positions_ptr == nullptr && !unit_diagonal &&
          std::abs(diag) <= std::numeric_limits<float>::epsilon()) {
        throw std::runtime_error(
            "csr_triangular_solve encountered a zero diagonal.");
      }
      if (unit_diagonal) {
        for (int rhs = cols.begin; rhs < cols.end; ++rhs) {
          x_ptr[row_base + rhs] = sum[static_cast<size_t>(rhs - cols.begin)];
        }
      } else {
        const float inv_diag = 1.0f / diag;
        for (int rhs = cols.begin; rhs < cols.end; ++rhs) {
          x_ptr[row_base + rhs] =
              sum[static_cast<size_t>(rhs - cols.begin)] * inv_diag;
        }
      }
    }
  }
}

template <typename I>
void csr_triangular_solve_matrix_rhs_cpu_impl(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, mx::array &x, int n_rows, int rhs_cols, bool lower,
    bool unit_diagonal, mx::Stream stream) {
  x.set_data(mx::allocator::malloc(x.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(b);
  encoder.set_output_array(x);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    b = mx::array::unsafe_weak_copy(b),
                    x = mx::array::unsafe_weak_copy(x), n_rows, rhs_cols, lower,
                    unit_diagonal]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    auto *x_ptr = x.data<float>();

    const int workers = configured_solver_worker_count();
    const auto ranges = equal_cpu_ranges(rhs_cols, workers);
    if (ranges.size() > 1) {
      parallel_for_cpu_ranges(ranges, [&](CpuRange cols) {
        csr_triangular_solve_matrix_rhs_range<I>(
            data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, rhs_cols,
            cols, lower, unit_diagonal);
      });
      return;
    }

    switch (rhs_cols) {
    case 2:
      csr_triangular_solve_matrix_fixed_cols<I, 2>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal);
      return;
    case 4:
      csr_triangular_solve_matrix_fixed_cols<I, 4>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal);
      return;
    case 8:
      csr_triangular_solve_matrix_fixed_cols<I, 8>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal);
      return;
    case 16:
      csr_triangular_solve_matrix_fixed_cols<I, 16>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal);
      return;
    case 32:
      csr_triangular_solve_matrix_fixed_cols<I, 32>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal);
      return;
    default:
      csr_triangular_solve_matrix_rhs_range<I>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, rhs_cols,
          {0, rhs_cols}, lower, unit_diagonal);
      return;
    }
  });
}

template <typename I>
void csr_triangular_solve_vector_analyzed_cpu_impl(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, const mx::array &diagonal_positions,
    const mx::array &level_offsets, const mx::array &level_rows, mx::array &x,
    int n_rows, bool lower, bool unit_diagonal, mx::Stream stream) {
  x.set_data(mx::allocator::malloc(x.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(b);
  encoder.set_input_array(diagonal_positions);
  encoder.set_input_array(level_offsets);
  encoder.set_input_array(level_rows);
  encoder.set_output_array(x);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    b = mx::array::unsafe_weak_copy(b),
                    diagonal_positions =
                        mx::array::unsafe_weak_copy(diagonal_positions),
                    level_offsets = mx::array::unsafe_weak_copy(level_offsets),
                    level_rows = mx::array::unsafe_weak_copy(level_rows),
                    x = mx::array::unsafe_weak_copy(x), n_rows, lower,
                    unit_diagonal]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    const auto *diagonal_positions_ptr = diagonal_positions.data<I>();
    const auto *level_offsets_ptr = level_offsets.data<int32_t>();
    const auto *level_rows_ptr = level_rows.data<int32_t>();
    auto *x_ptr = x.data<float>();

    const int workers = configured_solver_worker_count();
    const int n_levels = static_cast<int>(level_offsets.size()) - 1;
    if (workers > 1 && n_levels > 0 &&
        static_cast<int>(level_rows.size()) == n_rows) {
      parallel_for_triangular_levels(
          workers, n_levels, [&](int worker, int worker_count, int level) {
            const int begin = level_offsets_ptr[level];
            const int end = level_offsets_ptr[level + 1];
            const int width = end - begin;
            const int local_begin =
                begin +
                static_cast<int>((int64_t{worker} * width) / worker_count);
            const int local_end =
                begin +
                static_cast<int>((int64_t{worker + 1} * width) / worker_count);
            for (int p = local_begin; p < local_end; ++p) {
              const int row = level_rows_ptr[p];
              csr_triangular_solve_vector_cached_row<I>(
                  data_ptr, indices_ptr, indptr_ptr, diagonal_positions_ptr,
                  b_ptr, x_ptr, row, lower, unit_diagonal);
            }
          });
      return;
    }

    if (lower) {
      for (int row = 0; row < n_rows; ++row) {
        csr_triangular_solve_vector_cached_row<I>(
            data_ptr, indices_ptr, indptr_ptr, diagonal_positions_ptr, b_ptr,
            x_ptr, row, lower, unit_diagonal);
      }
    } else {
      for (int row = n_rows - 1; row >= 0; --row) {
        csr_triangular_solve_vector_cached_row<I>(
            data_ptr, indices_ptr, indptr_ptr, diagonal_positions_ptr, b_ptr,
            x_ptr, row, lower, unit_diagonal);
      }
    }
  });
}

template <typename I>
void csr_triangular_solve_matrix_rhs_analyzed_cpu_impl(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, const mx::array &diagonal_positions,
    const mx::array &level_offsets, const mx::array &level_rows, mx::array &x,
    int n_rows, int rhs_cols, bool lower, bool unit_diagonal,
    mx::Stream stream) {
  x.set_data(mx::allocator::malloc(x.nbytes()));

  auto &encoder = mx::cpu::get_command_encoder(stream);
  encoder.set_input_array(data);
  encoder.set_input_array(indices);
  encoder.set_input_array(indptr);
  encoder.set_input_array(b);
  encoder.set_input_array(diagonal_positions);
  encoder.set_input_array(level_offsets);
  encoder.set_input_array(level_rows);
  encoder.set_output_array(x);

  encoder.dispatch([data = mx::array::unsafe_weak_copy(data),
                    indices = mx::array::unsafe_weak_copy(indices),
                    indptr = mx::array::unsafe_weak_copy(indptr),
                    b = mx::array::unsafe_weak_copy(b),
                    diagonal_positions =
                        mx::array::unsafe_weak_copy(diagonal_positions),
                    level_offsets = mx::array::unsafe_weak_copy(level_offsets),
                    level_rows = mx::array::unsafe_weak_copy(level_rows),
                    x = mx::array::unsafe_weak_copy(x), n_rows, rhs_cols, lower,
                    unit_diagonal]() mutable {
    const auto *data_ptr = data.data<float>();
    const auto *indices_ptr = indices.data<I>();
    const auto *indptr_ptr = indptr.data<I>();
    const auto *b_ptr = b.data<float>();
    const auto *diagonal_positions_ptr = diagonal_positions.data<I>();
    const auto *level_offsets_ptr = level_offsets.data<int32_t>();
    const auto *level_rows_ptr = level_rows.data<int32_t>();
    auto *x_ptr = x.data<float>();

    const int workers = configured_solver_worker_count();
    const int n_levels = static_cast<int>(level_offsets.size()) - 1;
    if (workers > 1 && n_levels > 0 &&
        static_cast<int>(level_rows.size()) == n_rows) {
      std::vector<std::vector<float>> sums(
          static_cast<size_t>(workers),
          std::vector<float>(static_cast<size_t>(rhs_cols)));
      parallel_for_triangular_levels(
          workers, n_levels, [&](int worker, int worker_count, int level) {
            float *sum = sums[static_cast<size_t>(worker)].data();
            const int begin = level_offsets_ptr[level];
            const int end = level_offsets_ptr[level + 1];
            const int width = end - begin;
            const int local_begin =
                begin +
                static_cast<int>((int64_t{worker} * width) / worker_count);
            const int local_end =
                begin +
                static_cast<int>((int64_t{worker + 1} * width) / worker_count);
            for (int p = local_begin; p < local_end; ++p) {
              const int row = level_rows_ptr[p];
              csr_triangular_solve_matrix_cached_row<I>(
                  data_ptr, indices_ptr, indptr_ptr, diagonal_positions_ptr,
                  b_ptr, x_ptr, row, rhs_cols, lower, unit_diagonal, sum);
            }
          });
      return;
    }

    const auto ranges = equal_cpu_ranges(rhs_cols, workers);
    if (ranges.size() > 1) {
      parallel_for_cpu_ranges(ranges, [&](CpuRange cols) {
        csr_triangular_solve_matrix_rhs_range<I>(
            data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, rhs_cols,
            cols, lower, unit_diagonal, diagonal_positions_ptr);
      });
      return;
    }

    switch (rhs_cols) {
    case 2:
      csr_triangular_solve_matrix_fixed_cols<I, 2>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal, diagonal_positions_ptr);
      return;
    case 4:
      csr_triangular_solve_matrix_fixed_cols<I, 4>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal, diagonal_positions_ptr);
      return;
    case 8:
      csr_triangular_solve_matrix_fixed_cols<I, 8>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal, diagonal_positions_ptr);
      return;
    case 16:
      csr_triangular_solve_matrix_fixed_cols<I, 16>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal, diagonal_positions_ptr);
      return;
    case 32:
      csr_triangular_solve_matrix_fixed_cols<I, 32>(
          data_ptr, indices_ptr, indptr_ptr, b_ptr, x_ptr, n_rows, lower,
          unit_diagonal, diagonal_positions_ptr);
      return;
    default:
      break;
    }

    std::vector<float> sum(static_cast<size_t>(rhs_cols));
    if (lower) {
      for (int row = 0; row < n_rows; ++row) {
        csr_triangular_solve_matrix_cached_row<I>(
            data_ptr, indices_ptr, indptr_ptr, diagonal_positions_ptr, b_ptr,
            x_ptr, row, rhs_cols, lower, unit_diagonal, sum.data());
      }
    } else {
      for (int row = n_rows - 1; row >= 0; --row) {
        csr_triangular_solve_matrix_cached_row<I>(
            data_ptr, indices_ptr, indptr_ptr, diagonal_positions_ptr, b_ptr,
            x_ptr, row, rhs_cols, lower, unit_diagonal, sum.data());
      }
    }
  });
}

template <typename I>
mx::array csr_triangular_diagonal_positions_impl(mx::array indices,
                                                 mx::array indptr, int n_rows,
                                                 mx::Dtype index_dtype) {
  indices.eval();
  indptr.eval();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();
  std::vector<I> positions(static_cast<size_t>(n_rows), static_cast<I>(-1));
  for (int row = 0; row < n_rows; ++row) {
    for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
      if (indices_ptr[p] == static_cast<I>(row)) {
        positions[static_cast<size_t>(row)] = p;
      }
    }
    if (positions[static_cast<size_t>(row)] < 0) {
      throw std::runtime_error(
          "csr_triangular_diagonal_positions encountered a missing diagonal.");
    }
  }
  return mx::array(positions.begin(),
                   mx::Shape{static_cast<int>(positions.size())}, index_dtype);
}

template <typename I>
std::tuple<mx::array, mx::array>
csr_triangular_level_schedule_impl(mx::array indices, mx::array indptr,
                                   int n_rows, int n_cols, bool lower) {
  indices.eval();
  indptr.eval();
  const auto *indices_ptr = indices.data<I>();
  const auto *indptr_ptr = indptr.data<I>();

  std::vector<int32_t> levels(static_cast<size_t>(n_rows), 0);
  int32_t max_level = 0;
  if (lower) {
    for (int row = 0; row < n_rows; ++row) {
      int32_t level = 0;
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices_ptr[p]);
        if (col < 0 || col >= n_cols) {
          throw std::invalid_argument(
              "csr_triangular_level_schedule found an out-of-bounds column.");
        }
        if (col < row) {
          level =
              std::max<int32_t>(level, levels[static_cast<size_t>(col)] + 1);
        }
      }
      levels[static_cast<size_t>(row)] = level;
      max_level = std::max(max_level, level);
    }
  } else {
    for (int row = n_rows - 1; row >= 0; --row) {
      int32_t level = 0;
      for (I p = indptr_ptr[row]; p < indptr_ptr[row + 1]; ++p) {
        const int col = static_cast<int>(indices_ptr[p]);
        if (col < 0 || col >= n_cols) {
          throw std::invalid_argument(
              "csr_triangular_level_schedule found an out-of-bounds column.");
        }
        if (col > row) {
          level =
              std::max<int32_t>(level, levels[static_cast<size_t>(col)] + 1);
        }
      }
      levels[static_cast<size_t>(row)] = level;
      max_level = std::max(max_level, level);
    }
  }

  const int n_levels = static_cast<int>(max_level) + 1;
  std::vector<int32_t> counts(static_cast<size_t>(n_levels), 0);
  for (int row = 0; row < n_rows; ++row) {
    ++counts[static_cast<size_t>(levels[static_cast<size_t>(row)])];
  }

  const auto max_width = *std::max_element(counts.begin(), counts.end());
  if (max_width <= 1) {
    std::vector<int32_t> empty;
    return {mx::array(empty.begin(), mx::Shape{0}, mx::int32),
            mx::array(empty.begin(), mx::Shape{0}, mx::int32)};
  }

  std::vector<int32_t> offsets(static_cast<size_t>(n_levels) + 1, 0);
  for (int level = 0; level < n_levels; ++level) {
    offsets[static_cast<size_t>(level) + 1] =
        offsets[static_cast<size_t>(level)] +
        counts[static_cast<size_t>(level)];
  }
  std::vector<int32_t> rows(static_cast<size_t>(n_rows), 0);
  auto cursor = offsets;
  for (int row = 0; row < n_rows; ++row) {
    const int level = levels[static_cast<size_t>(row)];
    rows[static_cast<size_t>(cursor[static_cast<size_t>(level)]++)] = row;
  }

  return {mx::array(offsets.begin(),
                    mx::Shape{static_cast<int>(offsets.size())}, mx::int32),
          mx::array(rows.begin(), mx::Shape{static_cast<int>(rows.size())},
                    mx::int32)};
}

} // namespace

void CSRTriangularSolve::eval_cpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  const bool matrix_rhs = b.ndim() == 2;
  const bool has_analysis = has_analysis_ && inputs.size() == 7;

  if (indices.dtype() == mx::int32) {
    if (has_analysis && matrix_rhs) {
      csr_triangular_solve_matrix_rhs_analyzed_cpu_impl<int32_t>(
          data, indices, indptr, b, inputs[4], inputs[5], inputs[6], outputs[0],
          n_rows_, rhs_cols_, lower_, unit_diagonal_, stream());
    } else if (has_analysis) {
      csr_triangular_solve_vector_analyzed_cpu_impl<int32_t>(
          data, indices, indptr, b, inputs[4], inputs[5], inputs[6], outputs[0],
          n_rows_, lower_, unit_diagonal_, stream());
    } else if (matrix_rhs) {
      csr_triangular_solve_matrix_rhs_cpu_impl<int32_t>(
          data, indices, indptr, b, outputs[0], n_rows_, rhs_cols_, lower_,
          unit_diagonal_, stream());
    } else {
      csr_triangular_solve_vector_cpu_impl<int32_t>(data, indices, indptr, b,
                                                    outputs[0], n_rows_, lower_,
                                                    unit_diagonal_, stream());
    }
    return;
  }
  if (indices.dtype() == mx::int64) {
    if (has_analysis && matrix_rhs) {
      csr_triangular_solve_matrix_rhs_analyzed_cpu_impl<int64_t>(
          data, indices, indptr, b, inputs[4], inputs[5], inputs[6], outputs[0],
          n_rows_, rhs_cols_, lower_, unit_diagonal_, stream());
    } else if (has_analysis) {
      csr_triangular_solve_vector_analyzed_cpu_impl<int64_t>(
          data, indices, indptr, b, inputs[4], inputs[5], inputs[6], outputs[0],
          n_rows_, lower_, unit_diagonal_, stream());
    } else if (matrix_rhs) {
      csr_triangular_solve_matrix_rhs_cpu_impl<int64_t>(
          data, indices, indptr, b, outputs[0], n_rows_, rhs_cols_, lower_,
          unit_diagonal_, stream());
    } else {
      csr_triangular_solve_vector_cpu_impl<int64_t>(data, indices, indptr, b,
                                                    outputs[0], n_rows_, lower_,
                                                    unit_diagonal_, stream());
    }
    return;
  }
  throw std::runtime_error(
      "csr_triangular_solve requires int32 or int64 indices.");
}

#ifdef _METAL_
void CSRTriangularSolve::eval_gpu(const std::vector<mx::array> &inputs,
                                  std::vector<mx::array> &outputs) {
  auto &data = inputs[0];
  auto &indices = inputs[1];
  auto &indptr = inputs[2];
  auto &b = inputs[3];
  auto &x = outputs[0];

  x.set_data(mx::allocator::malloc(x.nbytes()));

  auto &s = stream();
  auto &device = mx::metal::device(s.device);
  auto *lib = device.get_library("mlx_sparse", current_binary_dir());
  auto kernel_name =
      sparse_kernel_name("csr_triangular_solve", data.dtype(), indices.dtype());
  auto *kernel = device.get_kernel(kernel_name, lib);

  auto &encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(data, 0);
  encoder.set_input_array(indices, 1);
  encoder.set_input_array(indptr, 2);
  encoder.set_input_array(b, 3);
  encoder.set_output_array(x, 4);
  encoder.set_bytes(n_rows_, 5);
  encoder.set_bytes(n_cols_, 6);
  int lower = lower_ ? 1 : 0;
  int unit_diagonal = unit_diagonal_ ? 1 : 0;
  encoder.set_bytes(lower, 7);
  encoder.set_bytes(unit_diagonal, 8);
  encoder.set_bytes(rhs_cols_, 9);
  auto threads = std::max<size_t>(static_cast<size_t>(rhs_cols_), 1);
  auto group = std::min(threads, kernel->maxTotalThreadsPerThreadgroup());
  encoder.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(group, 1, 1));
}
#else
void CSRTriangularSolve::eval_gpu(const std::vector<mx::array> &,
                                  std::vector<mx::array> &) {
  throw std::runtime_error(
      "csr_triangular_solve has no GPU implementation in this build.");
}
#endif

mx::array csr_triangular_solve(const mx::array &data, const mx::array &indices,
                               const mx::array &indptr, const mx::array &b,
                               int n_rows, int n_cols, bool lower,
                               bool unit_diagonal, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_triangular_solve requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_triangular_solve data");
  require_rank(indices, 1, "csr_triangular_solve indices");
  require_rank(indptr, 1, "csr_triangular_solve indptr");
  if (b.ndim() != 1 && b.ndim() != 2) {
    throw std::invalid_argument(
        "csr_triangular_solve b must be rank-1 or rank-2.");
  }
  require_linalg_float32(data, "csr_triangular_solve data");
  require_linalg_float32(b, "csr_triangular_solve b");
  require_same_index_dtype(indices, indptr, "csr_triangular_solve indices",
                           "csr_triangular_solve indptr");
  require_size(indptr, n_rows + 1, "csr_triangular_solve indptr");
  const bool matrix_rhs = b.ndim() == 2;
  const int rhs_cols = matrix_rhs ? b.shape(1) : 1;
  if (matrix_rhs) {
    if (b.shape(0) != n_rows) {
      throw std::invalid_argument(
          "csr_triangular_solve rank-2 b has incompatible row dimension.");
    }
    if (rhs_cols <= 0) {
      throw std::invalid_argument(
          "csr_triangular_solve rank-2 b must include at least one column.");
    }
  } else {
    require_size(b, n_rows, "csr_triangular_solve b");
  }
  if (indices.size() != data.size()) {
    throw std::invalid_argument(
        "csr_triangular_solve data and indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);

  const auto out_shape =
      matrix_rhs ? mx::Shape{n_rows, rhs_cols} : mx::Shape{n_rows};
  return mx::array(out_shape, mx::float32,
                   std::make_shared<CSRTriangularSolve>(
                       stream, n_rows, n_cols, lower, unit_diagonal, rhs_cols),
                   {data_contig, indices_contig, indptr_contig, b_contig});
}

mx::array csr_triangular_diagonal_positions(const mx::array &indices,
                                            const mx::array &indptr, int n_rows,
                                            int n_cols) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument("csr_triangular_diagonal_positions requires a "
                                "non-empty square matrix.");
  }
  require_rank(indices, 1, "csr_triangular_diagonal_positions indices");
  require_rank(indptr, 1, "csr_triangular_diagonal_positions indptr");
  require_same_index_dtype(indices, indptr,
                           "csr_triangular_diagonal_positions indices",
                           "csr_triangular_diagonal_positions indptr");
  require_size(indptr, n_rows + 1, "csr_triangular_diagonal_positions indptr");
  if (indices.dtype() == mx::int32) {
    return csr_triangular_diagonal_positions_impl<int32_t>(indices, indptr,
                                                           n_rows, mx::int32);
  }
  if (indices.dtype() == mx::int64) {
    return csr_triangular_diagonal_positions_impl<int64_t>(indices, indptr,
                                                           n_rows, mx::int64);
  }
  throw std::runtime_error(
      "csr_triangular_diagonal_positions requires int32 or int64 indices.");
}

std::tuple<mx::array, mx::array>
csr_triangular_level_schedule(const mx::array &indices, const mx::array &indptr,
                              int n_rows, int n_cols, bool lower) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_triangular_level_schedule requires a non-empty square matrix.");
  }
  require_rank(indices, 1, "csr_triangular_level_schedule indices");
  require_rank(indptr, 1, "csr_triangular_level_schedule indptr");
  require_same_index_dtype(indices, indptr,
                           "csr_triangular_level_schedule indices",
                           "csr_triangular_level_schedule indptr");
  require_size(indptr, n_rows + 1, "csr_triangular_level_schedule indptr");
  if (indices.dtype() == mx::int32) {
    return csr_triangular_level_schedule_impl<int32_t>(indices, indptr, n_rows,
                                                       n_cols, lower);
  }
  if (indices.dtype() == mx::int64) {
    return csr_triangular_level_schedule_impl<int64_t>(indices, indptr, n_rows,
                                                       n_cols, lower);
  }
  throw std::runtime_error(
      "csr_triangular_level_schedule requires int32 or int64 indices.");
}

mx::array csr_triangular_solve_analyzed(
    const mx::array &data, const mx::array &indices, const mx::array &indptr,
    const mx::array &b, const mx::array &diagonal_positions,
    const mx::array &level_offsets, const mx::array &level_rows, int n_rows,
    int n_cols, bool lower, bool unit_diagonal, mx::StreamOrDevice s) {
  if (n_rows <= 0 || n_cols <= 0 || n_rows != n_cols) {
    throw std::invalid_argument(
        "csr_triangular_solve_analyzed requires a non-empty square matrix.");
  }
  require_rank(data, 1, "csr_triangular_solve_analyzed data");
  require_rank(indices, 1, "csr_triangular_solve_analyzed indices");
  require_rank(indptr, 1, "csr_triangular_solve_analyzed indptr");
  require_rank(diagonal_positions, 1,
               "csr_triangular_solve_analyzed diagonal_positions");
  require_rank(level_offsets, 1, "csr_triangular_solve_analyzed level_offsets");
  require_rank(level_rows, 1, "csr_triangular_solve_analyzed level_rows");
  if (b.ndim() != 1 && b.ndim() != 2) {
    throw std::invalid_argument(
        "csr_triangular_solve_analyzed b must be rank-1 or rank-2.");
  }
  require_linalg_float32(data, "csr_triangular_solve_analyzed data");
  require_linalg_float32(b, "csr_triangular_solve_analyzed b");
  require_same_index_dtype(indices, indptr,
                           "csr_triangular_solve_analyzed indices",
                           "csr_triangular_solve_analyzed indptr");
  if (diagonal_positions.dtype() != indices.dtype()) {
    throw std::invalid_argument("csr_triangular_solve_analyzed "
                                "diagonal_positions must use the index dtype.");
  }
  if (level_offsets.dtype() != mx::int32 || level_rows.dtype() != mx::int32) {
    throw std::invalid_argument(
        "csr_triangular_solve_analyzed level schedule arrays must be int32.");
  }
  require_size(indptr, n_rows + 1, "csr_triangular_solve_analyzed indptr");
  require_size(diagonal_positions, n_rows,
               "csr_triangular_solve_analyzed diagonal_positions");
  const bool has_level_schedule = level_offsets.size() > 0;
  if (has_level_schedule) {
    if (level_offsets.size() < 2 || level_rows.size() != n_rows) {
      throw std::invalid_argument(
          "csr_triangular_solve_analyzed received an invalid level schedule.");
    }
  } else if (level_rows.size() != 0) {
    throw std::invalid_argument("csr_triangular_solve_analyzed level_rows must "
                                "be empty when level_offsets is empty.");
  }

  const bool matrix_rhs = b.ndim() == 2;
  const int rhs_cols = matrix_rhs ? b.shape(1) : 1;
  if (matrix_rhs) {
    if (b.shape(0) != n_rows) {
      throw std::invalid_argument("csr_triangular_solve_analyzed rank-2 b has "
                                  "incompatible row dimension.");
    }
    if (rhs_cols <= 0) {
      throw std::invalid_argument("csr_triangular_solve_analyzed rank-2 b must "
                                  "include at least one column.");
    }
  } else {
    require_size(b, n_rows, "csr_triangular_solve_analyzed b");
  }
  if (indices.size() != data.size()) {
    throw std::invalid_argument("csr_triangular_solve_analyzed data and "
                                "indices must have equal length.");
  }

  auto stream = mx::to_stream(s);
  auto data_contig = mx::contiguous(data, false, stream);
  auto indices_contig = mx::contiguous(indices, false, stream);
  auto indptr_contig = mx::contiguous(indptr, false, stream);
  auto b_contig = mx::contiguous(b, false, stream);
  auto diagonal_positions_contig =
      mx::contiguous(diagonal_positions, false, stream);
  auto level_offsets_contig = mx::contiguous(level_offsets, false, stream);
  auto level_rows_contig = mx::contiguous(level_rows, false, stream);

  const auto out_shape =
      matrix_rhs ? mx::Shape{n_rows, rhs_cols} : mx::Shape{n_rows};
  return mx::array(
      out_shape, mx::float32,
      std::make_shared<CSRTriangularSolve>(stream, n_rows, n_cols, lower,
                                           unit_diagonal, rhs_cols, true),
      {data_contig, indices_contig, indptr_contig, b_contig,
       diagonal_positions_contig, level_offsets_contig, level_rows_contig});
}

} // namespace mlx_sparse
