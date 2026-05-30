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

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <thread>
#include <type_traits>
#include <vector>

namespace mlx_sparse {

struct CpuRange {
  int begin;
  int end;
};

bool spgemm_parallel_enabled();
int configured_cpu_worker_count();
int configured_spgemm_worker_count();
std::vector<CpuRange> equal_cpu_ranges(int n_items, int partitions);
std::vector<CpuRange> weighted_cpu_ranges(const std::vector<int64_t> &row_work,
                                          int partitions);

template <typename Fn>
void parallel_for_cpu_ranges_indexed(const std::vector<CpuRange> &ranges,
                                     Fn &&fn) {
  if (ranges.empty()) {
    return;
  }
  if (ranges.size() == 1) {
    fn(size_t{0}, ranges[0]);
    return;
  }

  std::atomic<bool> failed{false};
  std::exception_ptr first_exception = nullptr;
  std::vector<std::thread> workers;
  workers.reserve(ranges.size() - 1);

  auto run_range = [&](size_t index, CpuRange range) {
    if (failed.load(std::memory_order_relaxed)) {
      return;
    }
    try {
      fn(index, range);
    } catch (...) {
      bool expected = false;
      if (failed.compare_exchange_strong(expected, true,
                                         std::memory_order_relaxed)) {
        first_exception = std::current_exception();
      }
    }
  };

  for (size_t i = 1; i < ranges.size(); ++i) {
    workers.emplace_back(run_range, i, ranges[i]);
  }
  run_range(size_t{0}, ranges[0]);
  for (auto &worker : workers) {
    worker.join();
  }
  if (first_exception != nullptr) {
    std::rethrow_exception(first_exception);
  }
}

template <typename Fn>
void parallel_for_cpu_ranges(const std::vector<CpuRange> &ranges, Fn &&fn) {
  parallel_for_cpu_ranges_indexed(ranges,
                                  [&](size_t, CpuRange range) { fn(range); });
}

template <typename CountT>
std::vector<CpuRange>
cpu_ranges_for_output_work(const std::vector<CountT> &work,
                           int requested_workers) {
  static_assert(std::is_integral_v<CountT>,
                "work estimates must use an integral type.");
  if (work.empty()) {
    return {};
  }
  if (requested_workers <= 1) {
    return {{0, static_cast<int>(work.size())}};
  }

  std::vector<int64_t> normalized;
  normalized.reserve(work.size());
  bool has_imbalance = false;
  int64_t previous = -1;
  for (const CountT value : work) {
    const int64_t clipped = std::max<int64_t>(0, static_cast<int64_t>(value));
    normalized.push_back(clipped);
    if (previous >= 0 && clipped != previous) {
      has_imbalance = true;
    }
    previous = clipped;
  }

  if (!has_imbalance) {
    return equal_cpu_ranges(static_cast<int>(work.size()), requested_workers);
  }
  return weighted_cpu_ranges(normalized, requested_workers);
}

template <typename IndexT>
std::vector<int64_t> compressed_segment_work(const IndexT *indptr,
                                             int n_segments) {
  static_assert(std::is_integral_v<IndexT>,
                "compressed indptr values must use an integral type.");
  std::vector<int64_t> work(static_cast<size_t>(std::max(n_segments, 0)));
  for (int segment = 0; segment < n_segments; ++segment) {
    const auto begin = static_cast<int64_t>(indptr[segment]);
    const auto end = static_cast<int64_t>(indptr[segment + 1]);
    work[static_cast<size_t>(segment)] = std::max<int64_t>(0, end - begin);
  }
  return work;
}

template <typename IndexT>
std::vector<CpuRange> cpu_ranges_for_compressed_segments(const IndexT *indptr,
                                                         int n_segments,
                                                         int workers) {
  if (n_segments <= 0) {
    return {};
  }
  if (workers <= 1) {
    return {{0, n_segments}};
  }
  return cpu_ranges_for_output_work(compressed_segment_work(indptr, n_segments),
                                    workers);
}

} // namespace mlx_sparse
