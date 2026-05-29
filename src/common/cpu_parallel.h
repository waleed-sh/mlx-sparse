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
#include <vector>

namespace mlx_sparse {

constexpr const char *kCpuThreadsEnv = "MLX_SPARSE_CPU_THREADS";
constexpr int kDefaultMaxCpuWorkers = 8;
constexpr int64_t kCpuParallelWorkThreshold = 262144;
constexpr int64_t kCpuParallelWorkPerWorker = 262144;

struct CpuRange {
  int begin;
  int end;
};

int default_cpu_worker_limit();
int configured_cpu_worker_count();
int cpu_worker_count_for_work(int64_t work_units, int max_partitions);
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

} // namespace mlx_sparse
