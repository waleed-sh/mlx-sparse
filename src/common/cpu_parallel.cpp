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

#include "common/cpu_parallel.h"

#include <cstdlib>
#include <limits>
#include <string>

namespace mlx_sparse {

namespace {

int parse_positive_int(const char *value) {
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return 0;
  }
  return static_cast<int>(parsed);
}

} // namespace

int default_cpu_worker_limit() {
  const auto hardware = static_cast<int>(std::thread::hardware_concurrency());
  if (hardware <= 0) {
    return 1;
  }
  return std::max(1, std::min(hardware, kDefaultMaxCpuWorkers));
}

int configured_cpu_worker_count() {
  const int requested = parse_positive_int(std::getenv(kCpuThreadsEnv));
  if (requested > 0) {
    return requested;
  }
  return default_cpu_worker_limit();
}

int cpu_worker_count_for_work(int64_t work_units, int max_partitions) {
  if (max_partitions <= 1 || work_units <= kCpuParallelWorkThreshold) {
    return 1;
  }
  const int configured = configured_cpu_worker_count();
  const auto workers_by_work = static_cast<int>(
      (work_units + kCpuParallelWorkPerWorker - 1) / kCpuParallelWorkPerWorker);
  return std::max(1, std::min({configured, max_partitions, workers_by_work}));
}

std::vector<CpuRange> equal_cpu_ranges(int n_items, int partitions) {
  if (n_items <= 0) {
    return {};
  }
  const int count = std::max(1, std::min(partitions, n_items));
  std::vector<CpuRange> ranges;
  ranges.reserve(static_cast<size_t>(count));
  for (int part = 0; part < count; ++part) {
    const int begin =
        static_cast<int>((static_cast<int64_t>(part) * n_items) / count);
    const int end =
        static_cast<int>((static_cast<int64_t>(part + 1) * n_items) / count);
    if (begin < end) {
      ranges.push_back({begin, end});
    }
  }
  return ranges;
}

std::vector<CpuRange> weighted_cpu_ranges(const std::vector<int64_t> &row_work,
                                          int partitions) {
  const int n_rows = static_cast<int>(row_work.size());
  if (n_rows <= 0) {
    return {};
  }
  const int count = std::max(1, std::min(partitions, n_rows));
  if (count == 1) {
    return {{0, n_rows}};
  }

  std::vector<int64_t> prefix(static_cast<size_t>(n_rows) + 1, 0);
  for (int row = 0; row < n_rows; ++row) {
    const auto work = std::max<int64_t>(0, row_work[static_cast<size_t>(row)]);
    prefix[static_cast<size_t>(row) + 1] =
        prefix[static_cast<size_t>(row)] + work;
  }
  if (prefix.back() <= 0) {
    return equal_cpu_ranges(n_rows, count);
  }

  std::vector<CpuRange> ranges;
  ranges.reserve(static_cast<size_t>(count));
  int begin = 0;
  for (int part = 0; part < count && begin < n_rows; ++part) {
    const int remaining_parts = count - part;
    if (remaining_parts == 1) {
      ranges.push_back({begin, n_rows});
      break;
    }

    const auto remaining_work =
        prefix.back() - prefix[static_cast<size_t>(begin)];
    const auto target =
        prefix[static_cast<size_t>(begin)] +
        (remaining_work + remaining_parts - 1) / remaining_parts;
    auto lower =
        std::lower_bound(prefix.begin() + begin + 1, prefix.end(), target);
    int end = static_cast<int>(lower - prefix.begin());
    end = std::max(end, begin + 1);
    end = std::min(end, n_rows - (remaining_parts - 1));
    ranges.push_back({begin, end});
    begin = end;
  }
  return ranges;
}

} // namespace mlx_sparse
