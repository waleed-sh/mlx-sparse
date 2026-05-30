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

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <string>

#if defined(__linux__)
#include <sched.h>
#endif

namespace mlx_sparse {

namespace {

constexpr const char *kCpuThreadsEnv = "MLX_SPARSE_CPU_THREADS";
constexpr const char *kCpuThreadsAliasEnv = "MLX_SPARSE_N_THREADS";
constexpr const char *kSpgemmParallelEnv = "MLX_SPARSE_SPGEMM_PARALLEL";
constexpr const char *kSpgemmThreadsEnv = "MLX_SPARSE_SPGEMM_THREADS";
constexpr const char *kSolverParallelEnv = "MLX_SPARSE_SOLVER_PARALLEL";
constexpr const char *kSolverThreadsEnv = "MLX_SPARSE_SOLVER_THREADS";

std::string lower_ascii(const char *value) {
  std::string out(value == nullptr ? "" : value);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

int parse_positive_int(const char *value) {
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    return 0;
  }
  if (*end != '\0' && *end != ',') {
    return 0;
  }
  return static_cast<int>(parsed);
}

bool parse_bool_env(const char *name, bool default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr) {
    return default_value;
  }
  const auto value = lower_ascii(raw);
  if (value == "1" || value == "true" || value == "t" || value == "yes" ||
      value == "y" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "f" || value == "no" ||
      value == "n" || value == "off") {
    return false;
  }
  return default_value;
}

int affinity_worker_count() {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    const int count = CPU_COUNT(&set);
    if (count > 0) {
      return count;
    }
  }
#endif
  return 0;
}

int hardware_worker_count() {
  const auto count = static_cast<int>(std::thread::hardware_concurrency());
  return count > 0 ? count : 1;
}

int auto_worker_count() {
  if (const int count = parse_positive_int(std::getenv("OMP_NUM_THREADS"));
      count > 0) {
    return count;
  }
  for (const char *name :
       {"SLURM_CPUS_PER_TASK", "PBS_NP", "LSB_DJOB_NUMPROC", "NSLOTS"}) {
    if (const int count = parse_positive_int(std::getenv(name)); count > 0) {
      return count;
    }
  }
  if (const int count = affinity_worker_count(); count > 0) {
    return count;
  }
  return hardware_worker_count();
}

int resolved_cpu_worker_count() {
  const char *raw = std::getenv(kCpuThreadsEnv);
  if (raw == nullptr || *raw == '\0') {
    raw = std::getenv(kCpuThreadsAliasEnv);
  }
  if (const int count = parse_positive_int(raw); count > 0) {
    return count;
  }
  return auto_worker_count();
}

} // namespace

bool spgemm_parallel_enabled() {
  return parse_bool_env(kSpgemmParallelEnv, true);
}

bool solver_parallel_enabled() {
  return parse_bool_env(kSolverParallelEnv, false);
}

int configured_cpu_worker_count() { return resolved_cpu_worker_count(); }

int configured_spgemm_worker_count() {
  if (!spgemm_parallel_enabled()) {
    return 1;
  }

  const char *raw = std::getenv(kSpgemmThreadsEnv);
  if (raw == nullptr || *raw == '\0') {
    return configured_cpu_worker_count();
  }

  const auto value = lower_ascii(raw);
  if (value == "inherit") {
    return resolved_cpu_worker_count();
  }
  if (value == "auto") {
    return auto_worker_count();
  }
  if (const int count = parse_positive_int(raw); count > 0) {
    return count;
  }
  return resolved_cpu_worker_count();
}

int configured_solver_worker_count() {
  if (!solver_parallel_enabled()) {
    return 1;
  }

  const char *raw = std::getenv(kSolverThreadsEnv);
  if (raw == nullptr || *raw == '\0') {
    return configured_cpu_worker_count();
  }

  const auto value = lower_ascii(raw);
  if (value == "inherit") {
    return resolved_cpu_worker_count();
  }
  if (value == "auto") {
    return auto_worker_count();
  }
  if (const int count = parse_positive_int(raw); count > 0) {
    return count;
  }
  return resolved_cpu_worker_count();
}

std::vector<CpuRange> equal_cpu_ranges(int n_items, int partitions) {
  if (n_items <= 0) {
    return {};
  }
  const int count = std::max(1, partitions);
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
  const int count = std::max(1, partitions);
  if (count == 1) {
    return {{0, n_rows}};
  }
  if (count >= n_rows) {
    return equal_cpu_ranges(n_rows, count);
  }

  std::vector<int64_t> prefix(static_cast<size_t>(n_rows) + 1, 0);
  for (int row = 0; row < n_rows; ++row) {
    const auto work = std::max<int64_t>(0, row_work[static_cast<size_t>(row)]);
    const auto next = prefix[static_cast<size_t>(row)] + work;
    prefix[static_cast<size_t>(row) + 1] =
        next < prefix[static_cast<size_t>(row)]
            ? std::numeric_limits<int64_t>::max()
            : next;
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
