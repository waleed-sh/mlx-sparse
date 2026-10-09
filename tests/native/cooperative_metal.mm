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

// Exercise the compiled cooperative kernels directly, including wide indexing.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      return 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    NSURL *url =
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (!library) {
      fprintf(stderr, "%s\n", error.description.UTF8String);
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::vector<float> data(96, 1.0f / 32);
    std::vector<int32_t> indices(96);
    for (int i = 0; i < 96; ++i) {
      indices[i] = i % 32;
    }
    const int32_t pointers[] = {0, 32, 64, 96};
    auto buffer = [&](const void *values, size_t bytes) {
      return [device newBufferWithBytes:values
                                 length:bytes
                                options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> data_buffer =
        buffer(data.data(), data.size() * sizeof(float));
    id<MTLBuffer> index_buffer =
        buffer(indices.data(), indices.size() * sizeof(int32_t));
    id<MTLBuffer> pointer_buffer = buffer(pointers, sizeof(pointers));
    const std::vector<std::string> operations = {
        "csr_matvec",         "csr_matmul",           "csr_batched_matvec",
        "csr_batched_matmul", "csr_diagonal",         "csr_row_sums",
        "csr_row_norms",      "csc_matvec_transpose", "csc_diagonal",
        "csc_col_sums",       "csc_col_norms"};
    for (const auto &op : operations) {
      const bool dense = op == "csr_matmul" || op == "csr_batched_matmul" ||
                         op == "csr_batched_matvec";
      const bool batched =
          op == "csr_batched_matmul" || op == "csr_batched_matvec";
      const bool matrix = op == "csr_matmul" || op == "csr_batched_matmul";
      const bool diagonal = op.find("diagonal") != std::string::npos;
      const bool reduction = op.find("sums") != std::string::npos ||
                             op.find("norms") != std::string::npos;
      const int width = matrix ? 4 : 1;
      const int batches = batched ? 2 : 1;
      const int count = 3 * width * batches;
      std::vector<float> rhs(32 * width * batches);
      for (int batch = 0; batch < batches; ++batch) {
        for (int row = 0; row < 32; ++row) {
          for (int col = 0; col < width; ++col) {
            rhs[(batch * 32 + row) * width + col] = (batch + 1) * (col + 1);
          }
        }
      }
      id<MTLBuffer> rhs_buffer = buffer(rhs.data(), rhs.size() * sizeof(float));
      for (int wide = 0; wide <= (dense ? 1 : 0); ++wide) {
        std::string name =
            op + "_vector_float32_int32" + (wide ? "_large" : "");
        id<MTLFunction> function = [library
            newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
        if (!function) {
          fprintf(stderr, "Missing Metal function: %s\n", name.c_str());
          return 1;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
          fprintf(stderr, "%s: %s\n", name.c_str(),
                  error.description.UTF8String);
          return 1;
        }
        std::vector<float> sentinel(count, NAN);
        id<MTLBuffer> output =
            buffer(sentinel.data(), sentinel.size() * sizeof(float));
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        auto bind = [&](id<MTLBuffer> value, int index) {
          [encoder setBuffer:value offset:0 atIndex:index];
        };
        auto scalar = [&](int value, int index) {
          [encoder setBytes:&value length:sizeof(value) atIndex:index];
        };
        bind(data_buffer, 0);
        if (reduction) {
          bind(pointer_buffer, 1);
          bind(output, 2);
          scalar(3, 3);
        } else {
          bind(index_buffer, 1);
          bind(pointer_buffer, 2);
          if (diagonal) {
            bind(output, 3);
            scalar(3, 4);
          } else {
            bind(rhs_buffer, 3);
            bind(output, 4);
            scalar(3, 5);
            if (batched) {
              scalar(32, 6);
              scalar(batches, 7);
              if (matrix) {
                scalar(width, 8);
              }
            } else if (matrix) {
              scalar(width, 6);
            }
          }
        }
        // Force two dimensions and an unused group for odd output counts.
        [encoder dispatchThreads:MTLSizeMake(2 * 128, (count + 1) / 2, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) {
          fprintf(stderr, "%s: %s\n", name.c_str(),
                  command.error.description.UTF8String);
          return 1;
        }
        const auto *values = static_cast<const float *>(output.contents);
        for (int i = 0; i < count; ++i) {
          const float expected =
              diagonal                                ? 1.0f / 32
              : op.find("norms") != std::string::npos ? std::sqrt(1.0f / 32)
              : reduction                             ? 1.0f
                          : float((i / (3 * width) + 1) * (i % width + 1));
          if (!std::isfinite(values[i]) ||
              std::abs(values[i] - expected) > 1e-6f) {
            fprintf(stderr, "%s output %d: %g != %g\n", name.c_str(), i,
                    values[i], expected);
            return 1;
          }
        }
      }
    }
  }
  return 0;
}
