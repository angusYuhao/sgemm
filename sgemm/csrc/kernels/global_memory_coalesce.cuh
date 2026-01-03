#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <const uint32_t BLOCK_SIZE>
__global__ void sgemm_global_memory_coalesce_v1_kernel(uint32_t M, uint32_t N, uint32_t K,
                                                       float alpha, const float *A, const float *B,
                                                       float beta, float *C) {
  // the row and col that this thread is calculating for in C
  const uint32_t global_row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  const uint32_t global_col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

  if (global_row < M && global_col < N) {
    float accum = 0;
    for (int k = 0; k < K; k++) {
      accum += A[global_row * K + k] * B[k * N + global_col];
    }
    C[global_row * N + global_col] = alpha * accum + beta * C[global_row * N + global_col];
  }
}

__global__ void sgemm_global_memory_coalesce_v2_kernel(uint32_t M, uint32_t N, uint32_t K,
                                                       float alpha, const float *A, const float *B,
                                                       float beta, float *C) {
  // the row and col that this thread is calculating for in C
  const uint32_t global_row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t global_col = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_row < M && global_col < N) {
    float accum = 0;
    for (int k = 0; k < K; k++) {
      accum += A[global_row * K + k] * B[k * N + global_col];
    }
    C[global_row * N + global_col] = alpha * accum + beta * C[global_row * N + global_col];
  }
}
