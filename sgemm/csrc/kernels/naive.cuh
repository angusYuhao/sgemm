#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void sgemm_naive_kernel(uint32_t M, uint32_t N, uint32_t K, float alpha, const float *A,
                                   const float *B, float beta, float *C) {
  // the row and col that this thread is calculating for in C
  // NOTE: consecutive threads are calculating for different rows in the same col, global memory
  // access is not coalesced
  const uint32_t global_row = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_col = blockIdx.y * blockDim.y + threadIdx.y;

  if (global_row < M && global_col < N) {
    float accum = 0;
    for (int k = 0; k < K; k++) {
      accum += A[global_row * K + k] * B[k * N + global_col];
    }
    C[global_row * N + global_col] = alpha * accum + beta * C[global_row * N + global_col];
  }
}
