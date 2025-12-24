#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <const uint32_t BLOCK_SIZE>
__global__ void
sgemm_shared_memory_blocking_kernel(uint32_t M, uint32_t N, uint32_t K,
                                    float alpha, const float *A, const float *B,
                                    float beta, float *C) {
  // declare shared memory storage for A and B blocks
  __shared__ float A_blk[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_blk[BLOCK_SIZE][BLOCK_SIZE];

  // define block and thread indices
  const uint32_t bx = blockIdx.x, by = blockIdx.y;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;

  // get position of C matrix that this thread is computing for
  const uint32_t row = by * BLOCK_SIZE + ty;
  const uint32_t col = bx * BLOCK_SIZE + tx;

  // outer loop: loop through the blocks in the K direction
  float tmp = 0.0;
  for (int blk_id = 0; blk_id < CEIL_DIV(K, BLOCK_SIZE); blk_id++) {
    // cooperatively load A and B elements from global memory
    if (row < M && (blk_id * BLOCK_SIZE + tx) < K) {
      A_blk[ty][tx] = A[row * K + blk_id * BLOCK_SIZE + tx];
    }
    else {
      A_blk[ty][tx] = 0.0f;
    }
    if ((blk_id * BLOCK_SIZE + ty) < K && col < N) {
      B_blk[ty][tx] = B[(blk_id * BLOCK_SIZE + ty) * N + col];
    }
    else {
      B_blk[ty][tx] = 0.0f;
    }
    __syncthreads();

    // inner loop
    for (int k = 0; k < BLOCK_SIZE; k++) {
      tmp += A_blk[ty][k] * B_blk[k][tx];
    }
    __syncthreads();
  }

  // populate the output tensor
  if (row < M && col < N){
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
}
