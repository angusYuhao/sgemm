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
  const uint32_t row = bx * BLOCK_SIZE + tx;
  const uint32_t col = by * BLOCK_SIZE + ty;

  // outer loop: loop through the blocks in the K direction
  float tmp = 0.0;
  for (int blk_id = 0; blk_id < CEIL_DIV(K, BLOCK_SIZE); blk_id++) {
    // cooperatively load A and B elements from global memory
    A_blk[tx][ty] = A[row * K + blk_id * BLOCK_SIZE + ty];
    B_blk[tx][ty] = B[(blk_id * BLOCK_SIZE + tx) * N + col];
    __syncthreads();

    // inner loop
    for (int k = 0; k < BLOCK_SIZE; k++) {
      tmp += A_blk[tx][k] * B_blk[k][ty];
    }
    __syncthreads();
  }

  // populate the output tensor
  C[row * N + col] = alpha * tmp + beta * C[row * N + col];
}
