#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <const uint32_t BLOCK_SIZE>
__global__ void sgemm_shared_memory_blocking_kernel_v1(uint32_t M, uint32_t N, uint32_t K,
                                                       float alpha, const float *A, const float *B,
                                                       float beta, float *C) {
  // block row and col of the thread
  const uint32_t block_row = blockIdx.x;
  const uint32_t block_col = blockIdx.y;

  // thread row and col of the thread in the block
  const uint32_t thread_row = threadIdx.x / BLOCK_SIZE;
  const uint32_t thread_col = threadIdx.x % BLOCK_SIZE;

  // global row and col of the thread in the grid
  const uint32_t global_row = block_row * BLOCK_SIZE + thread_row;
  const uint32_t global_col = block_col * BLOCK_SIZE + thread_col;

  // allocate shared memory containers for storing A and B tiles
  __shared__ float tile_a[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float tile_b[BLOCK_SIZE * BLOCK_SIZE];

  // move the matrix pointers to the starting block of the algorithm
  A += block_row * BLOCK_SIZE * K;
  B += block_col * BLOCK_SIZE;
  C += block_row * BLOCK_SIZE * N + block_col * BLOCK_SIZE;

  float accumulator = 0.0f;

  // outer loop: loop over the tiles in the K direction
  for (int tile_idx = 0; tile_idx < CEIL_DIV(K, BLOCK_SIZE); tile_idx++) {
    // load the A tile from global memory
    if (global_row < M && tile_idx * BLOCK_SIZE + thread_col < K) {
      tile_a[thread_row * BLOCK_SIZE + thread_col] = A[thread_row * K + thread_col];
    } else {
      tile_a[thread_row * BLOCK_SIZE + thread_col] = 0.0f;
    }
    // load the B tile from global memory
    if (tile_idx * BLOCK_SIZE + thread_row < K && global_col < N) {
      tile_b[thread_row * BLOCK_SIZE + thread_col] = B[thread_row * N + thread_col];
    } else {
      tile_b[thread_row * BLOCK_SIZE + thread_col] = 0.0f;
    }
    // sync the threads to ensure that A and B tiles are completely loaded into shared memory
    __syncthreads();

    // move the matrix pointers to the next block in the outer loop
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;

    // inner loop: compute the partial dot product
    for (int dot_idx = 0; dot_idx < BLOCK_SIZE; dot_idx++) {
      accumulator +=
          tile_a[thread_row * BLOCK_SIZE + dot_idx] * tile_b[dot_idx * BLOCK_SIZE + thread_col];
      if (thread_row == 0 && thread_col == 0)
        printf("tile_a val: %f, tile_b val: %f\n", tile_a[thread_row * BLOCK_SIZE + dot_idx],
               tile_b[dot_idx * BLOCK_SIZE + thread_col]);
    }

    // sync the threads to ensure faster threads don't continue to go to next outer loop iteration
    __syncthreads();
  }

  // write the result of this thread to C
  if (global_row < M && global_col < N) {
    C[thread_row * N + thread_col] = alpha * accumulator + beta * C[thread_row * N + thread_col];
  }
}

template <const uint32_t BLOCK_SIZE>
__global__ void sgemm_shared_memory_blocking_kernel_v2(uint32_t M, uint32_t N, uint32_t K,
                                                       float alpha, const float *A, const float *B,
                                                       float beta, float *C) {
  // declare shared memory storage for A and B blocks
  // this kernel assumes that a block is 2D and has BLOCKSIZE as both of its sides
  __shared__ float A_blk[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_blk[BLOCK_SIZE][BLOCK_SIZE];

  // define block and thread indices
  const uint32_t bx = blockIdx.x, by = blockIdx.y;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;

  // get position of C matrix that this thread is computing for
  const uint32_t global_row = by * BLOCK_SIZE + ty;
  const uint32_t global_col = bx * BLOCK_SIZE + tx;

  // outer loop: loop through the blocks in the K direction
  float tmp = 0.0;
  for (int blk_id = 0; blk_id < CEIL_DIV(K, BLOCK_SIZE); blk_id++) {
    // cooperatively load A and B elements from global memory
    if (global_row < M && (blk_id * BLOCK_SIZE + tx) < K) {
      A_blk[ty][tx] = A[global_row * K + blk_id * BLOCK_SIZE + tx];
    } else {
      A_blk[ty][tx] = 0.0f;
    }
    if ((blk_id * BLOCK_SIZE + ty) < K && global_col < N) {
      B_blk[ty][tx] = B[(blk_id * BLOCK_SIZE + ty) * N + global_col];
    } else {
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
  if (global_row < M && global_col < N) {
    C[global_row * N + global_col] = alpha * tmp + beta * C[global_row * N + global_col];
  }
}
