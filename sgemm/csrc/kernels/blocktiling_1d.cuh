#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <const uint32_t BM, const uint32_t BN, const uint32_t BK, const uint32_t TM>
__global__ void sgemm_1d_blocktiling_v1_kernel(uint32_t M, uint32_t N, uint32_t K, float alpha,
                                               const float *A, const float *B, float beta,
                                               float *C) {
  // ensure that the sizes chosen for BM, BN, BK is supported by the launch config
  assert(BM * BK == blockDim.x);
  assert(BK * BN == blockDim.x);

  // block level indices of the current thread
  const uint32_t block_row = blockIdx.x;
  const uint32_t block_col = blockIdx.y;

  // thread level indices within the block
  const uint32_t thread_row = threadIdx.x / BN;
  const uint32_t thread_col = threadIdx.x % BN;

  // indices that the thread is responsible for in C
  const uint32_t global_row = block_row * BM + thread_row * TM;
  const uint32_t global_col = block_col * BN + thread_col;

  // allocate shared memory containers for storing A and B tiles
  __shared__ float tile_a[BM * BK];
  __shared__ float tile_b[BK * BN];

  // move the matrix pointers to the starting block
  A += block_row * BM * K;
  B += block_col * BN;
  C += block_row * BM * N + block_col * BN;

  // index of the element to load in tile_a
  const uint32_t tile_a_row = threadIdx.x / BK;
  const uint32_t tile_a_col = threadIdx.x % BK;
  const uint32_t tile_b_row = threadIdx.x / BN;
  const uint32_t tile_b_col = threadIdx.x % BN;

  // allocate thread-local cache for results
  float thread_results[TM] = {0.0f};

  // outer loop
  for (int tile_idx = 0; tile_idx < CEIL_DIV(K, BK); tile_idx++) {
    // load tile_a from global memory
    if (global_row < M && tile_idx * BK + tile_a_col < K) {
      tile_a[tile_a_row * BK + tile_a_col] = A[tile_a_row * K + tile_a_col];
    } else {
      tile_a[tile_a_row * BK + tile_a_col] = 0.0f;
    }
    // load tile_b from global memory
    if (tile_idx * BK + tile_b_row < K && global_col < N) {
      tile_b[tile_b_row * BN + tile_b_col] = B[tile_b_row * N + tile_b_col];
    } else {
      tile_b[tile_b_row * BN + tile_b_col] = 0.0f;
    }

    __syncthreads();

    // move the matrix pointers to the next tile in the outer loop
    A += BK;
    B += BK * N;

    // make the dot product loop the outside loop, which allows us to cache the tile_b entry (the
    // entire point of this kernel)
    for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
      float tile_b_tmp = tile_b[dot_idx * BN + thread_col];
      // loop over the res_idx last
      for (int res_idx = 0; res_idx < TM; res_idx++) {
        thread_results[res_idx] +=
            tile_a[thread_row * TM * BK + res_idx * BK + dot_idx] * tile_b_tmp;
      }
    }

    __syncthreads();
  }

  // write the results to C
  for (int res_idx = 0; res_idx < TM; res_idx++) {
    if (global_row + res_idx < M && global_col < N) {
      C[(thread_row * TM + res_idx) * N + thread_col] =
          alpha * thread_results[res_idx] + beta * C[(thread_row * TM + res_idx) * N + thread_col];
    }
  }
}

template <const uint32_t BM, const uint32_t BN, const uint32_t BK, const uint32_t TM>
__global__ void sgemm_1d_blocktiling_v2_kernel(uint32_t M, uint32_t N, uint32_t K, float alpha,
                                               const float *A, const float *B, float beta,
                                               float *C) {
  // ensure that the sizes chosen for BM, BN, BK is supported by the launch config
  assert(BM * BK == blockDim.x);
  assert(BK * BN == blockDim.x);

  // block level indices of the current thread
  const uint32_t block_row = blockIdx.y;
  const uint32_t block_col = blockIdx.x;

  // thread level indices within the block
  const uint32_t thread_row = threadIdx.x / BN;
  const uint32_t thread_col = threadIdx.x % BN;

  // indices that the thread is responsible for in C
  const uint32_t global_row = block_row * BM + thread_row * TM;
  const uint32_t global_col = block_col * BN + thread_col;

  // allocate shared memory containers for storing A and B tiles
  __shared__ float tile_a[BM * BK];
  __shared__ float tile_b[BK * BN];

  // move the matrix pointers to the starting block
  A += block_row * BM * K;
  B += block_col * BN;
  C += block_row * BM * N + block_col * BN;

  // index of the element to load in tile_a
  const uint32_t tile_a_row = threadIdx.x / BK;
  const uint32_t tile_a_col = threadIdx.x % BK;
  const uint32_t tile_b_row = threadIdx.x / BN;
  const uint32_t tile_b_col = threadIdx.x % BN;

  // allocate thread-local cache for results
  float thread_results[TM] = {0.0f};

  // outer loop
  for (int tile_idx = 0; tile_idx < CEIL_DIV(K, BK); tile_idx++) {
    // load tile_a from global memory
    if (global_row < M && tile_idx * BK + tile_a_col < K) {
      tile_a[tile_a_row * BK + tile_a_col] = A[tile_a_row * K + tile_a_col];
    } else {
      tile_a[tile_a_row * BK + tile_a_col] = 0.0f;
    }
    // load tile_b from global memory
    if (tile_idx * BK + tile_b_row < K && global_col < N) {
      tile_b[tile_b_row * BN + tile_b_col] = B[tile_b_row * N + tile_b_col];
    } else {
      tile_b[tile_b_row * BN + tile_b_col] = 0.0f;
    }

    __syncthreads();

    // move the matrix pointers to the next tile in the outer loop
    A += BK;
    B += BK * N;

    // make the dot product loop the outside loop, which allows us to cache the tile_b entry (the
    // entire point of this kernel)
    for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
      float tile_b_tmp = tile_b[dot_idx * BN + thread_col];
      // loop over the res_idx last
      for (int res_idx = 0; res_idx < TM; res_idx++) {
        thread_results[res_idx] +=
            tile_a[thread_row * TM * BK + res_idx * BK + dot_idx] * tile_b_tmp;
      }
    }

    __syncthreads();
  }

  // write the results to C
  for (int res_idx = 0; res_idx < TM; res_idx++) {
    if (global_row + res_idx < M && global_col < N) {
      C[(thread_row * TM + res_idx) * N + thread_col] =
          alpha * thread_results[res_idx] + beta * C[(thread_row * TM + res_idx) * N + thread_col];
    }
  }
}
