#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

template <const uint32_t BLOCK_SIZE>
__global__ void sgemm_global_memory_coalesce_v1_kernel(uint32_t M, uint32_t N,
                                                       uint32_t K, float alpha,
                                                       const float *A,
                                                       const float *B,
                                                       float beta, float *C) {
    const uint32_t m = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const uint32_t n = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (m < M && n < N) {
        float accum = 0;
        for (int k = 0; k < K; k++) {
            accum += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = alpha * accum + beta * C[m * N + n];
    }
}

__global__ void sgemm_global_memory_coalesce_v2_kernel(uint32_t M, uint32_t N,
                                                       uint32_t K, float alpha,
                                                       const float *A,
                                                       const float *B,
                                                       float beta, float *C) {
    const uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        float accum = 0;
        for (int k = 0; k < K; k++) {
            accum += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = alpha * accum + beta * C[m * N + n];
    }
}
