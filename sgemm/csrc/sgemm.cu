#include <cuda_runtime.h>

#include "utils/macro.h"

#include "kernels/global_memory_coalesce.cuh"
#include "kernels/naive.cuh"
#include "kernels/shared_memory_blocking.cuh"

namespace sgemm {

void sgemm_naive_cuda(uint32_t M, uint32_t N, uint32_t K, const float *A, const float *B, float *C,
                      float alpha, float beta) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void sgemm_global_memory_coalesce_v1_cuda(uint32_t M, uint32_t N, uint32_t K, const float *A,
                                          const float *B, float *C, float alpha, float beta) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  sgemm_global_memory_coalesce_v1_kernel<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void sgemm_global_memory_coalesce_v2_cuda(uint32_t M, uint32_t N, uint32_t K, const float *A,
                                          const float *B, float *C, float alpha, float beta) {
  dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  dim3 blockDim(32, 32);
  sgemm_global_memory_coalesce_v2_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void sgemm_shared_memory_blocking_cuda(uint32_t M, uint32_t N, uint32_t K, const float *A,
                                       const float *B, float *C, float alpha, float beta) {
  dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  dim3 blockDim(32, 32);
  sgemm_shared_memory_blocking_kernel<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

} // namespace sgemm
