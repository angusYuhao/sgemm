#pragma once

namespace sgemm {

void sgemm_naive_cuda(uint32_t M, uint32_t N, uint32_t K, const float *A,
                      const float *B, float *C, float alpha, float beta);

void sgemm_global_memory_coalesce_v1_cuda(uint32_t M, uint32_t N, uint32_t K,
                                          const float *A, const float *B,
                                          float *C, float alpha, float beta);

void sgemm_global_memory_coalesce_v2_cuda(uint32_t M, uint32_t N, uint32_t K,
                                          const float *A, const float *B,
                                          float *C, float alpha, float beta);

} // namespace sgemm
