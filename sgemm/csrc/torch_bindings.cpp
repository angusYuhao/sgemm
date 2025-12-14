#include <Python.h>

#include <torch/all.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "sgemm.h"

// borrowed from:
// https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
  The import from Python will load the .so consisting of this file
  in this extension, so that the TORCH_LIBRARY static initializers
  below are run. */
PyObject *PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1,   /* size of per-interpreter state of the module,
              or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

void print_sgemm_version_error() { std::cout << "Invalid SgemmVersion!\n"; }

namespace sgemm {

enum class SgemmVersion {
  sgemm_naive,
  sgemm_global_memory_coalesce_v1,
  sgemm_global_memory_coalesce_v2,
  sgemm_shared_memory_blocking,
};

void sgemm_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C, double alpha,
                double beta, int64_t version) {
  SgemmVersion sgemm_version = static_cast<SgemmVersion>(version);
  uint32_t M = C.sizes()[0];
  uint32_t N = C.sizes()[1];
  uint32_t K = A.sizes()[1];
  switch (sgemm_version) {
  case SgemmVersion::sgemm_naive:
    sgemm_naive_cuda(M, N, K, (float *)(A.data_ptr()), (float *)(B.data_ptr()),
                     (float *)(C.data_ptr()), static_cast<float>(alpha), static_cast<float>(beta));
    break;
  case SgemmVersion::sgemm_global_memory_coalesce_v1:
    sgemm_global_memory_coalesce_v1_cuda(M, N, K, (float *)(A.data_ptr()), (float *)(B.data_ptr()),
                                         (float *)(C.data_ptr()), static_cast<float>(alpha),
                                         static_cast<float>(beta));
    break;
  case SgemmVersion::sgemm_global_memory_coalesce_v2:
    sgemm_global_memory_coalesce_v2_cuda(M, N, K, (float *)(A.data_ptr()), (float *)(B.data_ptr()),
                                         (float *)(C.data_ptr()), static_cast<float>(alpha),
                                         static_cast<float>(beta));
    break;
  case SgemmVersion::sgemm_shared_memory_blocking:
    print_sgemm_version_error();
    break;
  default:
    print_sgemm_version_error();
    break;
  }
}

TORCH_LIBRARY(sgemm, m) {
  m.def("sgemm(Tensor A, Tensor B, Tensor C,"
        "      float alpha, float beta, int version) -> ()");
}

TORCH_LIBRARY_IMPL(sgemm, CUDA, m) { m.impl("sgemm", &sgemm_cuda); }

} // namespace sgemm
