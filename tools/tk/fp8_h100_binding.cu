#define TORCH_COMPILE
#include "../../submodules/ThunderKittens/kernels/gemm/fp8_h100/fp8_h100_gemm.cu"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fp8_gemm",
        &fp8_gemm,
        "H100 fp8 GEMM. Expects A as (M, K) and B^T as (N, K)."
    );
}
