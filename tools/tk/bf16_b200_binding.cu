#include "../../submodules/ThunderKittens/kernels/gemm/bf16_b200/bf16_b200_gemm.cu"
#include "pyutils/torchutils.cuh"
#include <ATen/Functions.h>

at::Tensor bf16_b200_gemm_binding(const at::Tensor &A, const at::Tensor &B_t) {
    CHECK_INPUT(A);
    CHECK_INPUT(B_t);
    kittens::py::device_check(A, B_t);

    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B_t.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.scalar_type() == at::ScalarType::BFloat16, "A must be bfloat16");
    TORCH_CHECK(B_t.scalar_type() == at::ScalarType::BFloat16, "B must be bfloat16");
    TORCH_CHECK(A.size(1) == B_t.size(1), "K dimension mismatch");

    using C = config<256, 256, 64, 8, false, 4, 8>;
    using G = globals<C>;

    TORCH_CHECK(
        A.size(0) % (C::Mb * C::NUM_CONSUMERS) == 0,
        "M must be divisible by ",
        C::Mb * C::NUM_CONSUMERS
    );
    TORCH_CHECK(B_t.size(0) % C::Nb == 0, "N must be divisible by ", C::Nb);
    TORCH_CHECK(A.size(1) % C::Kb == 0, "K must be divisible by ", C::Kb);

    at::Tensor D = at::empty({A.size(0), B_t.size(0)}, A.options());

    G g{
        .a = kittens::py::tensor_to_gl<typename G::a_gl>(A),
        .b = kittens::py::tensor_to_gl<typename G::b_gl>(B_t),
        .d = kittens::py::tensor_to_gl<typename G::d_gl>(D),
    };

    CUDACHECK(
        cudaFuncSetAttribute(
            kernel<C>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            g.dynamic_shared_memory()
        )
    );
    LaunchConfig<true, false> launch_config(
        g.grid(),
        g.block(),
        g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(),
        C::CLUSTER_SIZE
    );
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel<C>, g));
    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "bf16_b200_gemm",
        &bf16_b200_gemm_binding,
        "Blackwell bf16 GEMM. Expects A as (M, K) and B^T as (N, K). Returns (M, N)."
    );
}
