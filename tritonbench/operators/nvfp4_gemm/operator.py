import argparse
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
import torch._inductor.config as inductor_config
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
    register_x_val,
)


def parse_op_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NVF4 blockscaled GEMM benchmark")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--out-dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
    )
    return parser.parse_args(args)


BUILTIN_SHAPES = [
    (128, 256, 512),
    (256, 512, 1024),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 1024),
    (8192, 4096, 1024),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (2048, 8192, 2048),
    (4096, 13312, 2304),
]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(x, multiple):
    return _ceil_div(x, multiple) * multiple


def _prep_nvfp4_inputs(m, n, k, device):
    """Prepare NVF4 inputs: FP4 data + E4M3FN scales, block_size=16."""
    block_size = 16
    packed_k = k // 2

    a_fp4 = torch.randint(0, 256, (m, packed_k), device=device, dtype=torch.uint8).view(
        torch.float4_e2m1fn_x2
    )
    b_fp4 = torch.randint(0, 256, (n, packed_k), device=device, dtype=torch.uint8).view(
        torch.float4_e2m1fn_x2
    )
    b_fp4_t = b_fp4.T

    num_k_blocks = _ceil_div(k, block_size)
    padded_k_blocks = _round_up(num_k_blocks, 4)
    block_size_mn = 128
    scale_a_numel = block_size_mn * _ceil_div(m, block_size_mn) * padded_k_blocks
    scale_b_numel = block_size_mn * _ceil_div(n, block_size_mn) * padded_k_blocks

    scale_a = torch.rand(scale_a_numel, device=device).to(torch.float8_e4m3fn)
    scale_b = torch.rand(scale_b_numel, device=device).to(torch.float8_e4m3fn)

    return a_fp4, b_fp4_t, scale_a, scale_b


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "speedup", "accuracy"]
    DEFAULT_PRECISION = "fp4"
    FWD_ONLY = True

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        self.use_cuda_graphs = True
        args = parse_op_args(self.extra_args)
        self.out_dtype = (
            torch.float32 if args.out_dtype == "float32" else torch.bfloat16
        )

        if args.m and args.n and args.k:
            self.shapes = [(args.m, args.n, args.k)]
        else:
            self.shapes = BUILTIN_SHAPES

    @register_benchmark(baseline=True)
    def torch_scaled_mm(self, a, b, scale_a, scale_b, m, n, k) -> Callable:
        out_dtype = self.out_dtype

        def fn():
            return torch._scaled_mm(
                a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype
            )

        return fn

    # TODO: disabled because of test failure
    # Error: "torch._inductor.exc.InductorError: LoweringException: NoValidChoicesError: No choices to select."
    @register_benchmark(enabled=False)
    def pt2_nvgemm_scaled_mm(self, a, b, scale_a, scale_b, m, n, k) -> Callable:
        out_dtype = self.out_dtype
        torch._dynamo.reset()

        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="NVGEMM",
            autotune_fallback_to_aten=False,
        ):
            compiled = torch.compile(
                lambda a, b: torch._scaled_mm(
                    a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype
                ),
                dynamic=False,
            )
            compiled(a, b)

        return lambda: compiled(a, b)

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        _, _, _, _, m, n, k = example_inputs
        return 2.0 * m * n * k

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        _, _, _, _, m, n, k = example_inputs
        return (m, n, k)

    def get_input_iter(self) -> Generator:
        device = self.device

        for m, n, k in self.shapes:
            if m % 128 != 0 or n % 128 != 0 or k % 32 != 0:
                continue

            a, b, sa, sb = _prep_nvfp4_inputs(m, n, k, device)
            yield a, b, sa, sb, m, n, k

    def _get_accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output = fn()
        baseline_output = baseline_fn()
        try:
            torch.testing.assert_close(output, baseline_output, atol=1e-2, rtol=0.5)
            return True
        except Exception:
            return False

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        return None
