import argparse
from typing import Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.operators.fused_rmsnorm_matmul.triton_autows import (
    triton_autows_fused_rmsnorm_gemm,
)
from tritonbench.utils.env_utils import IS_BLACKWELL
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    register_x_val,
)


def parse_op_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fused RMSNorm + matmul")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=12800)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument(
        "--a-rrms-source",
        choices=("scalar", "tma"),
        default="tma",
    )
    return parser.parse_args(args)


def _rmsnorm_matmul_aten_bf16_gemm_then_scale(
    x: torch.Tensor,
    b_weighted_t: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rrms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    # torch.mm returns bf16 here; .float() cannot recover the internal fp32 accumulator.
    return (torch.mm(x, b_weighted_t.t()).float() * rrms).to(x.dtype)


class Operator(BenchmarkOperator):
    FWD_ONLY = True

    def __init__(
        self,
        tb_args: argparse.Namespace,
        extra_args: Optional[List[str]] = None,
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.M = args.m
        self.N = args.n
        self.K = args.k
        self.eps = args.eps
        self.use_shared_a_rrms = args.a_rrms_source == "tma"

    @register_x_val(label="(M, N, K)")
    def get_x_val(self, example_inputs) -> Tuple[int, int, int]:
        x, b, _weight, _b_weighted_t = example_inputs
        return (x.shape[0], b.shape[1], x.shape[1])

    @register_benchmark(baseline=True)
    def aten_bf16_gemm_then_scale(self, x, b, weight, b_weighted_t) -> Callable:
        def inner():
            return _rmsnorm_matmul_aten_bf16_gemm_then_scale(
                x,
                b_weighted_t,
                self.eps,
            )

        return inner

    @register_benchmark(enabled=IS_BLACKWELL)
    def triton_autows(self, x, b, weight, b_weighted_t) -> Callable:
        def inner():
            return triton_autows_fused_rmsnorm_gemm(
                x,
                b_weighted_t,
                self.eps,
                use_shared_a_rrms=self.use_shared_a_rrms,
            )

        return inner

    def get_input_iter(self) -> Generator:
        x = torch.randn(
            self.M,
            self.K,
            dtype=self.dtype,
            device=self.device,
        )
        b = torch.randn(
            self.K,
            self.N,
            dtype=self.dtype,
            device=self.device,
        )
        weight = (
            torch.randn(
                self.K,
                dtype=self.dtype,
                device=self.device,
            )
            * 0.1
            + 1.0
        )
        b_weighted = (weight[:, None].float() * b.float()).to(self.dtype)
        b_weighted_t = b_weighted.T.contiguous()
        yield x, b, weight, b_weighted_t
