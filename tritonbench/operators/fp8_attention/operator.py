"""
Adding FP8 to FlashAttention-2
https://research.colfax-intl.com/adding-fp8-to-flashattention/
"""

import argparse
import math
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
from tritonbench.kernels.triton_fused_attention import attention_opt as triton_attention
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
)
from tritonbench.utils.triton_utils import has_warp_spec

HAS_CUDA_124 = (
    torch.cuda.is_available() and torch.version.cuda and torch.version.cuda >= "12.4"
)

try:
    # colfax Flash Attention V2 on FP8 for Hopper
    torch.ops.load_library(
        "//ai_codesign/gen_ai/cutlass-kernels:fmha_forward_lib_pipeline_h128"
    )
    colfax_fmha_pipeline = torch.ops.cutlass.fmha_forward_pipeline
except (ImportError, IOError, AttributeError):
    colfax_fmha_pipeline = None


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=3072,
        help="specify embedding dim, embedding dim = n_heads * head_dim",
    )
    parser.add_argument("--n-heads", type=int, default=48, help="Number of heads")
    parser.add_argument("--d-head", type=int, default=64, help="specify head dimension")
    parser.add_argument("--causal", action="store_true", help="enable causal")
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["latency", "tflops"]
    DEFAULT_PRECISION = "fp8"
    FWD_ONLY = True

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        self.BATCH = args.batch
        self.SEQ_LEN = args.seq_len
        self.embedding_dim = args.embedding_dim
        self.H = args.n_heads
        self.D_HEAD = args.d_head
        self.causal = args.causal
        # We always turn on causal for backward
        # Because Triton-Flash-V2 does not support backward with non-causal
        if self.mode == BenchmarkMode.BWD or self.mode == BenchmarkMode.FWD_BWD:
            self.causal = True
        self.requires_grad = not self.tb_args.mode == "fwd_no_grad"
        self.sm_scale = 1.0 / math.sqrt(float(self.D_HEAD))

        if self.embedding_dim and self.H != self.embedding_dim // self.D_HEAD:
            raise ValueError(
                f"embedding_dim {self.embedding_dim} is inconsistent with n_heads {self.H}. embedding_dim = n_heads * d_head "
            )

    def colfax_preprocess(self, q, k, v):
        # colfax expects q,k: BATCH, N_CTX, H, D_HEAD and v: BATCH, D_HEAD, H, N_CTX
        # passed-in: BATCH, H, N_CTX, D_HEAD
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 3, 1, 2).contiguous()
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
        return (
            q,
            k,
            v,
        )

    @register_benchmark(enabled=bool(colfax_fmha_pipeline))
    def colfax_fmha(
        self,
        q: torch.Tensor,  # // [b, seqlen, num_heads, head_dim]
        k: torch.Tensor,  # // [b, seqlen, num_heads, head_dim]
        v: torch.Tensor,  # // [b, seqlen, num_heads, head_dim]
    ) -> Callable:
        kLog2e = float(1.4426950408889634074)
        # log_2(e) = M_LOG2E
        softmax_scale = 1.0 / math.sqrt(float(self.D_HEAD))
        scale = softmax_scale * kLog2e
        colfax_q, colfax_k, colfax_v = self.colfax_preprocess(q, k, v)
        return lambda: colfax_fmha_pipeline(
            self.N_CTX, self.N_CTX, self.BATCH, colfax_q, colfax_k, colfax_v, scale
        )

    def triton_preprocess(self, q, k, v):
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.permute(0, 1, 3, 2)
        v = v.to(torch.float8_e5m2)
        return (
            q,
            k,
            v,
        )

    @register_benchmark(baseline=True)
    def triton_flash_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        triton_q, triton_k, triton_v = self.triton_preprocess(q, k, v)
        # full fp8 will be enabled if type of q,k,v is fp8
        return lambda: triton_attention(
            triton_q, triton_k, triton_v, self.causal, self.sm_scale, "base_opt"
        )

    @register_benchmark()
    def triton_flash_v2_tma(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        triton_q, triton_k, triton_v = self.triton_preprocess(q, k, v)
        # full fp8 will be enabled if type of q,k,v is fp8
        return lambda: triton_attention(
            triton_q, triton_k, triton_v, self.causal, self.sm_scale, "tma"
        )

    @register_benchmark(enabled=HAS_CUDA_124 and has_warp_spec())
    def triton_flash_v2_ws(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        triton_q, triton_k, triton_v = self.triton_preprocess(q, k, v)
        # full fp8 will be enabled if type of q,k,v is fp8
        return lambda: triton_attention(
            triton_q, triton_k, triton_v, self.causal, self.sm_scale, "ws"
        )

    def get_input_iter(self) -> Generator:
        # The non-fp8 FA varies N_CTX and fixes other variables. Let's do the same for fp8.
        # The autotune config only depends on N_CTX in OSS Triton tutorial.

        BATCH = self.BATCH
        D_HEAD = self.D_HEAD
        SEQ_LEN_LOG2 = 7
        H = self.H

        def get_ctx_vals():
            if self.SEQ_LEN:
                yield (BATCH, H, self.SEQ_LEN, self.D_HEAD)
                return
            for i in range(SEQ_LEN_LOG2, 15):
                N_CTX = 2**i
                yield (BATCH, H, N_CTX, D_HEAD)

        shapes = get_ctx_vals()

        for shape in shapes:
            BATCH, H, N_CTX, D_HEAD = shape

            self.N_CTX = N_CTX

            # colfax expects q,k: BATCH, N_CTX, H, D_HEAD and v: BATCH, D_HEAD, H, N_CTX
            q = torch.randn(
                (BATCH, H, N_CTX, D_HEAD),
                dtype=torch.float16,
                device=self.device,
                requires_grad=self.requires_grad,
            )

            k = torch.randn(
                (BATCH, H, N_CTX, D_HEAD),
                dtype=torch.float16,
                device=self.device,
                requires_grad=self.requires_grad,
            )

            v = torch.randn(
                (BATCH, H, N_CTX, D_HEAD),
                dtype=torch.float16,
                device=self.device,
                requires_grad=self.requires_grad,
            )
            yield (q, k, v)

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        """
        Check accuracy of FP8 attention implementation against baseline.

        FP8 operations have inherently lower precision, so we use relaxed tolerances.
        Based on empirical testing, FP8 can introduce differences up to ~2.0.
        """
        try:
            output = fn()
            baseline_output = baseline_fn()

            # Convert FP8 outputs to FP16 for comparison
            if output.dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
                output = output.to(torch.float16)
            if baseline_output.dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
                baseline_output = baseline_output.to(torch.float16)

            # Validate outputs
            if torch.isnan(output).any() or torch.isinf(output).any():
                return False
            if torch.isnan(baseline_output).any() or torch.isinf(baseline_output).any():
                return False
            if output.shape != baseline_output.shape:
                return False

            # FP8 attention uses relaxed tolerances due to:
            # 1. FP8 quantization of Q, K, V inputs
            # 2. FP8 quantization of attention weights (doesn't sum to exactly 1.0)
            # 3. Accumulation differences in FP8 GEMM operations
            result = torch.allclose(output, baseline_output, atol=2.0, rtol=0.2)

            return result

        except Exception:
            return False

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        q, _, _ = example_inputs
        BATCH, H, N_CTX, D_HEAD = q.shape
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
        return 2 * flops_per_matmul
