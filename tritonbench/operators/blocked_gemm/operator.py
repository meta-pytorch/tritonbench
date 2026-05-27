"""tritonbench operator for hammer.v3 blocked GEMM."""

import argparse
import sys
from typing import Any, Callable, List, Optional

import torch

from tritonbench.utils.env_utils import (
    IS_BLACKWELL,
    is_cuda,
    is_fbcode,
    IS_HOPPER,
)
from tritonbench.utils.path_utils import SUBMODULE_PATH
from tritonbench.utils.python_utils import try_import
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
    register_x_val,
)

from .generate_inputs import build_inputs, compute_flops, DTYPES


# In OSS, hammer.v3 and generative_recommenders are vendored under
# tritonbench/submodules. Insert them into sys.path once so the hammer
# imports below resolve identically to the fbcode build.
if not is_fbcode():
    for _sub in ("hammer", "generative-recommenders"):
        _p = str(SUBMODULE_PATH.joinpath(_sub))
        if _p not in sys.path:
            sys.path.insert(0, _p)


HAS_TRITON_BLOCKED_GEMM = False
HAS_PT_BLOCKED_GEMM = False

with try_import("HAS_TRITON_BLOCKED_GEMM"):
    from hammer.v3.ops.triton.triton_blocked_gemm import triton_blocked_gemm
with try_import("HAS_PT_BLOCKED_GEMM"):
    from hammer.v3.ops.pytorch.pt_blocked_gemm import pytorch_blocked_gemm


def parse_op_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TritonBench blocked_gemm operator"
    )
    parser.add_argument("--M", type=int, default=8192, help="Total M dimension")
    parser.add_argument("--N", type=int, default=8192, help="Total N dimension")
    parser.add_argument("--K", type=int, default=8192, help="Total K dimension")
    parser.add_argument(
        "--num-q-blocks",
        type=int,
        default=2,
        help="Number of row (A) blocks; must divide M",
    )
    parser.add_argument(
        "--num-w-blocks",
        type=int,
        default=2,
        help="weight (W) blocks; must divide N",
    )
    parser.add_argument(
        "--num-k-blocks",
        type=int,
        default=2,
        help="Number of inner-K blocks; must divide K",
    )
    parser.add_argument(
        "--has-bias",
        action="store_true",
        help="Add a per-W-block bias vector",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Tensor dtype for A/W",
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "tflops"]

    def __init__(
        self,
        tb_args: argparse.Namespace,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        super().__init__(tb_args, extra_args=extra_args)
        args = parse_op_args(self.extra_args)
        self.M = args.M
        self.N = args.N
        self.K = args.K
        self.num_q_blocks = args.num_q_blocks
        self.num_w_blocks = args.num_w_blocks
        self.num_k_blocks = args.num_k_blocks
        self.has_bias = args.has_bias
        self._op_dtype = DTYPES[args.dtype]
        self.requires_grad = self.mode != Mode.FWD_NO_GRAD

    def _dtype(self) -> torch.dtype:
        return self.dtype if self.dtype is not None else self._op_dtype

    def _triton_fn(
        self,
        A_list: List[List[torch.Tensor]],
        W_list: List[List[torch.Tensor]],
        bias_list: Optional[List[torch.Tensor]],
        version: str,
    ) -> Callable:
        return lambda: triton_blocked_gemm(
            A_list=A_list,
            W_list=W_list,
            bias_list=bias_list,
            version=version,
        )

    # -----------------------------------------------------------------
    # Input generation
    # -----------------------------------------------------------------

    def get_available_num_inputs(self) -> int:
        return 1

    def get_input_iter(self):
        device = torch.device(self.device)
        A_list, W_list, bias_list = build_inputs(
            M=self.M,
            N=self.N,
            K=self.K,
            num_q_blocks=self.num_q_blocks,
            num_k_blocks=self.num_k_blocks,
            num_w_blocks=self.num_w_blocks,
            has_bias=self.has_bias,
            dtype=self._dtype(),
            device=device,
            requires_grad=self.requires_grad,
        )
        yield A_list, W_list, bias_list

    # -----------------------------------------------------------------
    # Backends
    # -----------------------------------------------------------------

    @register_benchmark(baseline=True, enabled=HAS_PT_BLOCKED_GEMM)
    def pytorch(
        self,
        A_list: List[List[torch.Tensor]],
        W_list: List[List[torch.Tensor]],
        bias_list: Optional[List[torch.Tensor]],
    ) -> Callable:
        return lambda: pytorch_blocked_gemm(
            A_list=A_list, W_list=W_list, bias_list=bias_list
        )

    @register_benchmark(enabled=HAS_TRITON_BLOCKED_GEMM and is_cuda())
    def triton_blocked_gemm_persistent(
        self,
        A_list: List[List[torch.Tensor]],
        W_list: List[List[torch.Tensor]],
        bias_list: Optional[List[torch.Tensor]],
    ) -> Callable:
        # version="" picks the persistent TMA fwd kernel.
        return self._triton_fn(A_list, W_list, bias_list, version="")

    @register_benchmark(
        enabled=HAS_TRITON_BLOCKED_GEMM and is_cuda() and (IS_HOPPER or IS_BLACKWELL)
    )
    def triton_blocked_gemm_ws(
        self,
        A_list: List[List[torch.Tensor]],
        W_list: List[List[torch.Tensor]],
        bias_list: Optional[List[torch.Tensor]],
    ) -> Callable:
        # version="ws" picks the warp-specialized fwd kernel. The kernel now
        # has both Hopper (register accumulator + tlx.async_dot_wait) and
        # Blackwell (TMEM accumulator + tcgen05_commit + local_load) paths,
        # dispatched by an IS_BLACKWELL constexpr threaded from the Python op.
        return self._triton_fn(A_list, W_list, bias_list, version="ws")

    @register_benchmark(enabled=HAS_TRITON_BLOCKED_GEMM and is_cuda())
    def triton_blocked_gemm_bwd(
        self,
        A_list: List[List[torch.Tensor]],
        W_list: List[List[torch.Tensor]],
        bias_list: Optional[List[torch.Tensor]],
    ) -> Callable:
        """Forward wrapper used by the bwd-mode OKR config.

        Returns the autograd-wrapped fwd; when the bench runs this in
        mode=bwd, .backward() dispatches to
        _BlockedGemmFunction.backward -> triton_blocked_gemm_backward.
        """
        return self._triton_fn(A_list, W_list, bias_list, version="ws")

    # -----------------------------------------------------------------
    # x-val and metrics
    # -----------------------------------------------------------------

    @register_x_val(label="(M, N, K, num_q, num_w, num_k)")
    def get_x_val(self, example_inputs: Any):
        return (
            self.M,
            self.N,
            self.K,
            self.num_q_blocks,
            self.num_w_blocks,
            self.num_k_blocks,
        )

    @register_metric()
    def flops(
        self,
        fn_name: str,
        example_inputs: Any,
        metrics: BenchmarkOperatorMetrics,
    ) -> float:
        A_list, W_list, bias_list = example_inputs
        return compute_flops(A_list, W_list, bias_list, mode=self.mode.value)
