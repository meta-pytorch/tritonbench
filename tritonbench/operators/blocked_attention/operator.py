"""tritonbench operator for hammer.v3 blocked attention.

Each backend imports its kernel module independently so that a single
broken sub-import (e.g. cute-dsl runtime mismatch) only disables that
one backend instead of taking down the whole operator.
"""

import argparse
import logging
import sys
from typing import Any, Callable, List, Optional

import torch
from torch.utils._pytree import tree_map

from tritonbench.utils.env_utils import IS_BLACKWELL, is_cuda, is_fbcode
from tritonbench.utils.path_utils import SUBMODULE_PATH
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
    register_x_val,
)

# In OSS, hammer.v3 and generative_recommenders are vendored under
# tritonbench/submodules. Insert them into sys.path once so the hammer
# imports below resolve identically to the fbcode build.
if not is_fbcode():
    for _sub in ("hammer", "generative-recommenders"):
        _p = str(SUBMODULE_PATH.joinpath(_sub))
        if _p not in sys.path:
            sys.path.insert(0, _p)

from .generate_inputs import block_flops, build_inputs, DTYPES, HAS_HAMMER_V3


logger = logging.getLogger(__name__)


# [Optional] general triton kernel
try:
    # @manual=//hammer/v3/ops/triton:triton_attention
    from hammer.v3.ops.triton.triton_attention import triton_mha

    HAS_TRITON_MHA = True
except (ImportError, IOError, AttributeError):
    HAS_TRITON_MHA = False

# [Optional] TLX Blackwell kernel (warp-specialized, pipelined)
try:
    # @manual=//hammer/v3/ops/triton:tlx_block_attention
    from hammer.v3.ops.triton.tlx_block_attention import tlx_mha

    HAS_TLX_MHA = True
except (ImportError, IOError, AttributeError):
    HAS_TLX_MHA = False

# [Optional] CuTeDSL Blackwell kernel
try:
    # @manual=//hammer/v3/ops/cutedsl:cutedsl_attention
    from hammer.v3.ops.cutedsl.cutedsl_attention import cutedsl_mha

    HAS_CUTEDSL_MHA = True
except (ImportError, IOError, AttributeError):
    HAS_CUTEDSL_MHA = False

logger.info(
    "blocked_attention backends available: "
    "hammer_v3=%s triton=%s tlx=%s cutedsl=%s",
    HAS_HAMMER_V3,
    HAS_TRITON_MHA,
    HAS_TLX_MHA,
    HAS_CUTEDSL_MHA,
)


def parse_op_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TritonBench blocked_attention operator"
    )
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--attn-dim", type=int, default=128, help="Q/K head dim")
    parser.add_argument("--hidden-dim", type=int, default=128, help="V head dim")
    parser.add_argument(
        "--min-seq-len-log2", type=int, default=8, help="Inclusive log2 min seq_len"
    )
    parser.add_argument(
        "--max-seq-len-log2",
        type=int,
        default=10,
        help="Inclusive log2 max seq_len (sweep is 2**i for i in [min..max])",
    )
    parser.add_argument(
        "--seq-sparsity",
        type=float,
        default=0.9,
        help="Average sparsity of generated sequence lengths",
    )
    parser.add_argument(
        "--sampling-alpha",
        type=float,
        default=2.0,
        help="Sampling alpha for per-batch sequence-length distribution",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=0,
        help="Max target tokens per sequence (HSTU target block)",
    )
    parser.add_argument(
        "--full-attn-size",
        type=int,
        default=0,
        help="Local-full tokens per sequence (HSTU local-full block)",
    )
    parser.add_argument(
        "--max-attn-len",
        type=int,
        default=0,
        help="LOCAL window size (0 disables LOCAL handling)",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="causal",
        choices=["causal", "all", "local", "diagonal"],
        help="Mask used by the 1-block scenario only",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Tensor dtype for q/k/v",
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
        self.batch_size = args.batch_size
        self.num_heads = args.heads
        self.attn_dim = args.attn_dim
        self.hidden_dim = args.hidden_dim
        self.min_seq_len_log2 = args.min_seq_len_log2
        self.max_seq_len_log2 = args.max_seq_len_log2
        self.sparsity = args.seq_sparsity
        self.sampling_alpha = args.sampling_alpha
        self.target_size = args.target_size
        self.full_attn_size = args.full_attn_size
        self.max_attn_len = args.max_attn_len
        self.mask_type = args.mask_type
        self.alpha = 1.0 / (self.attn_dim**0.5)
        self._op_dtype = DTYPES[args.dtype]
        self.requires_grad = self.mode in (Mode.BWD, Mode.FWD_BWD)

    def _dtype(self) -> torch.dtype:
        return self.dtype if self.dtype is not None else self._op_dtype

    def get_available_num_inputs(self) -> int:
        return (self.max_seq_len_log2 + 1) - self.min_seq_len_log2

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        # The blocked-attention kernels (triton_mha / tlx_mha / cutedsl_mha)
        # return a List[Tensor] — one output per Q block. The base class's
        # get_bwd_fn assumes a single Tensor (or a tuple whose first element
        # is one) and would do randn_like(list), which fails. Build a per-
        # output dy list and use torch.autograd.backward instead.
        grad_tensors: List[torch.Tensor] = []

        def _collect(x: Any) -> Any:
            if isinstance(x, torch.Tensor) and x.requires_grad:
                grad_tensors.append(x)
            return x

        tree_map(_collect, self.example_inputs)

        state: dict[str, Any] = {"y": None, "dy": None}

        def bwd_fn():
            for t in grad_tensors:
                if t.grad is not None:
                    t.grad = None

            if state["y"] is None:
                output = fwd_fn()
                ys = (
                    list(output)
                    if isinstance(output, (list, tuple))
                    else [output]
                )
                state["y"] = ys
                torch.manual_seed(0)
                state["dy"] = [0.1 * torch.randn_like(t) for t in ys]

            torch.autograd.backward(state["y"], state["dy"], retain_graph=True)
            return grad_tensors

        return bwd_fn

    def get_input_iter(self):
        device = torch.device(self.device)
        for seq_len in (
            2**i for i in range(self.min_seq_len_log2, self.max_seq_len_log2 + 1)
        ):
            (
                q_list,
                k_list,
                v_list,
                q_offsets_list,
                kv_offsets_list,
                attn_scale_list,
                mask_matrix,
            ) = build_inputs(
                batch_size=self.batch_size,
                heads=self.num_heads,
                max_seq_len=seq_len,
                attn_dim=self.attn_dim,
                hidden_dim=self.hidden_dim,
                dtype=self._dtype(),
                device=device,
                sparsity=self.sparsity,
                target_size=self.target_size,
                full_attn_size=self.full_attn_size,
                sampling_alpha=self.sampling_alpha,
                mask_type=self.mask_type,
                requires_grad=self.requires_grad,
            )
            # mask_matrix is a List[List[MaskType]] which the tritonbench
            # input_cast tree-walker cannot handle (only tensors / primitives /
            # a few known wrapper types). Stash it on self instead and yield
            # only tensor-shaped data. mask_matrix is purely structural - it
            # depends on (target_size, full_attn_size, mask_type) which are
            # operator args, so it is identical every iteration.
            self.mask_matrix = mask_matrix
            yield (
                q_list,
                k_list,
                v_list,
                q_offsets_list,
                kv_offsets_list,
                attn_scale_list,
                seq_len,
            )

    @register_benchmark(baseline=True, enabled=HAS_TRITON_MHA)
    def triton(
        self,
        q_list,
        k_list,
        v_list,
        q_offsets_list,
        kv_offsets_list,
        attn_scale_list,
        max_seq_len,
    ) -> Callable:
        return lambda: triton_mha(
            alpha=self.alpha,
            q_list=q_list,
            k_list=k_list,
            v_list=v_list,
            q_seq_offsets_list=q_offsets_list,
            mask_matrix=self.mask_matrix,
            attn_scale_list=attn_scale_list,
            kv_seq_offsets_list=kv_offsets_list,
            max_attn_len=self.max_attn_len,
        )

    @register_benchmark(
        enabled=HAS_TLX_MHA and is_cuda() and IS_BLACKWELL,
        fwd_only=True,
    )
    def tlx_blackwell_ws_pipelined(
        self,
        q_list,
        k_list,
        v_list,
        q_offsets_list,
        kv_offsets_list,
        attn_scale_list,
        max_seq_len,
    ) -> Optional[Callable]:
        # The TLX kernel only supports 1- or 2-block scenarios
        # (asserts num_q_tensors <= 2 in tlx_block_attention_fwd).
        # 3-block HSTU layouts (--target-size > 0 AND --full-attn-size > 0)
        # are not yet supported.
        if len(q_list) > 2:
            return None
        return lambda: tlx_mha(
            alpha=self.alpha,
            q_list=q_list,
            k_list=k_list,
            v_list=v_list,
            q_seq_offsets_list=q_offsets_list,
            mask_matrix=self.mask_matrix,
            attn_scale_list=attn_scale_list,
            kv_seq_offsets_list=kv_offsets_list,
            max_attn_len=self.max_attn_len,
        )

    @register_benchmark(enabled=HAS_CUTEDSL_MHA and is_cuda() and IS_BLACKWELL)
    def cutedsl_blackwell(
        self,
        q_list,
        k_list,
        v_list,
        q_offsets_list,
        kv_offsets_list,
        attn_scale_list,
        max_seq_len,
    ) -> Optional[Callable]:
        # CuTeDSL blocked attention requires dim_q == dim_v and takes
        # stacked offsets rather than lists.
        if self.attn_dim != self.hidden_dim:
            return None
        q_seq_offsets = torch.stack(q_offsets_list)
        kv_seq_offsets = torch.stack(kv_offsets_list)
        return lambda: cutedsl_mha(
            Q_list=q_list,
            K_list=k_list,
            V_list=v_list,
            q_seq_offsets=q_seq_offsets,
            kv_seq_offsets=kv_seq_offsets,
            attn_scale_list=attn_scale_list,
            mask_matrix=self.mask_matrix,
            alpha=self.alpha,
            max_attn_len=self.max_attn_len,
        )

    @register_x_val(
        label="(B, H, S, D_q, D_v, target, full_attn, max_attn_len)"
    )
    def get_x_val(self, example_inputs: Any):
        seq_len = example_inputs[-1]
        return (
            self.batch_size,
            self.num_heads,
            seq_len,
            self.attn_dim,
            self.hidden_dim,
            self.target_size,
            self.full_attn_size,
            self.max_attn_len,
        )

    @register_metric()
    def flops(
        self,
        fn_name: str,
        example_inputs: Any,
        metrics: BenchmarkOperatorMetrics,
    ) -> float:
        (
            _q,
            _k,
            _v,
            q_offsets_list,
            kv_offsets_list,
            _scales,
            _seq_len,
        ) = example_inputs
        return block_flops(
            batch_size=self.batch_size,
            attn_dim=self.attn_dim,
            hidden_dim=self.hidden_dim,
            nheads=self.num_heads,
            q_offsets_list=q_offsets_list,
            kv_offsets_list=kv_offsets_list,
            mask_matrix=self.mask_matrix,
            max_attn_len=self.max_attn_len,
            mode=self.mode.value,
        )
