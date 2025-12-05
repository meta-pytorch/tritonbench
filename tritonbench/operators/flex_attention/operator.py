import argparse
import random
from functools import partial
from itertools import product
from typing import Any, Callable, Generator, List, NamedTuple, Optional, Tuple, Union

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

# from attn_gym.masks.document_mask import length_to_offsets
# from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.nn.functional import scaled_dot_product_attention as sdpa

# Optional Flash Attention v3 import
HAS_FLASH_V3 = False
try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func

    HAS_FLASH_V3 = True
except ImportError:
    pass

from tritonbench.utils.env_utils import is_hip
from tritonbench.utils.input import input_filter
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode as BenchmarkMode,
    register_benchmark,
    register_metric,
    register_x_val,
)

from .mods import (
    causal_mask,
    generate_alibi_bias,
    generate_doc_mask_mod,
    generate_prefix_lm_mask,
    generate_sliding_window,
    generate_tanh_softcap,
    length_to_offsets,
)


torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.recompile_limit = 10000
torch._dynamo.config.accumulated_recompile_limit = 10000

MOD_TYPES = [
    "noop",
    "causal",
    "rel",
    "head_bias",
    "alibi",
    "sliding_window",
    "document_mask",
    "prefix_lm",
    "softcap",
]


class FullShape(NamedTuple):
    B: int
    Hq: int
    M: int
    Hkv: int
    N: int
    D: int

    def __str__(self):
        return f"({self.B}, {self.Hq}, {self.M}, {self.Hkv}, {self.N}, {self.D})"


def unsupported_fn():
    raise NotImplementedError("not supported")


def parse_op_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument(
        "--n-heads-q", type=int, default=16, help="Number of query heads"
    )
    parser.add_argument(
        "--n-heads-kv",
        type=int,
        default=None,
        help="Number of key/value heads (defaults to n-heads-q)",
    )
    parser.add_argument(
        "--d-head", type=int, default=128, help="specify head dimension"
    )
    allowed_mods = MOD_TYPES + ["all"]
    parser.add_argument(
        "--mod-type",
        type=str,
        default="all",
        choices=allowed_mods,
        help="Mask type",
    )
    parser.add_argument(
        "--max-autotune", action="store_true", help="Whether to enable max autotune"
    )
    parser.add_argument(
        "--sliding-window-size", type=int, default=4096, help="sliding window size"
    )
    parser.add_argument("--prefix-length", type=int, default=128, help="prefix length")
    parser.add_argument(
        "--decoding",
        action="store_true",
        help="Benchmark decoding mode (query seq len = 1)",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Enable dynamic shapes compilation"
    )
    return parser.parse_args(args)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    DEFAULT_METRICS = ["latency", "tflops", "tbps"]

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)

        self.batch_size = args.batch
        self.seq_len = args.seq_len
        self.num_heads_q = args.n_heads_q
        self.num_heads_kv = (
            args.n_heads_kv if args.n_heads_kv is not None else args.n_heads_q
        )
        self.head_dim = args.d_head
        self.max_autotune = args.max_autotune
        self.mod_type = args.mod_type
        self.sliding_window_size = args.sliding_window_size
        self.prefix_length = args.prefix_length
        self.decoding = args.decoding
        self.dynamic = args.dynamic

    def get_input_iter(self) -> Generator:
        """Generate a single input configuration for benchmarking."""
        # Set seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)

        def get_ctx_vals():
            if self.seq_len:
                if self.decoding:
                    # Decoding mode: query length = 1, kv length = seq_len
                    q_shape = (self.batch_size, self.num_heads_q, 1, self.head_dim)
                    kv_shape = (
                        self.batch_size,
                        self.num_heads_kv,
                        self.seq_len,
                        self.head_dim,
                    )
                else:
                    q_shape = (
                        self.batch_size,
                        self.num_heads_q,
                        self.seq_len,
                        self.head_dim,
                    )
                    kv_shape = (
                        self.batch_size,
                        self.num_heads_kv,
                        self.seq_len,
                        self.head_dim,
                    )
                yield (q_shape, kv_shape)
                return
            # 128 -> 32768
            for i in range(7, 15):
                seq_len = 2**i
                if self.decoding:
                    q_shape = (self.batch_size, self.num_heads_q, 1, self.head_dim)
                    kv_shape = (
                        self.batch_size,
                        self.num_heads_kv,
                        seq_len,
                        self.head_dim,
                    )
                else:
                    q_shape = (
                        self.batch_size,
                        self.num_heads_q,
                        seq_len,
                        self.head_dim,
                    )
                    kv_shape = (
                        self.batch_size,
                        self.num_heads_kv,
                        seq_len,
                        self.head_dim,
                    )
                yield (q_shape, kv_shape)

        shapes = get_ctx_vals()
        mask_mods = MOD_TYPES if self.mod_type == "all" else [self.mod_type]

        for (q_shape, kv_shape), mod_type in product(shapes, mask_mods):
            full_shape = self.get_full_shape(q_shape, kv_shape)
            block_mask, mask_mod_kwargs = self.generate_block_mask(
                mod_type, full_shape, self.sliding_window_size, self.prefix_length
            )
            score_mod = self.generate_score_mod(mod_type, full_shape)
            B, Hq, M, Hkv, N, D = full_shape
            if mod_type == "document_mask":
                q_shape_packed = (1, Hq, M * B, D)
                kv_shape_packed = (1, Hkv, N * B, D)
            else:
                q_shape_packed = q_shape
                kv_shape_packed = kv_shape

            make_q = partial(
                torch.rand,
                q_shape_packed,
                device=self.device,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
            )
            make_kv = partial(
                torch.rand,
                kv_shape_packed,
                device=self.device,
                dtype=self.dtype,
                requires_grad=self.requires_grad,
            )

            q = make_q()
            k = make_kv()
            v = make_kv()

            # Default kernel options for flex_attention
            kernel_options = self.get_kernel_options(mod_type, full_shape)
            yield (
                q,
                k,
                v,
                score_mod,
                block_mask,
                mod_type,
                kernel_options,
            )

    @register_x_val(label="(B, Hq, M, Hkv, N, D) | Mask Type")
    def get_x_val(self, example_inputs) -> str:
        """Return a detailed string representation of the input configuration."""
        q, k, v, score_mod, block_mask, mod_type, *_ = example_inputs
        B, H, S, D = q.shape

        shape = self.get_full_shape(q, v)
        return f"{str(shape):>30} | {mod_type:>15}"

    @register_benchmark(baseline=True)
    def eager(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        score_mod: Optional[_score_mod_signature],
        block_mask: Optional[BlockMask],
        mod_type: str,
        kernel_options: dict[str, Any],
    ) -> Optional[Callable]:
        """Baseline implementation using eager mode."""
        # Get shape information
        B, H, S, D = q.shape

        # Check if we should skip based on criteria
        should_skip = False
        if mod_type == "document_mask" and B * S >= 4096:
            should_skip = True
            print(
                f"Skipping eager for document_mask with batch*seq_len={B * S} >= 4096"
            )
        elif mod_type != "document_mask" and S > 8192:
            should_skip = True
            print(f"Skipping eager for {mod_type} with seq_len={S} > 8192")

        # If should skip, return a function that raises an exception
        if should_skip:
            # TODO  Figure out a way to skip the bigger shapes that will oom w/ eager
            pass

        # Otherwise return the normal function
        return lambda: flex_attention(
            q, k, v, block_mask=block_mask, kernel_options=kernel_options
        )

    @register_benchmark()
    def compiled(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        score_mod: Optional[_score_mod_signature],
        block_mask: Optional[BlockMask],
        mod_type: str,
        kernel_options: dict[str, Any],
    ) -> Optional[Callable]:
        """Compiled implementation using torch.compile."""
        mode = "default" if not self.max_autotune else "max-autotune-no-cudagraphs"
        compiled_fn = torch.compile(
            flex_attention, fullgraph=True, mode=mode, dynamic=self.dynamic
        )

        return lambda: compiled_fn(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            kernel_options=kernel_options,
        )

    @register_benchmark(enabled=HAS_FLASH_V3)
    def flash_v3(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        score_mod: Optional[_score_mod_signature],
        block_mask: Optional[BlockMask],
        mod_type: str,
        kernel_options: dict[str, Any],
    ) -> Optional[Callable]:
        """Flash Attention v3 implementation."""

        # Check dtype compatibility
        if q.dtype not in [torch.float16, torch.bfloat16]:
            print(f"[SKIP] Flash Attention v3 only supports fp16/bf16, got {q.dtype}")
            raise NotImplementedError(
                f"Flash Attention v3 only supports fp16/bf16, got {q.dtype}"
            )

        # Get shape info
        B, Hq, M, D = q.shape
        _, Hkv, N, _ = k.shape

        # Flash attention expects [B, S, H, D] layout, while flex_attention uses [B, H, S, D]
        q_flash = q.transpose(1, 2).contiguous()
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.transpose(1, 2).contiguous()

        # Prepare Flash Attention kwargs based on attention type
        FA_kwargs = {}

        if mod_type == "alibi":
            """
            h = torch.arange(Hq, dtype=torch.float32, device="cuda")
            alibi_slopes = torch.exp2(-((h + 1) * 8.0 / Hq))
            FA_kwargs["alibi_slopes"] = alibi_slopes

            TODO: Flash Attention v3 runs into:
            RuntimeError: This flash attention build does not support alibi.
            """
            return unsupported_fn

        # Build the appropriate Flash Attention callable
        if mod_type == "noop":
            fa_fn = partial(flash_attn_func, causal=False)
        elif mod_type in ["causal", "prefix_lm", "softcap"]:
            fa_fn = partial(flash_attn_func, causal=True)
        elif mod_type == "alibi":
            fa_fn = partial(flash_attn_func, causal=True, **FA_kwargs)
        elif mod_type == "sliding_window":
            fa_fn = partial(
                flash_attn_func, window_size=(self.sliding_window_size, 0), causal=True
            )
        elif mod_type == "document_mask":
            # Document mask requires special handling with varlen function
            print(f"[SKIP] Flash Attention v3 document_mask not implemented yet")
            raise NotImplementedError(
                "Flash Attention v3 document_mask not implemented yet"
            )
        else:
            # Unsupported attention types for Flash Attention
            print(
                f"[SKIP] Flash Attention v3 does not support {mod_type} attention type"
            )
            raise NotImplementedError(
                f"Flash Attention v3 does not support {mod_type} attention type"
            )

        def flash_v3_fn():
            out = fa_fn(q_flash, k_flash, v_flash)
            # Transpose back to [B, H, S, D]
            return out.transpose(1, 2).contiguous()

        return flash_v3_fn

    @register_benchmark(enabled=torch.backends.cudnn.is_available())
    def sdpa_cudnn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        score_mod: Optional[_score_mod_signature],
        block_mask: Optional[BlockMask],
        mod_type: str,
        kernel_options: dict[str, Any],
    ) -> Optional[Callable]:
        """SDPA with cuDNN backend."""
        supported_mods = ["noop", "causal"]
        if mod_type not in supported_mods:
            return unsupported_fn

        is_causal = mod_type == "causal"

        def sdpa_fn():
            try:
                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                    return sdpa(q, k, v, is_causal=is_causal)
            except RuntimeError as e:
                raise NotImplementedError(str(e))

        return sdpa_fn

    def get_grad_to_none(self, args) -> List[torch.Tensor]:
        """Return tensors whose gradients should be set to None between iterations."""
        q, k, v, *_ = args
        return [q, k, v]

    @register_metric(x_only=True)
    def flops(
        self, fn_name: str, example_inputs: Tuple, metrics: BenchmarkOperatorMetrics
    ) -> float:
        """Calculate the number of floating point operations for flex_attention."""
        q, k, v, score_mod, block_mask, mod_type, *_ = example_inputs
        BATCH, H, N_CTX, D_HEAD = q.shape
        full_shape = self.get_full_shape(q, v)
        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
        flops = 2 * flops_per_matmul
        if fn_name == "sdpa_cudnn":
            flops *= 0.5 if mod_type == "causal" else 1.0
        if self.mode in (BenchmarkMode.FWD, BenchmarkMode.FWD_NO_GRAD):
            if fn_name == "compiled":
                return self.calculate_flops(full_shape, block_mask)
            return flops
        if self.mode == BenchmarkMode.BWD:
            flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        elif self.mode == BenchmarkMode.FWD_BWD:
            flops *= 3.5  # 1.0(fwd) + 2.0(bwd) + 0.5(recompute)
        return flops

    @register_metric()
    def tbps(
        self, fn_name: str, example_inputs: Tuple, metrics: BenchmarkOperatorMetrics
    ) -> float:
        """Calculate memory bandwidth in TB/s for flex_attention."""
        q, k, v, score_mod, block_mask, mod_type, *_ = example_inputs

        def nbytes(t):
            """Calculate the number of bytes occupied by a tensor."""
            return t.numel() * t.element_size()

        total_bytes = nbytes(q) + nbytes(k) + nbytes(v)

        # Add output tensor size (same shape as q for attention output)
        output_bytes = nbytes(q)
        total_bytes += output_bytes

        # Convert to TB and calculate bandwidth
        tb = total_bytes / 1e12
        tbps = tb / metrics.latency * 1e3  # latency is in ms, convert to seconds
        return tbps

    ### -------------------- Utilities for benchmarking ------------------- ###
    @staticmethod
    def calculate_flops(shape: FullShape, block_mask: BlockMask) -> float:
        (B, Hq, M, Hkv, N, D) = shape
        qk_flops = M * N * D * 2
        softmax_flops = M * N * 2  # Not counting online softmax overhead
        o_flops = M * D * N * 2
        # Not counting split k overhead
        sparsity = block_mask.sparsity() / 100.0 if block_mask is not None else 0.0
        total_flops = B * Hq * (qk_flops + softmax_flops + o_flops) * (1 - sparsity)
        return total_flops

    @staticmethod
    def get_full_shape(
        q: Union[torch.Tensor, Tuple[int, int, int, int]],
        v: Union[torch.Tensor, Tuple[int, int, int, int]],
    ) -> FullShape:
        """
        Get the full shape information from q and v tensors or their shape tuples.

        Overloaded to accept either:
        1. Two torch.Tensor objects
        2. Two shape tuples (B, Hq, M, D) and (B, Hkv, N, D_V)

        Returns:
            FullShape(B, Hq, M, Hkv, N, D)
        """
        if isinstance(q, torch.Tensor) and isinstance(v, torch.Tensor):
            B, Hq, M, D = q.shape
            B_v, Hkv, N, D_V = v.shape
            assert B == B_v, "Batch sizes must match"
            return FullShape(B, Hq, M, Hkv, N, D)
        elif isinstance(q, tuple) and isinstance(v, tuple):
            assert len(q) == 4, "q shape must be (B, Hq, M, D)"
            assert len(v) == 4, "v shape must be (B, Hkv, N, D_V)"
            B, Hq, M, D = q
            B_v, Hkv, N, D_V = v
            assert B == B_v, "Batch sizes must match"
            return FullShape(B, Hq, M, Hkv, N, D)
        else:
            raise TypeError(
                "Inputs must be either two torch.Tensor objects or two shape tuples"
            )

    @staticmethod
    def get_kernel_options(attn_type: str, shape: FullShape):
        B, Hq, M, Hkv, N, D = shape
        is_decoding = M == 1
        # TODO add ways to specify TMA and warp spec
        # "ENABLE_TMA": True,
        kernel_opt_training_dict = {
            "noop": None,
            "causal": None,
            "rel": None,
            "head_bias": None,
            "alibi": None,
            "sliding_window": None,
            "document_mask": (
                {
                    "BLOCK_N": 32,
                    "BLOCK_M": 128,
                    "fwd_num_warps": 8,
                    "fwd_num_stages": 4,
                    "BLOCK_M1": 64,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 64,
                }
                if torch.cuda.get_device_capability() >= (8, 0) and D <= 128
                else None
            ),
            "prefix_lm": None,
            "softcap": None,
        }

        def get_default_split_k(B: int, H: int, Mk: int) -> int:
            num_SM = torch.cuda.get_device_properties("cuda").multi_processor_count
            """Heuristic for the number of splits from xformer"""
            bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
            split_k = num_SM // bh * 2  # Each SM should at least get one block.
            split_k = max(split_k, 1)

            return split_k

        kernel_opt_decoding_dict = {
            "noop": None,
            "causal": {"SPLIT_KV": get_default_split_k(B, Hkv, N) * 2},
            "rel": None,
            "head_bias": None,
            "alibi": {"SPLIT_KV": get_default_split_k(B, Hkv, N) * 2},
            "sliding_window": None,
            "document_mask": None,
            "prefix_lm": None,
            "softcap": {"SPLIT_KV": get_default_split_k(B, Hkv, N) * 2},
        }

        return (
            kernel_opt_decoding_dict[attn_type]
            if is_decoding
            else kernel_opt_training_dict[attn_type]
        )

    @staticmethod
    def generate_block_mask(
        attn_type: str, shape: FullShape, sliding_window_size: int, prefix_length: int
    ):
        B, Hq, M, Hkv, N, D = shape
        is_decoding = M == 1

        def generate_random_lengths(total_length: int, num_documents: int):
            # Initialize all lengths to 1 to ensure each document has at least one token
            lengths = [1] * num_documents
            remaining_length = total_length - num_documents

            # Randomly distribute the remaining length
            for _ in range(remaining_length):
                index = random.randint(0, num_documents - 1)
                lengths[index] += 1
            return lengths

        mask_mod_kwargs = {}

        assert attn_type != "document_mask" or not is_decoding
        if attn_type == "document_mask":
            random.seed(0)
            lengths = generate_random_lengths(N * B, B)
            mask_mod_kwargs = dict(offsets=length_to_offsets(lengths, "cuda"))

        mask_mod_dict = {
            "noop": None,
            "causal": causal_mask,
            "rel": None,
            "head_bias": None,
            "alibi": causal_mask,
            "sliding_window": generate_sliding_window(sliding_window_size),
            "document_mask": partial(generate_doc_mask_mod, mask_mod=causal_mask),
            "prefix_lm": generate_prefix_lm_mask(prefix_length),
            "softcap": causal_mask,
        }

        mask_mod = mask_mod_dict[attn_type]

        if mask_mod_kwargs:
            mask_mod = mask_mod(**mask_mod_kwargs)

        if is_decoding and mask_mod:
            cached_seq_len = torch.tensor(N // 2).to("cuda")

            def decoding_w_cached_seq_len(b, h, m, n):
                return mask_mod(b, h, m + cached_seq_len, n)

            new_mask_mod = decoding_w_cached_seq_len
        else:
            new_mask_mod = mask_mod

        mask_shape = (
            (1, 1, M, N) if attn_type != "document_mask" else (1, 1, M * B, N * B)
        )
        compiled_block_mask = torch.compile(create_block_mask)
        block_mask = (
            compiled_block_mask(new_mask_mod, *mask_shape, "cuda")
            if new_mask_mod
            else None
        )
        return block_mask, mask_mod_kwargs

    @staticmethod
    def generate_score_mod(attn_type: str, shape: FullShape) -> Callable | None:
        _, Hq, M, _, N, _ = shape
        is_decoding = M == 1

        def relative_bias(score, b, h, m, n):
            return score + (m - n)

        def head_bias(score, b, h, m, n):
            return score + 2 * h

        # TODO: AMD does not support tanh approximation w/flex attention for now
        # need to use the fast_tanhf assembly at
        # https://github.com/triton-lang/triton/blob/28e7587c5f67d8a744a50a0890fe4cd8431f5516/third_party/amd/language/hip/libdevice.py#L77
        approx = False if is_hip() else True

        function_dict = {
            "noop": None,
            "causal": None,
            "rel": relative_bias,
            "head_bias": head_bias,
            "alibi": generate_alibi_bias(Hq),
            "sliding_window": None,
            "document_mask": None,
            "prefix_lm": None,
            "softcap": generate_tanh_softcap(20, approx=approx),
        }

        score_mod = function_dict[attn_type]
        is_decoding = M == 1
        if is_decoding and score_mod:
            offset = torch.tensor(N // 2).to("cuda")

            def score_mod_w_offset(score, b, h, m, n):
                return score_mod(score, b, h, m + offset, n)

            new_score_mod = score_mod_w_offset
        else:
            new_score_mod = score_mod

        return new_score_mod
