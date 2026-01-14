"""Generalized Dot Product Attention (GDPA) Triton Bench Operator.
Example commands:
```
# Run optimized GDPA with different sparsity levels for PFFN
buck2 run @mode/opt //pytorch/tritonbench:run -- \
    --op generalized_dot_product_attention \
    --metrics latency \
    --batch 1152 \
    --max_seq_len 1000 \
    --dim 128 \
    --head 4 \
    --kv_len 256 \
    --only gdpa,gdpa_opt,gdpa_opt_sorted \
    --mode fwd_bwd \
    --sparsity 0.7

# Small inputs for quick testing
python run.py \
    --op gdpa \
    --metrics speedup,accuracy \
    --batch 1 \
    --max_seq_len 4 \
    --dim 16 \
    --head 1
```
"""

import argparse
import gc
from typing import Any, Callable, Generator, List, Optional

import torch
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    Mode,
    register_benchmark,
    register_metric,
    register_x_val,
)
from tritonbench.utils.triton_utils import has_tlx

if has_tlx():
    from tritonbench.operators.gdpa.gdpa_blackwell_tlx import gdpa_forward_tlx

from .gdpa import gdpa as gdpa_kernel
from .gdpa_utils import generate_jagged_data


def calculate_memory_size(jagged_q, jagged_k, jagged_v, real_output, run_fwd, run_bwd):
    def tensor_size(tensor):
        if tensor is not None:
            return tensor.numel() * tensor.element_size()
        return 0

    input_size = tensor_size(jagged_q) + tensor_size(jagged_k) + tensor_size(jagged_v)
    output_size = tensor_size(real_output)

    fwd_size = input_size + output_size
    backward_size = (
        fwd_size
        + tensor_size(jagged_q)  # Dq
        + tensor_size(jagged_k)  # Dk
        + tensor_size(jagged_v)  # Dv
    )
    total_size = 0
    if run_fwd:
        total_size += fwd_size
    if run_bwd:
        total_size += backward_size

    return total_size


def get_attn_config(config_name, dtype=torch.bfloat16):
    # default is pFFN general setting
    default_config = {
        "B": 1024,
        "max_M": 1000,
        "D": 384,
        "H": 3,
        "dense_q_len": 192,
        "sparsity": 0.5,
        "dense_q": False,
        "dff": None,
        "bias": False,
        "dtype": dtype,
        "fused_kv": False,
        "window_size": None,
        "broadcast_q": False,
        "activation": "fast_gelu",
    }
    # per event pffn, pma, self_attn share the same setting

    return default_config


all_configs = [
    "_".join([event_size, attn_type])
    for event_size in ["long_event", "short_event"]
    for attn_type in ["self_attn", "sliding_window", "pma"]
]


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch",
        default=1024,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--max_seq_len",
        default=1000,
        type=int,
        help="Max sequence length for Q",
    )
    parser.add_argument(
        "--dim",
        default=512,
        type=int,
        help="Query dimension",
    )
    parser.add_argument(
        "--head",
        default=4,
        type=int,
        help="Multi head number",
    )
    parser.add_argument(
        "--kv_len",
        default=None,
        type=int,
        help=f"Sequence length for K/V, if None, the tensor will be jagged and have the same length as Q",
    )
    parser.add_argument(
        "--activation",
        default="fast_gelu",
        type=str,
        help="Activations",
    )
    parser.add_argument(
        "--sparsity",
        default=0.5,
        type=float,
        help="Sparsity of the jagged tensor",
    )
    parser.add_argument(
        "--config",
        default=all_configs[0],
        type=str,
        help="list of configs to run, separated by comma",
    )
    args = parser.parse_args(args)
    return args


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args=extra_args)
        args = parse_args(self.extra_args)
        self.config_names = args.config.split(",")
        self.sparsity = args.sparsity
        self.batch = args.batch
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.head = args.head
        self.kv_len = args.kv_len

    @register_benchmark(enabled=has_tlx())
    def tlx_gdpa(
        self,
        _config_name,
        jagged_q,
        jagged_k,
        jagged_v,
        jagged_data,
        padded_data,
        activation,
    ):
        def _inner():
            # print(jagged_q.shape, jagged_k.shape, jagged_v.shape, jagged_data["q_offsets"].shape)

            # print("D|dim:", self.dim) # D/self.dim, assume H * dim in script is D
            # print("sparsity:", self.sparsity)
            real_output = gdpa_forward_tlx(
                query=jagged_q,
                key=jagged_k,
                value=jagged_v,
                query_offset=jagged_data["q_offsets"],
                key_offset=jagged_data["k_offsets"],
                output_offset=jagged_data["output_offsets"],
                max_seq_len_q=jagged_data["max_seq_len_q"],
                max_seq_len_kv=jagged_data["max_seq_len_k"],
                activation=activation,
                is_causal=False,
                broadcast_q=jagged_data["broadcast_q"],
                window_size=jagged_data["window_size"],
                bwd_opt_tech="tlx",
            )
            return real_output

        return _inner

    @register_benchmark()
    def gdpa(
        self,
        _config_name,
        jagged_q,
        jagged_k,
        jagged_v,
        jagged_data,
        padded_data,
        activation,
    ):
        def _inner():
            real_output = gdpa_kernel(
                query=jagged_q,
                key=jagged_k,
                value=jagged_v,
                query_offset=jagged_data["q_offsets"],
                key_offset=jagged_data["k_offsets"],
                output_offset=jagged_data["output_offsets"],
                max_seq_len_q=jagged_data["max_seq_len_q"],
                max_seq_len_kv=jagged_data["max_seq_len_k"],
                activation=activation,
                is_causal=False,
                broadcast_q=jagged_data["broadcast_q"],
                window_size=jagged_data["window_size"],
            )
            return real_output

        return _inner

    @register_benchmark(baseline=True)
    def eager_gdpa(
        self,
        _config_name,
        jagged_q,
        jagged_k,
        jagged_v,
        jagged_data,
        padded_data,
        activation,
    ):
        def _inner():
            B = len(jagged_data["q_offsets"]) - 1
            outputs = []
            q_offsets = jagged_data["q_offsets"]
            k_offsets = jagged_data["k_offsets"]

            for i in range(B):
                # Extract sequence i without .item() to avoid graph breaks
                q_start = q_offsets[i]
                q_end = q_offsets[i + 1]
                k_start = k_offsets[i]
                k_end = k_offsets[i + 1]

                # Get Q, K, V for this sequence using tensor indices
                q_seq = jagged_q[q_start:q_end]  # [seq_len_q, H, D]
                k_seq = jagged_k[k_start:k_end] if jagged_k is not None else None
                v_seq = jagged_v[k_start:k_end] if jagged_v is not None else None

                # Handle fused KV case
                if v_seq is None and k_seq is not None:
                    # Fused KV case
                    kv = k_seq
                    H = q_seq.shape[-2] if q_seq.dim() == 3 else 1
                    D = q_seq.shape[-1]
                    k_seq = kv[..., :D]
                    v_seq = kv[..., D:]

                # For attention, we need [batch, heads, seq, dim] format
                # Input is [seq, heads, dim], so reshape to [1, heads, seq, dim]
                if q_seq.dim() == 3:
                    q_seq = q_seq.transpose(0, 1).unsqueeze(0)  # [1, H, seq_q, D]
                    k_seq = k_seq.transpose(0, 1).unsqueeze(0)  # [1, H, seq_k, D]
                    v_seq = v_seq.transpose(0, 1).unsqueeze(0)  # [1, H, seq_k, D]
                else:
                    # 2D case: [seq, dim] -> [1, 1, seq, dim]
                    q_seq = q_seq.unsqueeze(0).unsqueeze(0)
                    k_seq = k_seq.unsqueeze(0).unsqueeze(0)
                    v_seq = v_seq.unsqueeze(0).unsqueeze(0)

                # Create attention mask for window size if needed
                attn_mask = None
                window_size = jagged_data.get("window_size", None)
                if window_size is not None and window_size > 0:
                    seq_len_q = q_seq.shape[2]
                    seq_len_k = k_seq.shape[2]
                    row_indices = torch.arange(seq_len_q, device=q_seq.device).view(
                        -1, 1
                    )
                    col_indices = torch.arange(seq_len_k, device=q_seq.device).view(
                        1, -1
                    )
                    window_mask = torch.abs(row_indices - col_indices) > window_size
                    attn_mask = torch.zeros(
                        seq_len_q, seq_len_k, device=q_seq.device, dtype=q_seq.dtype
                    )
                    attn_mask.masked_fill_(window_mask, float("-inf"))

                scale = 1.0
                scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * scale

                if attn_mask is not None:
                    scores = scores + attn_mask

                if activation == "gelu":
                    scores = torch.nn.functional.gelu(scores)
                elif activation == "fast_gelu":
                    scores = torch.nn.functional.gelu(scores, approximate="tanh")
                elif activation == "tanh":
                    scores = torch.tanh(scores)

                output = torch.matmul(scores, v_seq)

                # Reshape back: [1, H, seq_q, D] -> [seq_q, H, D]
                output = output.squeeze(0).transpose(0, 1)
                outputs.append(output)

            # Concatenate all outputs
            real_output = torch.cat(outputs, dim=0)

            return real_output

        return _inner

    @register_benchmark()
    def torch_compile_gdpa(
        self,
        _config_name,
        jagged_q,
        jagged_k,
        jagged_v,
        jagged_data,
        padded_data,
        activation,
    ):
        return torch.compile(
            self.eager_gdpa(
                _config_name,
                jagged_q,
                jagged_k,
                jagged_v,
                jagged_data,
                padded_data,
                activation,
            ),
            mode="max-autotune-no-cudagraphs",
        )

    @register_benchmark(enabled=True)
    def gdpa_opt(
        self,
        _config_name,
        jagged_q,
        jagged_k,
        jagged_v,
        jagged_data,
        padded_data,
        activation,
    ):
        def _inner():
            real_output = gdpa_kernel(
                query=jagged_q,
                key=jagged_k,
                value=jagged_v,
                query_offset=jagged_data["q_offsets"],
                key_offset=jagged_data["k_offsets"],
                output_offset=jagged_data["output_offsets"],
                max_seq_len_q=jagged_data["max_seq_len_q"],
                max_seq_len_kv=jagged_data["max_seq_len_k"],
                activation=activation,
                is_causal=False,
                broadcast_q=jagged_data["broadcast_q"],
                window_size=jagged_data["window_size"],
                # Currently, this combination provides the best performance.
                # For simplicity, we only test this in triton bench.
                # Will add more combinations in the future.
                enable_persistent=True,
                enable_tma=True,
                enable_ws=False,
                use_dq_atomic_add=True,
                bwd_opt_tech="base",
            )
            return real_output

        return _inner

    @register_benchmark(enabled=True)
    def gdpa_opt_sorted(
        self,
        _config_name,
        jagged_q,
        jagged_k,
        jagged_v,
        jagged_data,
        padded_data,
        activation,
    ):
        def _inner():
            real_output = gdpa_kernel(
                query=jagged_q,
                key=jagged_k,
                value=jagged_v,
                query_offset=jagged_data["q_offsets"],
                key_offset=jagged_data["k_offsets"],
                output_offset=jagged_data["output_offsets"],
                max_seq_len_q=jagged_data["max_seq_len_q"],
                max_seq_len_kv=jagged_data["max_seq_len_k"],
                activation=activation,
                is_causal=False,
                broadcast_q=jagged_data["broadcast_q"],
                window_size=jagged_data["window_size"],
                enable_persistent=True,
                enable_tma=True,
                enable_ws=False,
                use_dq_atomic_add=True,
                seq_index=jagged_data["seq_index"],
                bwd_opt_tech="base",
            )
            return real_output

        return _inner

    def get_input_iter(self) -> Generator:
        for config_name in self.config_names:
            config = get_attn_config(config_name, self.dtype)
            B = self.batch
            max_M = self.max_seq_len
            D = self.dim
            H = self.head
            dense_q_len = config["dense_q_len"]
            sparsity = self.sparsity
            dense_q = config["dense_q"]
            bias = config["bias"]
            dtype = config["dtype"]
            fused_kv = config["fused_kv"]
            dff = self.kv_len
            window_size = config["window_size"]
            broadcast_q = config["broadcast_q"]

            jagged_data = generate_jagged_data(
                B,
                max_M,
                D,
                H=H,
                sparsity=sparsity,
                dense_q=dense_q,
                bias=bias,
                dtype=dtype,
                dense_q_len=dense_q_len,
                broadcast_q=broadcast_q,
                dff=dff,
            )
            jagged_data["max_seq_len"] = max_M
            jagged_data["q_offsets"] = jagged_data["q_offsets"].to(torch.int32)
            jagged_data["k_offsets"] = jagged_data["k_offsets"].to(torch.int32)
            jagged_data["window_size"] = window_size
            jagged_data["seq_index"] = torch.argsort(
                jagged_data["num_objects_q"], descending=True
            ).contiguous()

            jagged_q, jagged_k, jagged_v = (
                jagged_data["q_weights"],
                jagged_data["k_weights"],
                jagged_data["v_weights"],
            )
            # padded_data = jagged_to_padded(
            #     **jagged_data
            # )  # fail when seq length is long
            head_dim = int(D / H)
            # Handle case where dense_q_len might be None
            q_len = dense_q_len if dense_q_len is not None else max_M
            padded_q = torch.randn(B, H, q_len, head_dim)
            padded_k = torch.randn(B, H, max_M, head_dim)
            padded_v = torch.randn(B, H, max_M, head_dim)
            padded_data = {
                "padded_q": padded_q,
                "padded_k": padded_k,
                "padded_v": padded_v,
            }
            if fused_kv:
                jagged_q = jagged_data["q_weights"]
                jagged_k = (
                    torch.cat(
                        [jagged_data["k_weights"], jagged_data["v_weights"]], dim=-1
                    )
                    .contiguous()
                    .detach()
                    .requires_grad_(True)
                )
                jagged_v = None

            self.jagged_data = jagged_data  # needed in backward
            self.padded_data = padded_data
            self.activation = config["activation"]
            yield (
                config_name,
                jagged_q,
                jagged_k,
                jagged_v,
                jagged_data,
                padded_data,
                self.activation,
            )
        return

    @register_x_val(label="config_name")
    def get_x_val(self, example_inputs):
        config_name, *_ = example_inputs
        return config_name

    @register_metric()
    def gbps(
        self, fn: Callable, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ):
        fwd_fn_lambda = getattr(self, fn._name)
        fwd_fn = fwd_fn_lambda(*example_inputs)
        # Calculate memory size
        config_name, jagged_q, jagged_k, jagged_v, jagged_data, padded_data = (
            example_inputs
        )
        fwd_output = fwd_fn()
        run_fwd = self.mode in [Mode.FWD, Mode.FWD_BWD, Mode.FWD_NO_GRAD]
        run_bwd = self.mode in [Mode.BWD, Mode.FWD_BWD]
        memory_size_bytes = calculate_memory_size(
            jagged_q, jagged_k, jagged_v, fwd_output, run_fwd, run_bwd
        )

        # Convert memory size to GB
        memory_size_gb = memory_size_bytes / (1024**3)

        # Calculate memory bandwidth in GB/s
        ms = metrics.latency
        memory_bandwidth_gb_per_sec = memory_size_gb / (ms * 1e-3)
        return memory_bandwidth_gb_per_sec

    @register_metric()
    def activation_mb(
        self, fn: Callable, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ):
        fwd_fn_lambda = getattr(self, fn._name)
        fwd_fn = fwd_fn_lambda(*example_inputs)

        # Calculate activation
        gc.collect()
        torch.cuda.empty_cache()
        MB = 1024.0 * 1024.0
        torch.cuda.reset_peak_memory_stats(device="cuda")
        memory_start = torch.cuda.max_memory_allocated(device="cuda") / MB

        output = fwd_fn()
        do = torch.rand_like(output) * 0.01
        output.backward(do, retain_graph=True)

        memory_end = torch.cuda.max_memory_allocated(device="cuda") / MB
        memory_used_MB = memory_end - memory_start

        return memory_used_MB

    def get_bwd_fn(self, fwd_fn: Callable) -> Callable:
        output = fwd_fn()
        if output.dim() == 4:
            # padded
            return lambda: output.backward(
                self.padded_data["padded_do"].contiguous(), retain_graph=True
            )
        do = torch.rand_like(output) * 0.01
        return lambda: output.backward(do, retain_graph=True)
