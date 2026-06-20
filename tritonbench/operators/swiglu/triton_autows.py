# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_tile_body(
    gate_desc,
    up_desc,
    out_desc,
    tile_id,
    num_n_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tile_id // num_n_tiles
    pid_n = tile_id % num_n_tiles
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N

    gate = gate_desc.load([offs_m, offs_n])
    up = up_desc.load([offs_m, offs_n])
    out = _silu(gate.to(tl.float32)).to(up.dtype) * up
    out_desc.store([offs_m, offs_n], out)


# Triton TR001: this coverage backend intentionally keeps a fixed config while
# proper runtime gates for Meta Triton + Blackwell + AutoWS are validated.
@triton.jit
def _swiglu_fwd_ws_kernel(  # noqa: TR001
    gate_desc,
    up_desc,
    out_desc,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_m_tiles = tl.cdiv(n_rows, BLOCK_M)
    num_n_tiles = tl.cdiv(n_cols, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        warp_specialize=True,
        num_stages=2,
    ):
        _swiglu_tile_body(
            gate_desc,
            up_desc,
            out_desc,
            tile_id,
            num_n_tiles,
            BLOCK_M,
            BLOCK_N,
        )


def triton_autows_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    block_m: int = 8,
    block_n: int = 1024,
    num_warps: int = 4,
) -> torch.Tensor:
    """AutoWS fused SwiGLU activation: ``silu(gate).to(up.dtype) * up``."""
    if not hasattr(triton, "knobs"):
        raise NotImplementedError("triton_autows_swiglu requires Meta Triton")
    if gate.shape != up.shape:
        raise NotImplementedError("triton_autows_swiglu requires matching inputs")

    original_shape = gate.shape
    n_cols = original_shape[-1]
    gate_2d = gate.reshape(-1, n_cols).contiguous()
    up_2d = up.reshape(-1, n_cols).contiguous()
    n_rows = gate_2d.shape[0]
    out_2d = torch.empty_like(gate_2d)

    block_n = min(block_n, triton.next_power_of_2(n_cols))
    block_m = min(block_m, max(1, triton.next_power_of_2(n_rows)))

    gate_desc = TensorDescriptor(
        gate_2d,
        shape=gate_2d.shape,
        strides=gate_2d.stride(),
        block_shape=[block_m, block_n],
    )
    up_desc = TensorDescriptor(
        up_2d,
        shape=up_2d.shape,
        strides=up_2d.stride(),
        block_shape=[block_m, block_n],
    )
    out_desc = TensorDescriptor(
        out_2d,
        shape=out_2d.shape,
        strides=out_2d.stride(),
        block_shape=[block_m, block_n],
    )

    def alloc_fn(size: int, _alignment: int, _stream) -> torch.Tensor:
        return torch.empty(size, device=gate.device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    num_sms = torch.cuda.get_device_properties(gate.device).multi_processor_count

    assert triton.knobs.nvidia.use_meta_ws, (
        "triton_autows_swiglu requires TRITON_USE_META_WS=1"
    )
    assert triton.knobs.nvidia.disable_wsbarrier_reorder, (
        "triton_autows_swiglu currently requires TRITON_DISABLE_WSBARRIER_REORDER=1"
    )
    # TODO: Tune max registers across partitions.
    _swiglu_fwd_ws_kernel[(num_sms,)](
        gate_desc,
        up_desc,
        out_desc,
        n_rows,
        n_cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_SMS=num_sms,
        num_warps=num_warps,
    )
    return out_2d.reshape(original_shape)
