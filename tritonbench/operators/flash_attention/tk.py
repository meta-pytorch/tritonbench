from typing import Tuple

import torch
from tritonbench.utils.path_utils import ensure_build_subdir_on_sys_path

with ensure_build_subdir_on_sys_path():
    import thunderkittens as tk


class _TKAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal) -> torch.Tensor:
        ctx.causal = causal
        o, l_vec = tk.mha_forward(q, k, v, causal)
        saved_tensors = [q, k, v, o, l_vec]
        ctx.save_for_backward(*saved_tensors)
        return o

    @staticmethod
    def backward(ctx, do) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        causal = ctx.causal
        q, k, v, o, l_vec = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq, dk, dv = tk.mha_backward(q, k, v, o, l_vec, do, causal)
        return dq, dk, dv, None


def tk_attn(q, k, v, causal) -> torch.Tensor:
    o = _TKAttn.apply(q, k, v, causal)
    return o
