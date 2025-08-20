import torch
import triton
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
)

num_q_heads, num_kv_heads = 32, 32
hidden_dim = 128
dtype = torch.bfloat16


def prepare_input(batch_size, seq_length):
    rotary_emb = LlamaRotaryEmbedding(hidden_dim, device="cuda")
    q = torch.randn(
        (batch_size, num_q_heads, seq_length, hidden_dim),
        device="cuda",
        requires_grad=True,
        dtype=dtype,
    )
    k = torch.randn(
        (batch_size, num_kv_heads, seq_length, hidden_dim),
        device="cuda",
        requires_grad=True,
        dtype=dtype,
    )
    dq, dk = (
        torch.randn_like(q, device="cuda", dtype=dtype),
        torch.randn_like(k, device="cuda"),
    )
    pos_ids = torch.arange(seq_length, device="cuda", dtype=torch.long).unsqueeze(0)
    cos, sin = rotary_emb(k, pos_ids)
    return q, k, cos, sin, pos_ids


def liger_rotary_pos_emb_kernel(batch_size, seq_length):
    q, k, cos, sin, pos_ids = prepare_input(batch_size, seq_length)
    return lambda: liger_rotary_pos_emb(q, k, cos, sin, pos_ids)


def inductor_rotary_pos_emb_full_op(batch_size, seq_length):
    q, k, cos, sin, pos_ids = prepare_input(batch_size, seq_length)
    get_rotary_embedding = LlamaRotaryEmbedding(hidden_dim, device="cuda")
    cos, sin = get_rotary_embedding(k, pos_ids)
    compiled_func = torch.compile(
        apply_rotary_pos_emb, mode="max-autotune-no-cudagraphs"
    )
    return lambda: compiled_func(q, k, cos, sin, pos_ids)


compiled_fn = inductor_rotary_pos_emb_full_op(2, 1)
liger_kernel = liger_rotary_pos_emb_kernel(2, 1)


compiler_latency = triton.testing.do_bench_cudagraph(compiled_fn)
liger_latency = triton.testing.do_bench_cudagraph(liger_kernel)

print("compiler_latency:", compiler_latency, "liger_latency:", liger_latency)
