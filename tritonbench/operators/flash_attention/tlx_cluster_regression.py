from __future__ import annotations

import math
from dataclasses import dataclass

import torch


REFERENCE_TFLOPS = 1010.5
MIN_TFLOPS = 925.0
MIN_SNR_DB = 35.0


@dataclass(frozen=True)
class AttentionCase:
    dtype: torch.dtype
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    causal: bool


_DTYPES = (torch.float16, torch.bfloat16)
_MASKS = (False, True)
_D128_SEQUENCE_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)


def correctness_cases() -> tuple[AttentionCase, ...]:
    d128 = tuple(
        AttentionCase(dtype, 16384 // seq_len, 64, seq_len, 128, causal)
        for dtype in _DTYPES
        for causal in _MASKS
        for seq_len in _D128_SEQUENCE_LENGTHS
    )
    d64 = tuple(
        AttentionCase(dtype, 16, 64, 1024, 64, causal)
        for dtype in _DTYPES
        for causal in _MASKS
    )
    return d128 + d64


PERFORMANCE_CASE = AttentionCase(
    dtype=torch.bfloat16,
    batch=1,
    heads=64,
    seq_len=16384,
    head_dim=128,
    causal=False,
)


def attention_flops(case: AttentionCase) -> float:
    attended_pairs = (
        case.seq_len * (case.seq_len + 1) // 2 if case.causal else case.seq_len**2
    )
    return 4.0 * case.batch * case.heads * attended_pairs * case.head_dim


def snr_db(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    signal = torch.linalg.vector_norm(expected)
    noise = torch.linalg.vector_norm(actual - expected)
    if noise.item() == 0.0:
        return float("inf")
    if signal.item() == 0.0:
        return float("-inf")
    return float(20.0 * torch.log10(signal / noise))


def sample_indices(size: int) -> tuple[int, ...]:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    return tuple(dict.fromkeys((0, size // 2, size - 1)))


def sampled_fp32_attention(
    actual: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_indices = sample_indices(q.shape[0])
    head_indices = sample_indices(q.shape[1])
    query_indices = sample_indices(q.shape[2])
    query_positions = torch.tensor(query_indices, device=q.device)
    key_positions = torch.arange(k.shape[2], device=q.device)
    actual_rows = []
    reference_rows = []
    for batch in batch_indices:
        for head in head_indices:
            q_rows = q[batch, head, query_positions].float()
            k_rows = k[batch, head].float()
            v_rows = v[batch, head].float()
            scores = torch.matmul(q_rows, k_rows.transpose(0, 1)) * scale
            if causal:
                scores.masked_fill_(
                    key_positions[None, :] > query_positions[:, None],
                    float("-inf"),
                )
            reference_rows.append(torch.matmul(torch.softmax(scores, dim=-1), v_rows))
            actual_rows.append(actual[batch, head, query_positions].float())
    return torch.cat(actual_rows), torch.cat(reference_rows)


def assert_performance(measured_tflops: float) -> None:
    if not math.isfinite(measured_tflops) or measured_tflops <= 0.0:
        raise AssertionError(
            f"invalid throughput: {measured_tflops} TFLOP/s; expected a finite and positive value"
        )
    if measured_tflops < MIN_TFLOPS:
        raise AssertionError(
            f"PERF REGRESSION: {measured_tflops:.1f} TFLOP/s < "
            f"{MIN_TFLOPS:.1f} TFLOP/s floor "
            f"({REFERENCE_TFLOPS:.1f} TFLOP/s reference)"
        )
