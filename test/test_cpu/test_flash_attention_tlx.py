import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from tritonbench.operators.flash_attention import operator as flash_attention
from tritonbench.operators.flash_attention.tlx_cluster_regression import (
    assert_performance,
    attention_flops,
    correctness_cases,
    MIN_SNR_DB,
    MIN_TFLOPS,
    PERFORMANCE_CASE,
    sample_indices,
    sampled_fp32_attention,
    snr_db,
)
from tritonbench.utils.triton_op import REGISTERED_BENCHMARKS


def _make_operator():
    op = flash_attention.Operator.__new__(flash_attention.Operator)
    op.sm_scale = 0.125
    op.causal = True
    op.optims = {}
    return op


def _make_fake_cuda_inputs():
    return tuple(
        torch.empty(
            (1, 1, 64, 64),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    )


def test_tlx_amd_fa_cluster_is_registered_forward_only():
    backend = REGISTERED_BENCHMARKS["flash_attention"]["tlx_amd_fa_cluster"]

    assert backend.fwd_only


def test_tlx_amd_fa_cluster_validates_before_landed_entrypoint(monkeypatch):
    events = []

    def validate(actual_q, actual_k, actual_v):
        events.append(("validate", actual_q, actual_k, actual_v))

    def cluster_attention(actual_q, actual_k, actual_v, sm_scale, causal):
        events.append(("attention", actual_q, actual_k, actual_v, sm_scale, causal))
        return expected

    monkeypatch.setattr(
        flash_attention,
        "_validate_tlx_amd_fa_cluster_inputs",
        validate,
    )
    monkeypatch.setattr(
        flash_attention,
        "_tlx_amd_fa_cluster",
        cluster_attention,
    )

    with FakeTensorMode():
        q, k, v = _make_fake_cuda_inputs()
        expected = torch.empty_like(q)
        benchmark_fn = _make_operator().tlx_amd_fa_cluster(q, k, v)
        outputs = benchmark_fn()

    assert len(outputs) == 1
    assert outputs[0] is expected
    assert [event[0] for event in events] == ["validate", "attention"]
    assert all(actual is wanted for actual, wanted in zip(events[0][1:], (q, k, v)))
    assert all(actual is wanted for actual, wanted in zip(events[1][1:4], (q, k, v)))
    assert events[1][4:] == (0.125, True)


def test_tlx_amd_fa_cluster_regression_matrix():
    cases = correctness_cases()
    d128 = [case for case in cases if case.head_dim == 128]
    d64 = [case for case in cases if case.head_dim == 64]

    assert len(cases) == 28
    assert {
        (case.dtype, case.batch, case.heads, case.seq_len, case.head_dim, case.causal)
        for case in d128
    } == {
        (dtype, 16384 // seq_len, 64, seq_len, 128, causal)
        for dtype in (torch.float16, torch.bfloat16)
        for causal in (False, True)
        for seq_len in (512, 1024, 2048, 4096, 8192, 16384)
    }
    assert {
        (case.dtype, case.batch, case.heads, case.seq_len, case.head_dim, case.causal)
        for case in d64
    } == {
        (dtype, 16, 64, 1024, 64, causal)
        for dtype in (torch.float16, torch.bfloat16)
        for causal in (False, True)
    }
    assert PERFORMANCE_CASE in d128
    assert PERFORMANCE_CASE.dtype == torch.bfloat16
    assert not PERFORMANCE_CASE.causal
    assert (
        PERFORMANCE_CASE.batch,
        PERFORMANCE_CASE.heads,
        PERFORMANCE_CASE.seq_len,
        PERFORMANCE_CASE.head_dim,
    ) == (1, 64, 16384, 128)


def test_tlx_amd_fa_cluster_regression_helpers():
    expected = torch.ones(16)
    perturbed = expected.clone()
    perturbed[0] = 2.0

    assert math.isinf(snr_db(expected, expected))
    assert snr_db(perturbed, expected) < MIN_SNR_DB
    assert sample_indices(1) == (0,)
    assert sample_indices(8) == (0, 4, 7)
    assert attention_flops(PERFORMANCE_CASE) == 4 * 1 * 64 * 16384 * 16384 * 128
    causal_case = PERFORMANCE_CASE.__class__(
        dtype=torch.bfloat16,
        batch=2,
        heads=4,
        seq_len=512,
        head_dim=128,
        causal=True,
    )
    assert attention_flops(causal_case) == 2 * 2 * 4 * 512 * 513 * 128

    with pytest.raises(
        AssertionError, match=r"924\.9 TFLOP/s.*925\.0 TFLOP/s.*1010\.5 TFLOP/s"
    ):
        assert_performance(924.9)
    assert MIN_TFLOPS == 925.0
    assert_performance(MIN_TFLOPS)

    for invalid_tflops in (float("nan"), float("inf"), float("-inf"), 0.0):
        with pytest.raises(AssertionError, match="finite and positive"):
            assert_performance(invalid_tflops)


@pytest.mark.parametrize("causal", [False, True])
def test_tlx_amd_fa_cluster_sampled_fp32_reference_matches_full_attention(causal):
    torch.manual_seed(17)
    q = torch.randn((2, 2, 5, 3), dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 0.375
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        positions = torch.arange(q.shape[2])
        scores.masked_fill_(positions[None, :] > positions[:, None], float("-inf"))
    full_reference = torch.matmul(torch.softmax(scores, dim=-1), v)

    actual_rows, reference_rows = sampled_fp32_attention(
        full_reference,
        q,
        k,
        v,
        scale,
        causal,
    )

    torch.testing.assert_close(actual_rows, reference_rows)
