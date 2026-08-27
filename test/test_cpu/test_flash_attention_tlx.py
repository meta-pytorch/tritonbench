import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from tritonbench.operators.flash_attention import operator as flash_attention
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
