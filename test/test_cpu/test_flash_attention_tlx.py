import pytest
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


def _make_fake_cuda_inputs(shape=(1, 1, 64, 64), devices=None):
    devices = devices or ("cuda:0",) * 3
    with FakeTensorMode():
        return tuple(
            torch.empty(shape, device=device, dtype=torch.bfloat16)
            for device in devices
        )


def test_tlx_amd_fa_cluster_uses_landed_forward_entrypoint(monkeypatch):
    backend = REGISTERED_BENCHMARKS["flash_attention"]["tlx_amd_fa_cluster"]
    assert backend.fwd_only

    q = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expected = torch.empty_like(q)
    calls = []

    def cluster_attention(actual_q, actual_k, actual_v, sm_scale, causal):
        calls.append((actual_q, actual_k, actual_v, sm_scale, causal))
        return expected

    monkeypatch.setattr(
        flash_attention,
        "_tlx_amd_fa_cluster",
        cluster_attention,
        raising=False,
    )
    monkeypatch.setattr(
        flash_attention,
        "_validate_tlx_amd_fa_cluster_inputs",
        lambda *args: None,
    )
    op = _make_operator()

    benchmark_fn = op.tlx_amd_fa_cluster(q, k, v)

    outputs = benchmark_fn()

    assert len(outputs) == 1
    assert outputs[0] is expected
    assert calls == [(q, k, v, 0.125, True)]


@pytest.mark.parametrize(
    ("q", "k", "v", "message"),
    [
        (
            torch.randn(1, 1, 64, 64, dtype=torch.bfloat16),
            torch.randn(1, 1, 128, 64, dtype=torch.bfloat16),
            torch.randn(1, 1, 128, 64, dtype=torch.bfloat16),
            "same shape",
        ),
        (
            torch.randn(1, 1, 64, 64, dtype=torch.float32),
            torch.randn(1, 1, 64, 64, dtype=torch.float32),
            torch.randn(1, 1, 64, 64, dtype=torch.float32),
            "float16 or bfloat16",
        ),
        (
            torch.randn(1, 1, 64, 64, dtype=torch.bfloat16),
            torch.randn(1, 1, 64, 64, dtype=torch.float16),
            torch.randn(1, 1, 64, 64, dtype=torch.float16),
            "matching dtypes",
        ),
        (
            torch.randn(1, 64, 64, dtype=torch.bfloat16),
            torch.randn(1, 64, 64, dtype=torch.bfloat16),
            torch.randn(1, 64, 64, dtype=torch.bfloat16),
            "rank-4",
        ),
    ],
)
def test_tlx_amd_fa_cluster_rejects_inputs_outside_landed_contract(
    monkeypatch, q, k, v, message
):
    def unexpected_launch(*args):
        pytest.fail("cluster attention launched for an unsupported input")

    monkeypatch.setattr(
        flash_attention,
        "_tlx_amd_fa_cluster",
        unexpected_launch,
    )
    with pytest.raises(ValueError, match=message):
        _make_operator().tlx_amd_fa_cluster(q, k, v)


def test_tlx_amd_fa_cluster_rejects_unsupported_head_dimension():
    q, k, v = _make_fake_cuda_inputs(shape=(1, 1, 64, 32))

    with pytest.raises(ValueError, match="head dimension 64 or 128"):
        flash_attention._validate_tlx_amd_fa_cluster_inputs(q, k, v)


def test_tlx_amd_fa_cluster_rejects_cpu_inputs():
    q = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with pytest.raises(ValueError, match="same GPU"):
        flash_attention._validate_tlx_amd_fa_cluster_inputs(q, k, v)


def test_tlx_amd_fa_cluster_rejects_inputs_on_different_cuda_devices():
    q, k, v = _make_fake_cuda_inputs(devices=("cuda:0", "cuda:1", "cuda:0"))

    with pytest.raises(ValueError, match="same GPU"):
        flash_attention._validate_tlx_amd_fa_cluster_inputs(q, k, v)


def test_tlx_amd_fa_cluster_rejects_nonpositive_dimensions():
    q, k, v = _make_fake_cuda_inputs(shape=(1, 1, 0, 64))

    with pytest.raises(ValueError, match="dimensions must be positive"):
        flash_attention._validate_tlx_amd_fa_cluster_inputs(q, k, v)


def test_tlx_amd_fa_cluster_rejects_zero_sequence_stride():
    q, k, _ = _make_fake_cuda_inputs()
    with FakeTensorMode():
        v = torch.empty((1, 1, 1, 64), device="cuda", dtype=torch.bfloat16).expand(
            1, 1, 64, 64
        )

    with pytest.raises(ValueError, match="positive sequence strides"):
        flash_attention._validate_tlx_amd_fa_cluster_inputs(q, k, v)


def test_tlx_amd_fa_cluster_rejects_negative_feature_stride():
    q, k, v = _make_fake_cuda_inputs()
    original_strides = k.stride()
    k.stride = lambda dim: -1 if dim == 3 else original_strides[dim]

    with pytest.raises(ValueError, match="nonnegative K feature"):
        flash_attention._validate_tlx_amd_fa_cluster_inputs(q, k, v)
