# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

import torch
from parameterized import parameterized


# =============================================================================
# Flash Attention: Common utilities (mirrors test_correctness.py pattern)
# =============================================================================


class FlashAttention:
    """Common utilities for Blackwell Flash Attention kernel correctness tests."""

    # (Z, H, N_CTX, HEAD_DIM)
    SHAPES = [(2, 4, 256, 64), (2, 4, 256, 128)]

    @staticmethod
    def create_inputs(Z, H, N_CTX, HEAD_DIM, dtype=torch.bfloat16):
        torch.manual_seed(20)
        q = torch.empty(
            (Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=dtype
        ).normal_(mean=0.0, std=0.5)
        k = torch.empty(
            (Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=dtype
        ).normal_(mean=0.0, std=0.5)
        v = torch.empty(
            (Z, H, N_CTX, HEAD_DIM), device="cuda", dtype=dtype
        ).normal_(mean=0.0, std=0.5)
        return q, k, v

    @staticmethod
    def get_reference(q, k, v, sm_scale, causal):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale, is_causal=causal
        )

    @staticmethod
    def run_test(attention_fn, causal, dtype=torch.bfloat16, shapes=None):
        if shapes is None:
            shapes = FlashAttention.SHAPES
        for Z, H, N_CTX, HEAD_DIM in shapes:
            sm_scale = 1.0 / (HEAD_DIM**0.5)
            q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM, dtype)
            ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
            tri_out = attention_fn(q, k, v, causal, sm_scale, "ws_persistent")
            # Triton kernels use exp2 approximation, so tolerance is relaxed
            torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)

    @staticmethod
    def run_backward_test(attention_fn, causal, dtype=torch.bfloat16, shapes=None):
        if shapes is None:
            shapes = FlashAttention.SHAPES
        for Z, H, N_CTX, HEAD_DIM in shapes:
            sm_scale = 1.0 / (HEAD_DIM**0.5)
            q, k, v = FlashAttention.create_inputs(Z, H, N_CTX, HEAD_DIM, dtype)
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            # Triton backward
            tri_out = attention_fn(q, k, v, causal, sm_scale, "ws_persistent")
            tri_out.sum().backward()
            tri_dq, tri_dk, tri_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

            # Reset grads
            q.grad, k.grad, v.grad = None, None, None

            # Reference backward
            ref_out = FlashAttention.get_reference(q, k, v, sm_scale, causal)
            ref_out.sum().backward()

            torch.testing.assert_close(tri_dq, q.grad, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(tri_dk, k.grad, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(tri_dv, v.grad, atol=1e-2, rtol=1e-2)


# =============================================================================
# Blackwell FA2 autoWS (fwd + bwd)
# =============================================================================

@unittest.skip("TODO: RuntimeError on autoWS")
class TestBlackwellTritonFusedAttention(unittest.TestCase):
    """Tests for blackwell_triton_fused_attention.py (autoWS variant)."""

    @classmethod
    def setUpClass(cls):
        from tritonbench.kernels.blackwell_triton_fused_attention import (
            attention_opt,
        )

        cls.attention_opt = staticmethod(attention_opt)

    @parameterized.expand(
        [
            ("non_causal_bf16", False, torch.bfloat16),
            ("causal_bf16", True, torch.bfloat16),
            ("non_causal_fp16", False, torch.float16),
        ]
    )
    def test_forward(self, _name, causal, dtype):
        FlashAttention.run_test(self.attention_opt, causal=causal, dtype=dtype)

    @parameterized.expand(
        [
            ("non_causal_bf16", False, torch.bfloat16),
            ("causal_bf16", True, torch.bfloat16),
            ("non_causal_fp16", False, torch.float16),
        ]
    )
    def test_backward(self, _name, causal, dtype):
        FlashAttention.run_backward_test(self.attention_opt, causal=causal, dtype=dtype)


# =============================================================================
# Blackwell FA2 with data partition (fwd only)
# =============================================================================


class TestBlackwellTritonFusedAttentionDP(unittest.TestCase):
    """Tests for blackwell_triton_fused_attention_dp.py (data partition variant)."""

    @classmethod
    def setUpClass(cls):
        from tritonbench.kernels.blackwell_triton_fused_attention_dp import (
            attention_opt,
        )

        cls.attention_opt = staticmethod(attention_opt)

    @parameterized.expand(
        [
            ("non_causal_bf16", False, torch.bfloat16),
            ("causal_bf16", True, torch.bfloat16),
            ("non_causal_fp16", False, torch.float16),
        ]
    )
    def test_forward(self, _name, causal, dtype):
        FlashAttention.run_test(self.attention_opt, causal=causal, dtype=dtype)


if __name__ == "__main__":
    unittest.main()
