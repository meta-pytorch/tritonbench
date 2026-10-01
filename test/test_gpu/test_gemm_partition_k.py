import unittest

import torch
import triton
from tritonbench.operators.gemm.partition_k import (
    _matmul_partition_k,
    _reduce,
    get_mm_configs,
    matmul_partition_k,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class TestPartitionKMatmul(unittest.TestCase):
    def test_partition_boundaries(self):
        torch.manual_seed(0)
        # Exercise partial M/N tiles, uneven K splits, and empty partitions.
        m, n = 65, 35
        for k, pk in [(65, 1), (1024, 31), (17, 32), (2048, 32)]:
            a = torch.randint(-2, 3, (m, k), device="cuda").to(torch.float16)
            b = torch.randint(-2, 3, (k, n), device="cuda").to(torch.float16)
            expected = a.float() @ b.float()
            for config in get_mm_configs():
                with self.subTest(k=k, pk=pk, config=str(config)):
                    c_buf = torch.full(
                        (m, n, pk), float("nan"), device="cuda", dtype=torch.float32
                    )
                    grid = (
                        triton.cdiv(m, config.kwargs["BLOCK_SIZE_M"])
                        * triton.cdiv(n, config.kwargs["BLOCK_SIZE_N"])
                        * pk,
                    )
                    _matmul_partition_k.fn[grid](
                        a,
                        b,
                        c_buf,
                        m,
                        n,
                        k,
                        pk,
                        triton.cdiv(k, pk),
                        *a.stride(),
                        *b.stride(),
                        *c_buf.stride(),
                        **config.kwargs,
                        num_warps=config.num_warps,
                        num_stages=config.num_stages,
                    )
                    torch.testing.assert_close(
                        c_buf.sum(dim=2), expected, atol=0, rtol=0
                    )
                    c = torch.empty_like(expected)
                    _reduce[(triton.cdiv(m, 32) * triton.cdiv(n, 32),)](
                        c,
                        c_buf,
                        m,
                        n,
                        *c.stride(),
                        *c_buf.stride(),
                        pk,
                        32,
                        32,
                    )
                    torch.testing.assert_close(c, expected, atol=0, rtol=0)

    def test_forward_backward(self):
        torch.manual_seed(0)
        for triton_reduce in (False, True):
            with self.subTest(triton_reduce=triton_reduce):
                a = torch.randn(
                    (33, 65), device="cuda", dtype=torch.float16, requires_grad=True
                )
                b = torch.randn(
                    (65, 35), device="cuda", dtype=torch.float16, requires_grad=True
                )
                actual = matmul_partition_k(a, b, triton_reduce=triton_reduce)
                expected = a @ b
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
                grad = torch.randn_like(actual)
                actual_grads = torch.autograd.grad(actual, (a, b), grad)
                expected_grads = torch.autograd.grad(expected, (a, b), grad)
                torch.testing.assert_close(
                    actual_grads, expected_grads, atol=1e-2, rtol=1e-2
                )
