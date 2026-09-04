import gc
import math
import os
import statistics
import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F
import triton
from tritonbench.operators.flash_attention import operator as flash_attention
from tritonbench.operators.flash_attention.tlx_cluster_regression import (
    assert_performance,
    attention_flops,
    correctness_cases,
    MIN_SNR_DB,
    PERFORMANCE_CASE,
    REFERENCE_TFLOPS,
    sampled_fp32_attention,
    snr_db,
)
from tritonbench.utils.env_utils import is_hip_mi350


_REGRESSION_WORKER = "TRITONBENCH_TLX_AMD_FA_CLUSTER_REGRESSION_WORKER"
_TEST_CLASS = (
    "test.test_gpu.test_flash_attention_tlx_regression.TestTlxAmdFaClusterRegression"
)
_LLVM_OPT = "disable-machine-sink"


@unittest.skipUnless(is_hip_mi350(), "requires an MI350/gfx950 GPU")
class TestTlxAmdFaClusterRegression(unittest.TestCase):
    def _run_isolated(self):
        test_name = self._testMethodName
        if os.environ.get(_REGRESSION_WORKER) == test_name:
            self.assertEqual(os.environ.get("DISABLE_LLVM_OPT"), _LLVM_OPT)
            return False

        # LLVM command-line options are process-global, so compile the tuned
        # variants in a child process instead of affecting later GPU tests.
        env = os.environ.copy()
        env[_REGRESSION_WORKER] = test_name
        env["DISABLE_LLVM_OPT"] = _LLVM_OPT
        result = subprocess.run(
            [sys.executable, "-m", "unittest", f"{_TEST_CLASS}.{test_name}", "-v"],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
        if result.returncode != 0:
            self.fail(
                "isolated regression check failed:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        print(result.stdout)
        return True

    def _attention(self):
        attention = flash_attention._tlx_amd_fa_cluster
        if attention is None:
            self.fail("the meta-triton environment must provide tlx_amd_fa_cluster")
        return attention

    def _check_case(self, case):
        attention = self._attention()
        torch.manual_seed(42)
        shape = (case.batch, case.heads, case.seq_len, case.head_dim)
        q = torch.randn(shape, dtype=case.dtype, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        scale = 1.0 / math.sqrt(case.head_dim)
        out = attention(q, k, v, scale, case.causal)
        reference = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=case.causal,
            scale=scale,
        )

        self.assertTrue(torch.isfinite(out).all().item())
        measured_snr = snr_db(out, reference)
        self.assertGreaterEqual(
            measured_snr,
            MIN_SNR_DB,
            f"{case}: {measured_snr:.2f} dB SNR is below the {MIN_SNR_DB:.2f} dB floor",
        )
        actual_rows, reference_rows = sampled_fp32_attention(
            out,
            q,
            k,
            v,
            scale,
            case.causal,
        )
        torch.testing.assert_close(actual_rows, reference_rows, atol=2e-2, rtol=2e-2)

        del q, k, v, out, reference, actual_rows, reference_rows
        gc.collect()
        torch.cuda.empty_cache()

    def test_numerical_matrix(self):
        if self._run_isolated():
            return

        for case in correctness_cases():
            with self.subTest(case=case):
                self._check_case(case)

    def test_performance(self):
        if self._run_isolated():
            return

        attention = self._attention()
        case = PERFORMANCE_CASE
        torch.manual_seed(42)
        shape = (case.batch, case.heads, case.seq_len, case.head_dim)
        q = torch.randn(shape, dtype=case.dtype, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        scale = 1.0 / math.sqrt(case.head_dim)

        fn = lambda: attention(q, k, v, scale, case.causal)
        out = fn()
        reference = F.scaled_dot_product_attention(q, k, v, scale=scale)
        self.assertTrue(torch.isfinite(out).all().item())
        measured_snr = snr_db(out, reference)
        self.assertGreaterEqual(measured_snr, MIN_SNR_DB)
        actual_rows, reference_rows = sampled_fp32_attention(
            out, q, k, v, scale, case.causal
        )
        torch.testing.assert_close(actual_rows, reference_rows, atol=2e-2, rtol=2e-2)

        latencies_ms = [
            triton.testing.do_bench(fn, warmup=500, rep=500, return_mode="median")
            for _ in range(3)
        ]
        median_ms = statistics.median(latencies_ms)
        measured_tflops = attention_flops(case) * 1e-12 / (median_ms * 1e-3)
        print(
            f"\nTLX AMD FA cluster: windows_ms={latencies_ms}, median_ms={median_ms:.4f}, "
            f"throughput={measured_tflops:.1f} TFLOP/s, reference={REFERENCE_TFLOPS:.1f} TFLOP/s"
        )
        assert_performance(measured_tflops)
