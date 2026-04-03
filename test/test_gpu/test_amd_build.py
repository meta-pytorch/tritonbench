
"""
AMD/ROCm build verification test for tritonbench.

This test verifies that the tritonbench target builds and imports correctly
on AMD/ROCm, catching NVIDIA-only dependencies that may accidentally leak
into AMD builds (e.g. cute_dsl, tlx, nvidia-ml-py, hammer tritoncc deps).

Run locally:
    buck2 test @mode/opt-amd-gpu -c fbcode.rocm_arch=mi300 \
        //pytorch/tritonbench/test/test_gpu:test_amd_build
"""

import unittest


class TritonbenchAMDBuildTest(unittest.TestCase):
    def test_tritonbench_core_import(self) -> None:
        """Verify tritonbench core imports without NVIDIA-only deps."""
        from tritonbench.utils.triton_op import BenchmarkOperator  # noqa: F401

    def test_tritonbench_operators_import(self) -> None:
        """Verify tritonbench operator loader imports on AMD."""
        from tritonbench.operators import load_opbench_by_name  # noqa: F401

    def test_tritonbench_operator_list(self) -> None:
        """Verify operator listing works on AMD build."""
        from tritonbench.operators_collection import list_operators_by_collection

        ops = list_operators_by_collection(op_collection="default")
        self.assertIsInstance(ops, list)
        self.assertGreater(len(ops), 0, "Expected at least one operator to be listed")
