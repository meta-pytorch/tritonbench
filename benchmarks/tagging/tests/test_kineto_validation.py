# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import gzip
import json
import os
import tempfile
import unittest

from pytorch.tritonbench.benchmarks.tagging.ast_analyzer import build_backend_callees
from pytorch.tritonbench.benchmarks.tagging.run import (
    extract_kernel_names,
    parse_kineto_trace,
)


SAMPLE_TRACE_PATH = os.path.join(os.path.dirname(__file__), "sample_kineto_trace.json")


class ExtractKernelNamesTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        with open(SAMPLE_TRACE_PATH, "r") as f:
            self.trace_data = json.load(f)

    def test_extracts_kernel_events(self) -> None:
        names = extract_kernel_names(self.trace_data)
        self.assertIn("triton_gemm_kernel_0d1d2d", names)
        self.assertIn("sm90_xmma_gemm_f16f16_f32", names)

    def test_deduplicates_kernel_names(self) -> None:
        names = extract_kernel_names(self.trace_data)
        self.assertEqual(len(names), len(set(names)))

    def test_excludes_non_kernel_categories(self) -> None:
        names = extract_kernel_names(self.trace_data)
        self.assertNotIn("aten::mm", names)

    def test_excludes_memcpy_memset_events(self) -> None:
        names = extract_kernel_names(self.trace_data)
        self.assertNotIn("Memcpy HtoD", names)
        self.assertNotIn("Memcpy DtoD", names)

    def test_includes_elementwise_kernel(self) -> None:
        names = extract_kernel_names(self.trace_data)
        self.assertTrue(any("vectorized_elementwise_kernel" in n for n in names))

    def test_returns_sorted(self) -> None:
        names = extract_kernel_names(self.trace_data)
        self.assertEqual(names, sorted(names))

    def test_empty_trace(self) -> None:
        names = extract_kernel_names({"traceEvents": []})
        self.assertEqual(names, [])

    def test_no_kernel_events(self) -> None:
        trace = {
            "traceEvents": [
                {"ph": "X", "cat": "cpu_op", "name": "aten::mm"},
            ]
        }
        names = extract_kernel_names(trace)
        self.assertEqual(names, [])

    def test_missing_trace_events_key(self) -> None:
        names = extract_kernel_names({})
        self.assertEqual(names, [])

    def test_non_dict_events_skipped(self) -> None:
        trace = {
            "traceEvents": [
                "not a dict",
                {"ph": "X", "cat": "kernel", "name": "my_kernel"},
            ]
        }
        names = extract_kernel_names(trace)
        self.assertEqual(names, ["my_kernel"])


class ParseKinetoTraceFromJsonTest(unittest.TestCase):
    def test_parse_json_file(self) -> None:
        names = parse_kineto_trace(SAMPLE_TRACE_PATH)
        self.assertIn("triton_gemm_kernel_0d1d2d", names)
        self.assertIn("sm90_xmma_gemm_f16f16_f32", names)

    def test_parse_gzip_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            gz_path = f.name
        try:
            with open(SAMPLE_TRACE_PATH, "r") as src:
                data = src.read()
            with gzip.open(gz_path, "wt") as gz:
                gz.write(data)
            names = parse_kineto_trace(gz_path)
            self.assertIn("triton_gemm_kernel_0d1d2d", names)
        finally:
            os.unlink(gz_path)

    def test_fallback_to_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "trace.json")
            with open(SAMPLE_TRACE_PATH, "r") as src:
                data = src.read()
            with open(dest, "w") as f:
                f.write(data)
            names = parse_kineto_trace("/nonexistent/path.json", output_dir=tmpdir)
            self.assertIn("triton_gemm_kernel_0d1d2d", names)

    def test_fallback_to_output_dir_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "trace.json.gz")
            with open(SAMPLE_TRACE_PATH, "r") as src:
                data = src.read()
            with gzip.open(dest, "wt") as gz:
                gz.write(data)
            names = parse_kineto_trace("/nonexistent/path.json", output_dir=tmpdir)
            self.assertIn("sm90_xmma_gemm_f16f16_f32", names)

    def test_nonexistent_path_returns_empty(self) -> None:
        names = parse_kineto_trace("/nonexistent/path.json")
        self.assertEqual(names, [])

    def test_nonexistent_path_and_dir_returns_empty(self) -> None:
        names = parse_kineto_trace(
            "/nonexistent/path.json", output_dir="/nonexistent/dir"
        )
        self.assertEqual(names, [])

    def test_url_without_output_dir_returns_empty(self) -> None:
        names = parse_kineto_trace(
            "https://www.internalfb.com/intern/perfdoctor/trace_view?filepath=foo"
        )
        self.assertEqual(names, [])


class IfExpKernelResolutionTest(unittest.TestCase):
    """Static analysis must resolve a kernel launched through a ternary-bound
    variable (``fn = A if cond else B; fn[grid](...)``) to the real kernel
    name(s), not the launcher-variable placeholder ``fn``. This mirrors
    ``mslk.gemm.triton.grouped_gemm._grouped_gemm`` (fp8_gemm_rowwise_grouped).
    """

    SOURCE = (
        "def _kernel_ws():\n"
        "    pass\n"
        "\n"
        "def _kernel_plain():\n"
        "    pass\n"
        "\n"
        "class Operator:\n"
        "    def my_backend(self):\n"
        "        fn = _kernel_ws if self.use_ws else _kernel_plain\n"
        "        fn[grid](a, b)\n"
    )

    def _callee_simple_names(self):
        backends = build_backend_callees(
            source=self.SOURCE,
            filename="operator.py",
            module_name="tritonbench.operators.test.operator",
            backends=["my_backend"],
        )
        # Recorded callees may be module-qualified; compare on the simple name,
        # which is what is matched against kineto kernel names.
        return {c.rsplit(".", 1)[-1] for c in backends["my_backend"]}

    def test_resolves_both_ternary_branches(self) -> None:
        names = self._callee_simple_names()
        self.assertIn("_kernel_ws", names, f"actual={names}")
        self.assertIn("_kernel_plain", names, f"actual={names}")

    def test_does_not_record_launcher_variable(self) -> None:
        # Without IfExp handling the analyzer recorded the bare ``fn`` name.
        self.assertNotIn("fn", self._callee_simple_names())
