import json
import logging
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from benchmarks.common import post_run_callback
from tritonbench.operators import load_opbench_by_name
from tritonbench.operators_collection import list_operators_by_collection
from tritonbench.utils.parser import get_parser


class TestTritonbenchCpu(unittest.TestCase):
    def _get_test_op(self, op_name="test_op", extra_args=[]):
        parser = get_parser(["--device", "cpu", "--op", op_name])
        args = ["--device", "cpu", "--op", op_name]
        if extra_args:
            args.extend(extra_args)
        tb_args, extra_args = parser.parse_known_args(args)
        Operator = load_opbench_by_name(tb_args.op)
        test_op = Operator(tb_args, extra_args)
        return test_op

    def test_cpu_swiglu_input_parameters(self):
        from tritonbench.operators.swiglu.operator import parse_op_args

        defaults = parse_op_args([])
        self.assertEqual(
            (
                defaults.batch,
                defaults.seq_len,
                defaults.hidden_size,
                defaults.intermediate_size,
            ),
            (4, None, 4096, 11008),
        )
        op = self._get_test_op(
            "swiglu",
            "--batch 2 --seq-len 3 --hidden-size 64 --intermediate-size 128".split(),
        )
        inputs = list(op.get_input_iter())
        self.assertEqual(len(inputs), 1)
        (x,) = inputs[0]
        self.assertEqual(tuple(x.shape), (2, 3, 64))
        self.assertEqual(tuple(op.baseline_op.gate_proj.weight.shape), (128, 64))
        self.assertEqual(tuple(op.baseline_op.down_proj.weight.shape), (64, 128))
        output = op.torch_swiglu(x)()
        self.assertEqual(tuple(output.shape), (2, 3, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_cpu_moe_input_parameters(self):
        for options, first, second in (
            ("--hidden-size 16", (1, 2, 16), (4, 8, 16)),
            ("--top-k 3 --hidden-size 16", (1, 3, 16), (4, 3, 16)),
            ("--tokens 7", (7, 8, 4096), None),
        ):
            with self.subTest(options=options):
                op = self._get_test_op("moe_fused_mul_sum", options.split())
                inputs = op.get_input_iter()
                self.assertEqual(tuple(next(inputs)[0].shape), first)
                self.assertEqual(tuple(next(inputs)[0].shape), first)
                if second is None:
                    self.assertEqual(list(inputs), [])
                else:
                    self.assertEqual(tuple(next(inputs)[0].shape), second)
        op = self._get_test_op(
            "moe_fused_mul_sum",
            "--precision fp32 --tokens 7 --top-k 3 --hidden-size 16 --num-experts 8".split(),
        )
        for values in op.get_input_iter():
            x, weights, ids, expert_map = values
            if expert_map is not None:
                self.assertEqual(expert_map.numel(), 8)
                self.assertLess(int(ids.max()), 8)
                x = x * (expert_map[ids] >= 0).unsqueeze(-1)
            torch.testing.assert_close(
                op.eager(*values)(), torch.einsum("tkh,tk->th", x, weights)
            )

    def test_cpu_rms_epsilon_parameter(self):
        options = ["--precision", "fp32", "--M", "2", "--H", "128"]
        self.assertEqual(self._get_test_op("rms_norm", options).eps, 1e-6)
        x = torch.full((2, 128), 1e-4)
        weight = torch.nn.Parameter(torch.ones(128), requires_grad=False)
        for eps in (1e-6, 1e-4):
            with (
                self.subTest(eps=eps),
                patch("torch.compile", side_effect=lambda module, **kwargs: module),
            ):
                op = self._get_test_op("rms_norm", options + ["--eps", str(eps)])
                expected = torch.full_like(x, 1e-4 / math.sqrt(1e-8 + eps))
                for provider in (
                    op.llama_rms,
                    op.torch_compile_rms,
                    op.triton_fused_rmsnorm,
                ):
                    torch.testing.assert_close(provider(128, x, weight)(), expected)

    def test_cpu_layer_norm(self):
        layer_norm_op = self._get_test_op(
            "layer_norm",
            extra_args=[
                "--only",
                "torch_layer_norm,torch_compile_layer_norm",
                "--metrics",
                "latency,accuracy",
                "--num-inputs",
                "1",
            ],
        )
        layer_norm_op.run()
        benchmark_output = layer_norm_op.output
        headers, table = benchmark_output._table()
        self.assertIn("torch_layer_norm-latency", headers)
        self.assertIn("torch_compile_layer_norm-latency", headers)
        self.assertIn("torch_compile_layer_norm-accuracy", headers)
        # accuracy metric should be True in the table
        self.assertEqual(True, table[0][-1])

    def test_cpu_metric_x_only_true(
        self,
    ):  # test x_only = True argument in register_metric()
        test_op = self._get_test_op()
        test_op.run()
        benchmark_operator_result = test_op.output
        headers, table = benchmark_operator_result._table()

        self.assertIn("test_metric", headers)  # x_only = True
        self.assertNotIn(
            "test_op-test_metric", headers
        )  # test_op-test_metric occurs only when x_only = False

    def test_cpu_metric_custom_label(self):
        test_op = self._get_test_op()
        test_op.run()
        benchmark_operator_result = test_op.output
        headers, table = benchmark_operator_result._table()

        self.assertTrue(
            ["new_op_label-" in header for header in headers]
        )  # custom benchmark label should be used in headers
        self.assertFalse(
            any(["test_op-" in header for header in headers])
        )  # default benchmark label should not be present in headers

    def test_ci_aggregate_metrics_are_all_or_nothing(self):
        test_op = self._get_test_op(
            extra_args=[
                "--metrics",
                "test_metric_per_benchmark",
            ]
        )
        test_op.run()
        benchmark_result = test_op.output
        for _, backend_metrics in benchmark_result.result:
            metrics = next(iter(backend_metrics.values()))
            metric_value = metrics.extra_metrics["test_metric_per_benchmark"]
            metrics.extra_metrics["test_metric_per_benchmark"] = metric_value[0]
        avg_key = "tritonbench_test_op_fwd[new_op_label]-test_metric_per_benchmark-avg"
        pass_key = "tritonbench_test_op_fwd-pass"

        complete_metrics = benchmark_result.userbenchmark_dict
        self.assertAlmostEqual(complete_metrics[avg_key], 23 / 3)
        self.assertEqual(complete_metrics[pass_key], 1)

        benchmark_result.metrics.append("kernel_source_hash")
        hash_metrics = benchmark_result.userbenchmark_dict
        self.assertAlmostEqual(hash_metrics[avg_key], 23 / 3)
        self.assertEqual(len(benchmark_result.result), 3)
        benchmark_result.metrics.remove("kernel_source_hash")

        last_result = benchmark_result.result.pop()
        with self.assertLogs("tritonbench.utils.triton_op", level="WARNING") as logs:
            truncated_metrics = benchmark_result.userbenchmark_dict
        self.assertNotIn(avg_key, truncated_metrics)
        self.assertEqual(truncated_metrics[pass_key], 0)
        self.assertIn("expected 3 input results, got 2", logs.output[0])

        benchmark_result.result.append(last_result)
        failed_metrics = next(iter(benchmark_result.result[1][1].values()))
        failed_metrics.error_msg = "test failure"
        partial_metrics = benchmark_result.userbenchmark_dict
        self.assertNotIn(avg_key, partial_metrics)
        self.assertEqual(partial_metrics[pass_key], 0)

    def test_ci_callback_preserves_partial_failure(self):
        reported_pass_key = "tritonbench_test_op_fwd-pass"
        ci_pass_key = "tritonbench_configured_test_op-pass"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_op.json"
            with open(output_file, "w") as file:
                json.dump({reported_pass_key: 0}, file)
            output_files = []

            post_run_callback(
                logging.getLogger(__name__),
                "test_group",
                "configured_test_op",
                output_file,
                output_files,
                disabled=False,
            )

            with open(output_file, "r") as file:
                metrics = json.load(file)
            self.assertEqual(metrics[ci_pass_key], 0)
            self.assertNotIn(reported_pass_key, metrics)
            self.assertEqual(output_files, [output_file])

    def test_cpu_list_operators_by_collection(self):
        all_ops = list_operators_by_collection(op_collection="all")
        self.assertTrue("aten.add.Tensor" in all_ops)
        self.assertTrue(len(all_ops) > 0)
        default_ops = list_operators_by_collection("default")
        self.assertTrue(len(default_ops) > 0)
        liger_ops = list_operators_by_collection("liger")
        self.assertTrue(len(liger_ops) > 0)
