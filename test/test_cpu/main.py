import unittest

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

    def test_cpu_list_operators_by_collection(self):
        all_ops = list_operators_by_collection(op_collection="all")
        self.assertTrue("aten.add.Tensor" in all_ops)
        self.assertTrue(len(all_ops) > 0)
        default_ops = list_operators_by_collection("default")
        self.assertTrue(len(default_ops) > 0)
        liger_ops = list_operators_by_collection("liger")
        self.assertTrue(len(liger_ops) > 0)
