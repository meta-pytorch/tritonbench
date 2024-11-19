import argparse
import logging
import unittest

from typing import List, Optional

import yaml

from tritonbench.operators import load_opbench_by_name
from tritonbench.operators_collection import list_operators_by_collection

from tritonbench.utils.parser import get_parser
from tritonbench.utils.triton_op import IS_FBCODE

if IS_FBCODE:
    import importlib

    fbcode_skip_file_path = "fb/skip_tests_h100_fbcode.yaml"
    SKIP_FILE = importlib.resources.files(__package__).joinpath(fbcode_skip_file_path)
else:
    import os

    SKIP_FILE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "skip_tests_h100_pytorch.yaml")
    )

with open(SKIP_FILE, "r") as f:
    skip_tests = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ops that we can skip the unit tests
SKIP_OPS = {
    "test_op",
}

TEST_OPERATORS = set(list_operators_by_collection(op_collection="default")) - SKIP_OPS


def check_ci_output(op):
    from tritonbench.utils.triton_op import REGISTERED_BENCHMARKS

    output = op.output
    output_impls = output.result[0][1].keys()
    ci_enabled_impls = [
        x for x in REGISTERED_BENCHMARKS[output.op_name].keys() if x not in op._skip
    ]
    # Make sure that all the ci_enabled impls are in the output
    logger.info(f"output impls: {output_impls}, ci_enabled impls: {ci_enabled_impls}")
    assert set(output_impls) == set(
        ci_enabled_impls
    ), f"output impls: {output_impls} != ci_enabled impls: {ci_enabled_impls}"


def _run_one_operator(args: List[str]):
    parser = get_parser(args)
    tb_args, extra_args = parser.parse_known_args(args)
    if tb_args.op in skip_tests:
        # If the op itself is in the skip list, skip all tests
        if not skip_tests[tb_args.op]:
            return
        tb_args.skip = ",".join(skip_tests[tb_args.op])
    Operator = load_opbench_by_name(tb_args.op)

    op = Operator(tb_args=tb_args, extra_args=extra_args)
    op.run()
    check_ci_output(op)
    del op
    # Test backward (if applicable)
    try:
        tb_args.mode = "bwd"
        op = Operator(tb_args=tb_args, extra_args=extra_args)
        op.run()
        check_ci_output(op)
    except NotImplementedError:
        logger.info(
            f"Operator {op.name} does not support backward, skipping backward test."
        )


def _run_operator_in_task(op: str, args: List[str]):
    from tritonbench.operators.op_task import OpTask

    if op in skip_tests:
        # If the op itself is in the skip list, skip all tests
        if not skip_tests[op]:
            return
        skip = ",".join(skip_tests[op])
        args.extend(["--skip", skip])
    task = OpTask(op)
    task.make_operator_instance(args=args)
    task.run()
    task.check_output()
    task.del_op_instance()
    # Test backward (if applicable)
    try:
        args.extend(["--bwd"])
        task.make_operator_instance(args=args)
        task.run()
        task.check_output()
    except NotImplementedError:
        # Operator does not support backward, skip the test
        pass


def make_test(operator):
    def test_case(self):
        # Add `--test-only` to disable Triton autotune in tests
        args = [
            "--op",
            operator,
            "--device",
            "cuda",
            "--num-inputs",
            "1",
            "--test-only",
        ]
        if IS_FBCODE:
            _run_one_operator(args)
        else:
            _run_operator_in_task(op=operator, args=args)

    return test_case


class TestTritonbenchGpu(unittest.TestCase):
    pass


for operator in TEST_OPERATORS:
    setattr(
        TestTritonbenchGpu,
        f"test_gpu_tritonbench_{operator}",
        make_test(operator),
    )
