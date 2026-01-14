import logging
import unittest
from typing import Dict, List

import yaml
from tritonbench.operators import (  # @manual=//pytorch/tritonbench:tritonbench
    load_opbench_by_name,
)
from tritonbench.operators_collection import (
    list_operators_by_collection,  # @manual=//pytorch/tritonbench:tritonbench
)
from tritonbench.utils.env_utils import (
    is_fbcode,  # @manual=//pytorch/tritonbench:tritonbench
    is_hip,  # @manual=//pytorch/tritonbench:tritonbench
)
from tritonbench.utils.parser import get_parser

if is_fbcode():
    import importlib

    fbcode_skip_file_path = "fb/skip_tests_h100_fbcode.yaml"
    SKIP_FILE = importlib.resources.files(__package__).joinpath(fbcode_skip_file_path)
else:
    import triton  # @manual

    if "site-packages" in triton.__file__:
        SKIP_FILE_NAME = "skip_tests_h100_pytorch.yaml"
    else:
        SKIP_FILE_NAME = (
            "skip_tests_mi350_triton_main.yaml"
            if is_hip()
            else "skip_tests_h100_triton_main.yaml"
        )
    import os

    SKIP_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), SKIP_FILE_NAME))

with open(SKIP_FILE, "r") as f:
    skip_tests = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ops that we run forward only
FWD_ONLY_OPS = skip_tests.get("fwd_only_ops", [])
# Ops that require special arguments in backwards
BWD_ARGS_OPS: Dict[str, List[str]] = skip_tests.get("bwd_args", {})

TEST_OPERATORS = (
    set(list_operators_by_collection(op_collection="buck"))
    if is_fbcode()
    else set(list_operators_by_collection(op_collection="default"))
)


def check_ci_output(op):
    from tritonbench.utils.triton_op import (
        find_enabled_benchmarks,  # @manual=//pytorch/tritonbench:tritonbench
        REGISTERED_BENCHMARKS,  # @manual=//pytorch/tritonbench:tritonbench
    )

    output = op.output
    output_impls = output.result[0][1].keys()
    ci_enabled_impls = find_enabled_benchmarks(
        op.mode, REGISTERED_BENCHMARKS[op.name], op._skip
    )
    # Make sure that all the ci_enabled impls are in the output
    logger.info(f"output impls: {output_impls}, ci_enabled impls: {ci_enabled_impls}")
    assert set(output_impls) == set(ci_enabled_impls), (
        f"output impls: {output_impls} != ci_enabled impls: {ci_enabled_impls}"
    )


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
    # Test backward (if applicable)
    if tb_args.op in FWD_ONLY_OPS:
        return
    if op.has_bwd():
        del op
        if tb_args.op in BWD_ARGS_OPS:
            args.extend(BWD_ARGS_OPS[tb_args.op])
            tb_args, extra_args = parser.parse_known_args(args)
        tb_args.mode = "bwd"
        op = Operator(tb_args=tb_args, extra_args=extra_args)
        op.run()
        check_ci_output(op)


def _run_operator_in_task(op: str, args: List[str]):
    from tritonbench.operators.op_task import (  # @manual=//pytorch/tritonbench:tritonbench
        OpTask,
    )

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
    # Test backward (if applicable)
    if op in FWD_ONLY_OPS:
        return
    if task.get_attribute("has_bwd", method=True):
        task.del_op_instance()
        args.extend(["--bwd"])
        if op in BWD_ARGS_OPS:
            args.extend(BWD_ARGS_OPS[op])
        task.make_operator_instance(args=args)
        task.run()
        task.check_output()


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
        if is_fbcode():
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
