import logging
import os
import unittest
from typing import List

import yaml
from tritonbench.operators import (  # @manual=//pytorch/tritonbench:tritonbench
    load_opbench_by_name,
)
from tritonbench.operators.op_task import (  # @manual=//pytorch/tritonbench:tritonbench
    OpTask,
)
from tritonbench.operators_collection import (
    list_operators_by_collection,  # @manual=//pytorch/tritonbench:tritonbench
)
from tritonbench.utils.env_utils import (
    is_blackwell,  # @manual=//pytorch/tritonbench:tritonbench
    is_fbcode,  # @manual=//pytorch/tritonbench:tritonbench
)
from tritonbench.utils.parser import get_parser
from tritonbench.utils.run_utils import _env_check

if is_fbcode():
    import importlib

    fbcode_skip_file_path = "fb/skip_tests.yaml"
    SKIP_FILE = importlib.resources.files(__package__).joinpath(fbcode_skip_file_path)
else:
    SKIP_FILE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "skip_tests.yaml")
    )

with open(SKIP_FILE, "r") as f:
    skip_tests = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_OPERATORS = (
    set(list_operators_by_collection(op_collection="buck"))
    if is_fbcode()
    else set(list_operators_by_collection(op_collection="default"))
)


def _gen_test_operators(test_ops, skip_tests) -> set[str]:
    # to save capacity, only run tests specific to b200 on b200
    if is_blackwell():
        test_ops = {
            test_op: {"disabled": False}
            for test_op in skip_tests
            if "devices" in skip_tests[test_op]
            and "b200" in skip_tests[test_op]["devices"]
        }
    else:
        test_ops = {test_op: {"disabled": False} for test_op in test_ops}
    for skip_op in skip_tests:
        if not skip_op in test_ops:
            continue
        # remove operators that are unconditionally bypassed on CI
        # ususally CI environment issue or broken operators need to be fixed
        if skip_tests[skip_op] == None:
            test_ops[skip_op]["disabled"] = True
            continue
        for field_name in ["devices", "channels"]:
            test_ops[skip_op]["disabled"] = test_ops[skip_op][
                "disabled"
            ] or not _env_check(skip_tests[skip_op], field_name)
    return {test_op for test_op in test_ops if not test_ops[test_op]["disabled"]}


TEST_OPERATORS = _gen_test_operators(TEST_OPERATORS, skip_tests)


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


class MaybeTestOperatorTask:
    def __init__(self, op: str, args: List[str], in_task: bool = False):
        if in_task:
            self.in_task = True
            task = OpTask(op)
            task.make_operator_instance(args=args)
            self.op = task
        else:
            self.in_task = False
            Operator = load_opbench_by_name(op)
            parser = get_parser(args)
            tb_args, extra_args = parser.parse_known_args(args)
            self.op = Operator(tb_args=tb_args, extra_args=extra_args)

    def check_output(self):
        if not self.in_task:
            check_ci_output(self.op)
        else:
            self.op.check_output()

    def run(self):
        self.op.run()

    def has_bwd(self):
        return self.op.has_bwd()


def _run_one_operator(op: str, args: List[str], in_task: bool = False):
    extra_args_from_skip_files = (
        skip_tests[op]["extra_args"].split(" ")
        if skip_tests.get(op, None) and skip_tests[op].get("extra_args", None)
        else []
    )
    args.extend(extra_args_from_skip_files)
    opbench = MaybeTestOperatorTask(op, args, in_task)
    opbench.run()
    opbench.check_output()

    # Test backward (if applicable)
    if opbench.has_bwd():
        del opbench
        extra_bwd_args = (
            skip_tests[op]["extra_bwd_args"].split(" ")
            if skip_tests.get(op, None) and skip_tests[op].get("extra_bwd_args", None)
            else []
        )
        if extra_bwd_args:
            args.extend(extra_bwd_args)
        args.extend(["--bwd"])
        opbench = MaybeTestOperatorTask(op, args, in_task)
        opbench.run()
        opbench.check_output()


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
        in_task = not is_fbcode()
        _run_one_operator(op=operator, args=args, in_task=in_task)

    return test_case


class TestTritonbenchGpu(unittest.TestCase):
    pass


for operator in TEST_OPERATORS:
    setattr(
        TestTritonbenchGpu,
        f"test_gpu_tritonbench_{operator}",
        make_test(operator),
    )
