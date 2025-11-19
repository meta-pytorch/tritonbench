import argparse
import copy
import logging
import os
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path

from typing import Dict, List, Optional

import torch
import yaml

from tritonbench.operator_loader import get_op_loader_bench_cls_by_name, is_loader_op
from tritonbench.operators import load_opbench_by_name
from tritonbench.operators_collection import list_operators_by_collection
from tritonbench.utils.ab_test import compare_ab_results, run_ab_test
from tritonbench.utils.env_utils import is_fbcode
from tritonbench.utils.git_utils import get_branch, get_commit_time, get_current_hash
from tritonbench.utils.gpu_utils import gpu_lockdown
from tritonbench.utils.list_operator_details import list_operator_details
from tritonbench.utils.parser import get_parser
from tritonbench.utils.path_utils import (
    add_cmd_parameter,
    get_cmd_parameter,
    remove_cmd_parameter,
    REPO_PATH,
)
from tritonbench.utils.triton_op import BenchmarkOperatorResult
from tritonbench.utils.tritonparse_utils import tritonparse_init, tritonparse_parse

try:
    if is_fbcode():
        from .fb.utils import usage_report_logger  # @manual
    else:
        usage_report_logger = lambda *args, **kwargs: None
except ImportError:
    usage_report_logger = lambda *args, **kwargs: None

BENCHMARKS_OUTPUT_DIR = REPO_PATH.joinpath(".benchmarks")
FWD_ONLY_OPS = ["triton_dot_compress", "triton_group_index_select"]
BWD_ARGS_OPS = {
    # flash_attention/triton_tutorial_flash_v2 does not support non-causal in backward
    "flash_attention": ["--causal"],
    # pffn_baseline does not support backward
    "generalized_dot_product_attention": [
        "--skip",
        "pffn_baseline,mkl_jfav3",
    ],
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_run_env(
    run_timestamp: str, repo_locs: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Gather environment of the benchmark.
    repo_locs: Git repository dict of the repositories.
    """
    run_env = {}
    run_env["benchmark_date"] = run_timestamp
    run_env["cuda_version"] = torch.version.cuda if torch.version.cuda else "unknown"
    try:
        run_env["device"] = torch.cuda.get_device_name()
    except AssertionError:
        run_env["device"] = "unknown"
    run_env["conda_env"] = os.environ.get("CONDA_ENV", "unknown")
    run_env["pytorch_commit"] = torch.version.git_version
    # we assume Tritonbench CI will properly set Triton commit hash in env
    run_env["triton_commit"] = os.environ.get(
        "TRITONBENCH_TRITON_COMMIT_HASH", get_current_hash(repo_locs["triton"])
    )
    run_env["tritonbench_commit"] = get_current_hash(repo_locs["tritonbench"])
    for repo in ["triton", "pytorch", "tritonbench"]:
        repo_loc = repo_locs.get(repo, None)
        if not run_env[f"{repo}_commit"] == "unknown" and repo_loc:
            run_env[f"{repo}_branch"] = get_branch(repo_loc, run_env[f"{repo}_commit"])
            run_env[f"{repo}_commit_time"] = get_commit_time(
                repo_loc, run_env[f"{repo}_commit"]
            )
        else:
            run_env[f"{repo}_branch"] = "unknown"
            run_env[f"{repo}_commit_time"] = "unknown"
    return run_env


def _env_get_str(var_name: str, default: str) -> str:
    value = os.environ.get(var_name)
    if value is None:
        return default
    return value.strip() or default


def run_in_helion(op_args: Dict[str, str], extra_envs: Dict[str, str]):
    # Allow override via TRITONBENCH_HELION_PATH; fallback to the current default.
    default_helion = REPO_PATH.joinpath(".install", "helion")
    helion_root = (
        Path(_env_get_str("TRITONBENCH_HELION_PATH", str(default_helion)))
        .expanduser()
        .resolve()
    )
    helion_entry = helion_root / "benchmarks"
    if not helion_entry.exists():
        raise FileNotFoundError(
            f"Invalid TRITONBENCH_HELION_PATH: {helion_root}\n"
            "Expected to find 'benchmarks/run.py'. "
            "Set TRITONBENCH_HELION_PATH to a Helion checkout or run 'python install.py --helion'."
        )
    environ = os.environ.copy()
    environ.update(extra_envs)
    cmd = [sys.executable, "benchmarks/run.py"] + op_args
    print(
        f"[tritonbench] Running helion benchmark: " + " ".join(cmd),
        flush=True,
    )
    subprocess.check_call(
        cmd,
        cwd=helion_root,
        env=environ,
    )


def tritonbench_run(args: Optional[List[str]] = None):
    if args == None or args == []:
        args = sys.argv[1:]
    if config := os.environ.get("TRITONBENCH_RUN_CONFIG", None):
        run_config(config, args)
        return

    # Log the tool usage
    usage_report_logger(benchmark_name="tritonbench")
    parser = get_parser()
    args, extra_args = parser.parse_known_args(args)

    tritonparse_init(args.tritonparse)

    if args.device == "mtia":
        import mtia.host_runtime.torch_mtia.dynamic_library  # noqa
        from mtia.host_runtime.torch_mtia import dynamo_backends  # noqa
        from triton_mtia.python.mtia.eager import mtia_triton_launcher

        # Initialize MTIA's streaming runtime.
        torch.mtia.init()
        mtia_triton_launcher.init()

    if args.op:
        ops = args.op.split(",")
    else:
        ops = list_operators_by_collection(args.op_collection)

    # Handle --list-metrics and --list-backends after determining operators list
    if args.list_metrics or args.list_backends:
        print(
            list_operator_details(
                operators=ops if ops else None,
                show_metrics=args.list_metrics,
                show_backends=args.list_backends,
            )
        )
        return

    # Check if A/B testing mode is enabled
    if args.side_a is not None and args.side_b is not None:
        # A/B testing mode - only support single operator
        assert (
            len(ops) == 1
        ), "A/B testing validation should have caught multiple operators"
        op = ops[0]
        args.op = op

        print("[A/B Testing Mode Enabled]")
        print(f"Operator: {op}")
        print()

        with gpu_lockdown(args.gpu_lockdown):
            try:
                result_a, result_b = run_ab_test(args, extra_args, _run)

                from tritonbench.utils.ab_test import parse_ab_config

                config_a_args = parse_ab_config(args.side_a)
                config_b_args = parse_ab_config(args.side_b)
                compare_ab_results(result_a, result_b, config_a_args, config_b_args)

            except Exception as e:
                print(f"A/B test failed: {e}")
                if not args.bypass_fail:
                    raise
    else:
        # Normal mode
        # Force isolation in subprocess if testing more than one op.
        if len(ops) >= 2:
            args.isolate = True

        with gpu_lockdown(args.gpu_lockdown):
            for op in ops:
                args.op = op
                if args.isolate:
                    run_in_task(op)
                else:
                    _run(args, extra_args)

    tritonparse_parse(args.tritonparse)


def _run(args: argparse.Namespace, extra_args: List[str]) -> BenchmarkOperatorResult:
    run_timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    if is_loader_op(args.op):
        Opbench = get_op_loader_bench_cls_by_name(args.op)
    else:
        Opbench = load_opbench_by_name(args.op)
    opbench = Opbench(
        tb_args=args,
        extra_args=extra_args,
    )
    try:
        opbench.run(args.warmup, args.rep, sleep=args.sleep)
    finally:
        metrics = opbench.output
        if is_fbcode() and args.log_scuba:
            from .fb.utils import log_benchmark  # @manual

            kwargs = {
                "metrics": metrics,
                "benchmark_name": args.op,
                "device": args.device,
                "logging_group": args.logging_group or args.op,
                "precision": args.precision,
            }
            if args.production_shapes:
                from tritonbench.utils.fb.durin_data import productionDataLoader

                kwargs["weights_loader"] = productionDataLoader

            if "hardware" in args:
                kwargs["hardware"] = args.hardware
            if "triton_type" in args:
                kwargs["triton_type"] = args.triton_type
            log_benchmark(**kwargs)
        # Log benchmark output to scuba even if not in fbcode
        if args.log_scuba and not is_fbcode():
            from tritonbench.utils.scuba_utils import log_benchmark

            log_benchmark(
                benchmark_data=None, run_timestamp=run_timestamp, opbench=opbench
            )

        if args.plot:
            try:
                opbench.plot()
            except NotImplementedError:
                print(f"Plotting is not implemented for {args.op}")

        if args.output:
            with open(args.output, "w") as f:
                metrics.write_csv_to_file(f)
            print(f"[tritonbench] Output result csv to {args.output}")
        if args.output_json:
            with open(args.output_json, "w") as f:
                metrics.write_json_to_file(f)
        if args.output_dir:
            if args.csv:
                output_file = os.path.join(args.output_dir, f"{args.op}.csv")
                with open(output_file, "w") as f:
                    metrics.write_csv_to_file(f)
            else:
                output_file = os.path.join(args.output_dir, f"{args.op}.json")
                with open(output_file, "w") as f:
                    metrics.write_json_to_file(f)
        if not args.skip_print:
            if args.csv:
                metrics.write_csv_to_file(sys.stdout)
            else:
                print(metrics)
        return metrics


def run_config(config_file: str, args: List[str]):
    assert Path(config_file).exists(), f"Config file {config_file} must exist."
    # Remove "TRITONBENCH_RUN_CONFIG" env
    if "TRITONBENCH_RUN_CONFIG" in os.environ:
        del os.environ["TRITONBENCH_RUN_CONFIG"]
    with open(config_file, "r") as fp:
        config = yaml.safe_load(fp)
    for benchmark_name in config:
        benchmark_config = config[benchmark_name]
        runner = benchmark_config.get("runner", None)
        op_args = benchmark_config["args"].split(" ") + args
        env_string = benchmark_config.get("envs", None)
        extra_envs = {}
        if env_string:
            for env_part in env_string.split(" "):
                key, val = env_part.split("=")
                extra_envs[key] = val
        disabled = benchmark_config.get("disabled", False)
        if disabled:
            logger.info(f"Skipping disabled benchmark {benchmark_name}.")
            continue
        if runner == "helion":
            run_in_helion(op_args, extra_envs)
        else:
            op_name = benchmark_config["op"]
            run_in_task(
                op=op_name,
                op_args=op_args,
                benchmark_name=benchmark_name,
                extra_envs=extra_envs,
            )


def load_operator_by_args(task_args: List[str]):
    parser = get_parser(task_args)
    tb_args, extra_args = parser.parse_known_args(task_args)
    Operator = load_opbench_by_name(tb_args.op)
    return Operator(tb_args=tb_args, extra_args=extra_args)


def run_one_operator(task_args: List[str], with_bwd: bool = False):
    op = load_operator_by_args(task_args)
    op.run()
    if with_bwd and op.has_bwd() and not op.name in FWD_ONLY_OPS:
        op_name = copy.deepcopy(op.name)
        del op
        if op_name in BWD_ARGS_OPS:
            task_args = copy.deepcopy(task_args)
            task_args.extend(BWD_ARGS_OPS[tb_args.op])
        task_args.extend(["--mode", "bwd"])
        op = load_operator_by_args(task_args)
        op.run()


def run_in_task(
    op: Optional[str] = None,
    op_args: Optional[List[str]] = None,
    benchmark_name: Optional[str] = None,
    extra_envs: Optional[Dict[str, str]] = None,
) -> None:
    op_task_cmd = [] if is_fbcode() else [sys.executable]
    if not op_args:
        assert op, "If op_args is none, op must not be None."
        copy_sys_argv = copy.deepcopy(sys.argv)
        copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--op")
        copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--isolate")
        copy_sys_argv = remove_cmd_parameter(copy_sys_argv, "--op-collection")
        add_cmd_parameter(copy_sys_argv, "--op", op)
        op_task_cmd.extend(copy_sys_argv)
    else:
        if is_fbcode():
            op_task_cmd.append(sys.argv[0])
        op_task_cmd.extend(op_args)
    if not op and op_args:
        op = get_cmd_parameter(op_args, "--op")
    if benchmark_name:
        op_args.extend(["--benchmark-name", benchmark_name])
    else:
        assert op, "If benchmark_name is none, op must not be None."
        benchmark_name = op

    # In OSS, we assume always using the run.py benchmark driver
    if not is_fbcode() and not op_task_cmd[1] == "run.py":
        op_task_cmd.insert(1, "run.py")
    try:
        print(
            f"[tritonbench] Running {benchmark_name}: " + " ".join(op_task_cmd),
            flush=True,
        )
        subprocess_env = os.environ.copy()
        subprocess_env.update(extra_envs or {})
        subprocess.check_call(
            op_task_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=REPO_PATH,
            env=subprocess_env,
        )
    except subprocess.CalledProcessError:
        # By default, we will continue on the failed operators
        pass
    except KeyboardInterrupt:
        logger.warning("[tritonbench] KeyboardInterrupt received, exiting...")
        sys.exit(1)


def setup_output_dir(bm_name: str, ci: bool = False):
    current_timestamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
    output_dir = BENCHMARKS_OUTPUT_DIR.joinpath(bm_name, f"run-{current_timestamp}")
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    # set writable permission for all users (used by the ci env)
    if ci:
        output_dir.chmod(0o777)
    return current_timestamp, output_dir.absolute()
