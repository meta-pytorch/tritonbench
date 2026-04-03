import argparse
import copy
import csv
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import yaml
from tritonbench.operator_loader import get_op_loader_bench_cls_by_name, is_loader_op
from tritonbench.operators import load_opbench_by_name
from tritonbench.operators_collection import list_operators_by_collection
from tritonbench.utils.ab_test import compare_ab_results, run_ab_test
from tritonbench.utils.env_utils import (
    is_blackwell,
    is_cuda,
    is_fbcode,
    is_h100,
    is_hip,
    is_meta_triton,
    is_triton_beta,
    is_triton_main,
    is_triton_stable,
    set_torchrun_env,
)
from tritonbench.utils.git_utils import get_branch, get_commit_time, get_current_hash
from tritonbench.utils.gpu_telemetry_observer import TelemetryContext
from tritonbench.utils.gpu_utils import get_amd_device_name, gpu_lockdown
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

ENV_CHECK_MAP = {
    "devices": {
        "h100": is_h100,
        "b200": is_blackwell,
        "cuda": is_cuda,
        "hip": is_hip,
    },
    "channels": {
        "triton-main": is_triton_main,
        "triton-beta": is_triton_beta,
        "meta-triton": is_meta_triton,
        "triton-stable": is_triton_stable,
    },
}

SPECIAL_CONFIG_FIELDS = {"common_args"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_run_env(
    run_timestamp: str, repo_locs: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Gather environment of the benchmark.
    repo_locs: Git repository dict of the repositories.
    """
    run_env = {}
    run_env["benchmark_date"] = run_timestamp
    if is_hip():
        run_env["cuda_version"] = torch.version.hip
    else:
        run_env["cuda_version"] = (
            torch.version.cuda if torch.version.cuda else "unknown"
        )
    try:
        run_env["device"] = (
            get_amd_device_name() if is_hip() else torch.cuda.get_device_name()
        )
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
            print(
                "trying to get commit branch for",
                repo,
                "from",
                repo_loc,
                " commit hash: ",
                run_env[f"{repo}_commit"],
            )
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


def _env_check(benchmark_config: Dict[str, str], field_name: str) -> bool:
    """True means we should run the benchmark, False means we should skip it."""
    check_map = ENV_CHECK_MAP[field_name]
    field_val = benchmark_config.get(field_name, None)
    assert field_val is None or all(
        [field_val in check_map for field_val in check_map]
    ), f"Unknown {field_name} value: {field_val}"
    if field_val is None:
        return True
    return any([check_map[val]() for val in field_val])


def _get_helion_root():
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
    return helion_root


def _get_total_available_inputs(
    args: argparse.Namespace, extra_args: List[str]
) -> Tuple[int, int]:
    """Determine the total number of inputs to shard, respecting --input-id and --num-inputs.

    Returns (start_input_id, total_inputs_to_run).
    """
    probe_args = copy.deepcopy(args)
    probe_args.input_id = "0"
    probe_args.num_inputs = 1

    if is_loader_op(probe_args.op):
        Opbench = get_op_loader_bench_cls_by_name(probe_args.op)
    else:
        Opbench = load_opbench_by_name(probe_args.op)
    opbench = Opbench(tb_args=probe_args, extra_args=list(extra_args))
    available = opbench._available_num_inputs
    del opbench

    start_id = int(args.input_id) if args.input_id else 0
    if args.num_inputs is not None:
        total = min(args.num_inputs, available - start_id)
    else:
        total = available - start_id

    return start_id, total


def _build_device_cmd(
    base_argv: List[str],
    input_id_start: int,
    num_inputs: int,
    output_dir: str,
    use_csv: bool = True,
) -> List[str]:
    """Build a subprocess command for a single device shard."""
    cmd = list(base_argv)
    cmd = remove_cmd_parameter(cmd, "--devices")
    cmd = remove_cmd_parameter(cmd, "--shard-by-inputs")
    cmd = remove_cmd_parameter(cmd, "--input-id")
    cmd = remove_cmd_parameter(cmd, "--num-inputs")
    cmd = remove_cmd_parameter(cmd, "--output")
    cmd = remove_cmd_parameter(cmd, "--output-dir")
    cmd = remove_cmd_parameter(cmd, "--output-json")
    cmd = remove_cmd_parameter(cmd, "--skip-print")
    cmd = remove_cmd_parameter(cmd, "--csv")
    cmd.extend(["--input-id", str(input_id_start)])
    cmd.extend(["--num-inputs", str(num_inputs)])
    cmd.extend(["--output-dir", output_dir])
    if use_csv:
        cmd.extend(["--csv"])
    cmd.extend(["--skip-print"])
    return cmd


def _merge_csv_files(csv_files: List[str], output_path: str) -> None:
    """Merge multiple semicolon-delimited CSV files into one, preserving header from the first."""
    header = None
    all_rows = []
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue
        with open(csv_file, "r", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            rows = list(reader)
            if not rows:
                continue
            if header is None:
                header = rows[0]
            all_rows.extend(rows[1:])

    if header is None:
        logger.warning("[tritonbench] No CSV output files found to merge.")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        writer.writerows(all_rows)


def _merge_json_files(json_files: List[str], output_path: str) -> None:
    """Merge multiple JSON benchmark output files into one by combining their metric dicts."""
    merged: Dict[str, Any] = {}
    for json_file in json_files:
        if not os.path.exists(json_file):
            continue
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(f"[tritonbench] Skipping unreadable JSON file: {json_file}")
            continue
        if isinstance(data, dict):
            merged.update(data)

    if not merged:
        logger.warning("[tritonbench] No JSON output files found to merge.")
        return

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=4)


def run_multi_device(
    args: argparse.Namespace,
    extra_args: List[str],
) -> None:
    """Run benchmarks across multiple GPU devices with input sharding."""
    from tritonbench.utils.device_utils import (
        compute_input_shards,
        MIN_INPUTS_PER_DEVICE,
        parse_device_range,
        validate_device_ids,
    )

    device_ids = parse_device_range(args.devices)
    validate_device_ids(device_ids)

    if not args.shard_by_inputs:
        raise NotImplementedError(
            "--devices without --shard-by-inputs is not yet supported. "
            "Currently, multi-device mode requires --shard-by-inputs."
        )

    start_id, total_inputs = _get_total_available_inputs(args, extra_args)
    shards = compute_input_shards(total_inputs, len(device_ids))

    if len(shards) < len(device_ids):
        logger.warning(
            f"[tritonbench] Reducing devices from {len(device_ids)} to {len(shards)} "
            f"to ensure each device gets at least {MIN_INPUTS_PER_DEVICE} inputs "
            f"(total_inputs={total_inputs})"
        )
        device_ids = device_ids[: len(shards)]

    use_csv = bool(getattr(args, "csv", False))
    output_ext = "csv" if use_csv else "json"

    logger.info(
        f"[tritonbench] Multi-device mode: {len(device_ids)} devices, "
        f"{total_inputs} total inputs (starting from id {start_id}), "
        f"output format: {output_ext}"
    )
    for device_id, (shard_offset, shard_size) in zip(device_ids, shards):
        logger.info(
            f"[tritonbench]   Device {device_id}: input_id={start_id + shard_offset}, "
            f"num_inputs={shard_size}"
        )

    base_argv = [] if is_fbcode() else [sys.executable]
    base_argv.extend(copy.deepcopy(sys.argv))
    if not is_fbcode() and len(base_argv) > 1 and base_argv[1] != "run.py":
        base_argv.insert(1, "run.py")

    top_level_output_dir = tempfile.mkdtemp(prefix="tritonbench_multi_device_")
    user_output = args.output
    user_output_json = getattr(args, "output_json", None)
    user_output_dir = args.output_dir

    processes = []
    device_output_dirs = []
    for device_id, (shard_offset, shard_size) in zip(device_ids, shards):
        if shard_size == 0:
            continue

        device_dir = os.path.join(top_level_output_dir, f"device_{device_id}")
        os.makedirs(device_dir, exist_ok=True)
        device_output_dirs.append((device_id, device_dir))

        cmd = _build_device_cmd(
            base_argv,
            input_id_start=start_id + shard_offset,
            num_inputs=shard_size,
            output_dir=device_dir,
            use_csv=use_csv,
        )

        subprocess_env = os.environ.copy()
        subprocess_env["CUDA_VISIBLE_DEVICES"] = str(device_id)

        stdout_path = os.path.join(device_dir, "stdout.log")
        stderr_path = os.path.join(device_dir, "stderr.log")

        stdout_f = open(stdout_path, "w")
        stderr_f = open(stderr_path, "w")

        logger.info(f"[tritonbench] Launching device {device_id}: " + " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=REPO_PATH,
                env=subprocess_env,
            )
        except Exception:
            stdout_f.close()
            stderr_f.close()
            raise
        processes.append((device_id, proc, stdout_f, stderr_f))

    failures = []
    successes = []
    for device_id, proc, stdout_f, stderr_f in processes:
        proc.wait()
        stdout_f.close()
        stderr_f.close()
        if proc.returncode != 0:
            device_dir = os.path.join(top_level_output_dir, f"device_{device_id}")
            stderr_path = os.path.join(device_dir, "stderr.log")
            try:
                with open(stderr_path, "r") as f:
                    stderr_content = f.read()
            except OSError:
                stderr_content = "<unable to read stderr>"
            failures.append((device_id, proc.returncode, stderr_content))
            logger.error(
                f"[tritonbench] Device {device_id} failed with return code "
                f"{proc.returncode}:\n{stderr_content}"
            )
        else:
            successes.append(device_id)

    op_name = args.op
    per_device_outputs = []
    for device_id, device_dir in device_output_dirs:
        output_file = os.path.join(device_dir, f"{op_name}.{output_ext}")
        if os.path.exists(output_file):
            per_device_outputs.append(output_file)

    merged_path = None
    if per_device_outputs:
        if user_output:
            merged_path = user_output
        elif user_output_json and not use_csv:
            merged_path = user_output_json
        elif user_output_dir:
            os.makedirs(user_output_dir, exist_ok=True)
            merged_path = os.path.join(user_output_dir, f"{op_name}.{output_ext}")
        else:
            merged_path = os.path.join(
                top_level_output_dir, f"{op_name}_merged.{output_ext}"
            )

        if use_csv:
            _merge_csv_files(per_device_outputs, merged_path)
        else:
            _merge_json_files(per_device_outputs, merged_path)

    _print_multi_device_summary(
        op_name=op_name,
        device_ids=device_ids,
        successes=successes,
        failures=failures,
        merged_path=merged_path,
        output_format=output_ext,
        top_level_output_dir=top_level_output_dir,
    )

    if len(failures) == len(processes):
        raise RuntimeError(
            f"[tritonbench] All {len(failures)} device(s) failed. "
            f"Check logs in {top_level_output_dir}"
        )


def _print_multi_device_summary(
    op_name: str,
    device_ids: List[int],
    successes: List[int],
    failures: List[Tuple[int, int, str]],
    merged_path: Optional[str],
    output_format: str,
    top_level_output_dir: str,
) -> None:
    """Print a consolidated summary of the multi-device benchmark run."""
    print(f"\n{'=' * 60}")
    print(f"Multi-Device Benchmark Summary: {op_name}")
    print(f"{'=' * 60}")
    print(f"Devices:    {len(device_ids)} ({', '.join(str(d) for d in device_ids)})")
    print(f"Succeeded:  {len(successes)}")
    print(f"Failed:     {len(failures)}")
    if merged_path and os.path.exists(merged_path):
        print(f"Output:     {merged_path} ({output_format})")
    if failures:
        print(f"\nFailed devices:")
        for device_id, returncode, _ in failures:
            stderr_log = os.path.join(
                top_level_output_dir, f"device_{device_id}", "stderr.log"
            )
            print(f"  Device {device_id}: exit code {returncode} (see {stderr_log})")
    print(f"Logs:       {top_level_output_dir}")
    print(f"{'=' * 60}\n")


def tritonbench_run(args: Optional[List[str]] = None):
    if args == None or args == []:
        args = sys.argv[1:]
    if config := os.environ.get("TRITONBENCH_RUN_CONFIG", None):
        run_config(config, args)
        return

    set_torchrun_env()

    # Log the tool usage
    usage_report_logger(benchmark_name="tritonbench")
    parser = get_parser()
    args, extra_args = parser.parse_known_args(args)

    tritonparse_init(args.tritonparse)

    if args.devices:
        run_multi_device(args, extra_args)
        tritonparse_parse(args.tritonparse)
        return

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
        assert len(ops) == 1, (
            "A/B testing validation should have caught multiple operators"
        )
        op = ops[0]
        args.op = op

        print("[A/B Testing Mode Enabled]")
        print(f"Operator: {op}")
        print()

        lockdown_enabled = args.gpu_lockdown or (args.gpu_lock_clock_mhz is not None)
        with gpu_lockdown(lockdown_enabled, args.gpu_lock_clock_mhz):
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

        lockdown_enabled = args.gpu_lockdown or (args.gpu_lock_clock_mhz is not None)
        with gpu_lockdown(lockdown_enabled, args.gpu_lock_clock_mhz):
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

    # Set up GPU telemetry observer if enabled
    gpu_telemetry_enabled = getattr(args, "gpu_telemetry", False)
    telemetry_ctx = None
    if gpu_telemetry_enabled:
        try:
            interval_ms = getattr(args, "gpu_telemetry_interval_ms", 50.0)
            telemetry_ctx = TelemetryContext(gpu_id=0, sample_interval_ms=interval_ms)
            logger.info(
                f"[tritonbench] GPU telemetry enabled (interval={interval_ms}ms)"
            )
        except Exception:
            logger.warning(
                "[tritonbench] GPU telemetry requested but observer not available"
            )

    try:
        # Start telemetry if enabled
        if telemetry_ctx is not None:
            telemetry_ctx.__enter__()
            telemetry_ctx.annotate("benchmark_start")

        opbench.run(args.warmup, args.rep, sleep=args.sleep)
    finally:
        metrics = opbench.output

        # Stop telemetry and save data
        if telemetry_ctx is not None:
            telemetry_ctx.annotate("benchmark_end")
            telemetry_ctx.__exit__(None, None, None)

            if telemetry_ctx.data is not None and telemetry_ctx.data.samples:
                telemetry_output = getattr(args, "gpu_telemetry_output", None)
                if telemetry_output:
                    telemetry_base = telemetry_output
                else:
                    telemetry_base = f"/tmp/gpu_telemetry_{args.op}_{run_timestamp}"

                telemetry_csv = f"{telemetry_base}.csv"
                telemetry_png = f"{telemetry_base}.png"

                telemetry_ctx.save_csv(telemetry_csv)
                telemetry_ctx.plot(telemetry_png, title=f"GPU Telemetry: {args.op}")

                logger.info(
                    f"[tritonbench] GPU telemetry saved to {telemetry_csv} and {telemetry_png} "
                    f"({len(telemetry_ctx.data.samples)} samples)"
                )
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
                logger.error(f"Plotting is not implemented for {args.op}")

        if args.output:
            with open(args.output, "w") as f:
                metrics.write_csv_to_file(f)
            logger.info(f"[tritonbench] Output result csv to {args.output}")
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


def _process_common_args(common_args: str) -> List[str]:
    if not common_args:
        return []
    common_args = common_args.split(" ")
    common_args = [arg for arg in common_args if arg]
    return common_args


def _run_config_entry(
    benchmark_name: str,
    benchmark_config: Dict[str, Any],
    per_benchmark_enable_cond: Callable,
    args: List[str] | None = None,
    extra_envs: Optional[Dict[str, str]] = None,
    override_envs: bool = False,
    capture_output: Optional[str] = None,
) -> bool:
    runner = benchmark_config.get("runner", None)
    op_args = benchmark_config["args"].split(" ") + args
    env_string = benchmark_config.get("envs", None)
    config_extra_envs = {}
    forbidden_list = [";", "$", "&"]
    if env_string:
        _ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
        if any(char in env_string for char in forbidden_list):
            raise ValueError(
                f"Env string containing '{forbidden_list}' is not supported."
            )
        for env_part in shlex.split(env_string, posix=True):
            if not _ASSIGN_RE.match(env_part):
                raise ValueError(
                    f"Env string must be in the form of key=value, get {env_part}"
                )
            key, val = env_part.split("=", 1)
            config_extra_envs[key] = val
    if extra_envs:
        config_extra_envs.update(extra_envs)
    op_name = get_cmd_parameter(op_args, "--op")
    disabled = (
        benchmark_config.get("disabled", False)
        or not _env_check(benchmark_config, "devices")
        or not _env_check(benchmark_config, "channels")
        or not per_benchmark_enable_cond(op_name)
    )
    if disabled:
        logger.info(f"Skipping disabled benchmark {benchmark_name}.")
        return True
    run_in_task(
        op=op_name,
        op_args=op_args,
        runner=runner,
        benchmark_name=benchmark_name,
        extra_envs=config_extra_envs,
        override_envs=override_envs,
        capture_output=capture_output,
    )
    return False


def run_config(
    config_file: str,
    args: List[str],
    extra_envs: Optional[Dict[str, str]] = None,
    override_envs: bool = False,
    capture_output: Optional[str] = None,
    per_config_entry: Dict[str, Any] | None = None,
):
    assert Path(config_file).exists(), (
        f"Config file {config_file} must exist. Current working directory {os.getcwd()}"
    )  # Fbcode only: need to run from fbsource root directory
    # Remove "TRITONBENCH_RUN_CONFIG" env
    if "TRITONBENCH_RUN_CONFIG" in os.environ:
        del os.environ["TRITONBENCH_RUN_CONFIG"]
    with open(config_file, "r") as fp:
        config = yaml.safe_load(fp)
    common_args = _process_common_args(config.get("common_args", ""))
    for field in SPECIAL_CONFIG_FIELDS:
        if field in config:
            del config[field]
    if args is None:
        args = []
    for benchmark_name in config:
        per_benchmark_enable_cond = lambda x: True
        per_benchmark_callback = lambda x: None
        per_benchmark_args = common_args.copy() if common_args else []
        per_benchmark_args += args
        if per_config_entry and benchmark_name in per_config_entry:
            if per_config_entry[benchmark_name].get("extra_args", None):
                per_benchmark_args += per_config_entry[benchmark_name]["extra_args"]
            if per_config_entry[benchmark_name].get("enable_condition", None):
                per_benchmark_enable_cond = per_config_entry[benchmark_name][
                    "enable_condition"
                ]
            if per_config_entry[benchmark_name].get("callback", None):
                per_benchmark_callback = per_config_entry[benchmark_name]["callback"]
        disabled = _run_config_entry(
            benchmark_name,
            config[benchmark_name],
            per_benchmark_enable_cond=per_benchmark_enable_cond,
            args=per_benchmark_args,
            extra_envs=extra_envs,
            override_envs=override_envs,
            capture_output=capture_output,
        )
        per_benchmark_callback(disabled)


def load_operator_by_args(task_args: List[str]):
    parser = get_parser(task_args)
    tb_args, extra_args = parser.parse_known_args(task_args)
    Operator = load_opbench_by_name(tb_args.op)
    return Operator(tb_args=tb_args, extra_args=extra_args)


def run_one_operator(task_args: List[str], with_bwd: bool = False):
    op = load_operator_by_args(task_args)
    op.run()
    if with_bwd and op.has_bwd():
        op_name = copy.deepcopy(op.name)
        del op
        task_args.extend(["--mode", "bwd"])
        op = load_operator_by_args(task_args)
        op.run()


def run_in_task(
    op: Optional[str] = None,
    op_args: Optional[List[str]] = None,
    runner: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    extra_envs: Optional[Dict[str, str]] = None,
    override_envs: bool = False,
    capture_output: Optional[str] = None,
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

    # In OSS, we assume using the run.py benchmark driver
    # In helion, use "benchmarks/run.py" as the runner script
    cwd = REPO_PATH if not runner == "helion" else _get_helion_root()
    runner_script = "run.py" if not runner == "helion" else "benchmarks/run.py"
    if not is_fbcode() and not op_task_cmd[1] == runner_script:
        op_task_cmd.insert(1, runner_script)

    try:
        # if simple output, disable all the logs
        if "--simple-output" in op_task_cmd:
            logger.setLevel(logging.ERROR)
        start_time = time.perf_counter()
        logger.info(
            f"[tritonbench] Running benchmark {benchmark_name}: "
            + " ".join(op_task_cmd)
        )
        if override_envs:
            subprocess_env = extra_envs.copy()
        else:
            subprocess_env = os.environ.copy()
            subprocess_env.update(extra_envs or {})
        if capture_output:
            assert os.path.isdir(capture_output), (
                f"specified capture output dir {capture_output} must exist"
            )
        if capture_output:
            with (
                open(os.path.join(capture_output, "stdout.log"), "w") as stdout,
                open(os.path.join(capture_output, "stderr.log"), "w") as stderr,
            ):
                subprocess.check_call(
                    op_task_cmd,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=cwd,
                    env=subprocess_env,
                )
        else:
            subprocess.check_call(
                op_task_cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                cwd=cwd,
                env=subprocess_env,
            )
        benchmark_time = time.perf_counter() - start_time
        logger.info(
            f"[tritonbench] Complete benchmark {benchmark_name} in {benchmark_time:.3f} seconds."
        )
        return 0
    except subprocess.CalledProcessError as e:
        # By default, we will continue on the failed operators
        return e.returncode
    except KeyboardInterrupt:
        logger.warning("[tritonbench] KeyboardInterrupt received, exiting...")
        sys.exit(1)
