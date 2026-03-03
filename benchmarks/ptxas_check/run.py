"""
PTXAS Options Compatibility Check Benchmark.

Validates that PTXAS_OPTIONS environment variable does not affect benchmark
outputs by running the same command with and without PTXAS_OPTIONS, then
comparing the results.

Usage:
    PTXAS_OPTIONS="..." python -m benchmarks.ptxas_check.run -- \
        --op gemm --precision bf16 --only triton_tutorial_matmul
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..common import setup_output_dir, setup_tritonbench_cwd


setup_tritonbench_cwd()

from tritonbench.utils.run_utils import run_config, run_in_task


def find_pt_files(path: str) -> List[str]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return []
    return [
        f
        for f in os.listdir(abs_path)
        if os.path.isfile(os.path.join(abs_path, f)) and f.endswith(".pt")
    ]


def find_file_with_suffix(path: str, suffix: str = "stderr.log") -> str:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Directory {path} does not exist")

    target_files = [
        f
        for f in os.listdir(abs_path)
        if os.path.isfile(os.path.join(abs_path, f)) and f.endswith(suffix)
    ]
    assert len(target_files) == 1, (
        f"Expected exactly one stderr file, found {len(target_files)}"
    )
    return target_files[0]


def load_pt(filepath: str) -> Any:
    with open(filepath, "rb") as f:
        return torch.load(f)


def check_tensor_numeric(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        try:
            torch.testing.assert_close(a, b)
            return True
        except AssertionError:
            return False

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not check_tensor_numeric(a[i], b[i]):
                return False
        return True

    return a == b


def load_best_config_from_stderr(file_path: str) -> str | None:
    config = None
    config_type = None
    with open(file_path, "r") as f:
        lines = f.readlines()
        for lineno, line in enumerate(lines):
            if "Autotune Choices Stats:" in line:
                config = lines[lineno + 1].strip()
                config_type = "pt2"
            if "best config selected:" in line:
                config = line[len("best config selected: ") :].strip()
                config_type = "triton"
            if "@helion.kernel" in line:
                config = line.strip()
                config_type = "helion"
    if not config_type:
        return None
    elif config_type == "triton" or config_type == "helion":
        # triton style autotune config
        return config
    else:
        # pt2 style autotune config
        config_dict = json.loads(config)
        return config_dict.get("best_kernel_desc", None)


def load_perf_results(stdout_file: str) -> Tuple[Any] | None:
    with open(stdout_file, "r") as f:
        lines = f.readlines()
        if len(lines) == 0:
            return None
        last_line = lines[-1].split(",")
    try:
        results = [float(x) for x in last_line]
    except ValueError:
        return None
    return tuple(results)


def compare_configs(config_a: Any, config_b: Any) -> Optional[bool]:
    if config_a is None and config_b is None:
        return None
    if config_a is None or config_b is None:
        return False
    if isinstance(config_a, dict) and isinstance(config_b, dict):
        if set(config_a.keys()) != set(config_b.keys()):
            return False
        for key in config_a:
            if config_a[key] != config_b[key]:
                return False
        return True
    return str(config_a) == str(config_b)


def run_tritonbench(
    config_file: str,
    extra_args: List[str],
    export_dir: str,
    env: Dict[str, str],
) -> int:
    cmd = []
    cmd.extend(extra_args)
    cmd.extend(["--export", "both", "--export-dir", export_dir])

    env["TRITON_ALWAYS_COMPILE"] = "1"
    env["TRITON_PRINT_AUTOTUNING"] = "1"
    env["HELION_BENCHMARK_DISABLE_LOGGING"] = "0"

    if config_file:
        result = run_config(
            config_file=config_file,
            args=cmd,
            extra_envs=env,
            override_envs=True,
            capture_output=export_dir,
        )
    else:
        result = run_in_task(
            op=None,
            op_args=cmd,
            extra_envs=env,
            override_envs=True,
            capture_output=export_dir,
        )
    return result


def compare_outputs(dir_a: str, dir_b: str) -> Tuple[bool, List[str]]:
    issues = []

    stderr_files_a = find_file_with_suffix(dir_a, suffix="stderr.log")
    stderr_files_b = find_file_with_suffix(dir_b, suffix="stderr.log")
    assert stderr_files_b == stderr_files_a, (
        f"Expected same stderr files, found {stderr_files_b} and {stderr_files_a}"
    )

    stdout_files_a = find_file_with_suffix(dir_a, suffix="stdout.log")
    stdout_files_b = find_file_with_suffix(dir_b, suffix="stdout.log")
    assert stdout_files_a == stdout_files_b, (
        f"Expected same stderr files, found {stdout_files_a} and {stdout_files_b}"
    )
    perf_a = load_perf_results(os.path.join(dir_a, stdout_files_a))
    perf_b = load_perf_results(os.path.join(dir_b, stdout_files_b))
    if perf_a and perf_b:
        print(f"[ptxas-check] perf check: {perf_b} -> {perf_a} ✓")
    elif perf_a and not perf_b:
        issues.append(
            f"Perf results mismatch: only found in run with PTXAS_OPTIONS: {perf_a}"
        )
    elif not perf_a and perf_b:
        issues.append(
            f"Perf results mismatch: only found in run without PTXAS_OPTIONS: {perf_b}"
        )
    else:
        issues.append("Perf results are missing.")

    path_a = os.path.join(dir_a, stderr_files_a)
    path_b = os.path.join(dir_b, stderr_files_a)
    data_a = load_best_config_from_stderr(path_a)
    data_b = load_best_config_from_stderr(path_b)

    compare_configs_result = compare_configs(data_a, data_b)
    if compare_configs_result is None:
        print(f"[ptxas-check] Warning: Autotune config missing in stderr.log files.")
    elif not compare_configs(data_a, data_b):
        issues.append(f"[ptxas-check] Autotune config mismatch in {stderr_files_a}")
        issues.append(f"[ptxas-check]   With PTXAS_OPTIONS: {data_a}")
        issues.append(f"[ptxas-check]   Without PTXAS_OPTIONS: {data_b}")
    else:
        print(f"[ptxas-check] configs match in {stderr_files_a}: {data_a} ✓")

    pt_files_a = set(find_pt_files(dir_a))
    pt_files_b = set(find_pt_files(dir_b))
    common_pkl = pt_files_a & pt_files_b
    only_in_a = pt_files_a - pt_files_b
    only_in_b = pt_files_b - pt_files_a

    if only_in_a:
        issues.append(f"PT files only in run with PTXAS_OPTIONS: {sorted(only_in_a)}")
    if only_in_b:
        issues.append(
            f"PT files only in run without PTXAS_OPTIONS: {sorted(only_in_b)}"
        )
    if not common_pkl:
        issues.append("Numeric files are missing.")

    for pkl_file in sorted(common_pkl):
        path_a = os.path.join(dir_a, pkl_file)
        path_b = os.path.join(dir_b, pkl_file)
        data_a = load_pt(path_a)
        data_b = load_pt(path_b)
        if not check_tensor_numeric(data_a, data_b):
            issues.append(f"Numeric mismatch in {pkl_file}")
        else:
            print(f"[ptxas-check] {pkl_file}: numerics match ✓")

    return len(issues) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PTXAS Options Compatibility Check",
        usage="%(prog)s [options] -- <tritonbench args>",
    )
    args, extra_args = parser.parse_known_args()

    if "--" in extra_args:
        extra_args.remove("--")

    ptxas_options = os.environ.get("PTXAS_OPTIONS", None)
    if ptxas_options is None:
        print("[ptxas-check] ERROR: PTXAS_OPTIONS environment variable is not set.")
        print("[ptxas-check] Please set PTXAS_OPTIONS before running this benchmark.")
        print(
            "[ptxas-check] Example: PTXAS_OPTIONS='-v' python -m benchmarks.ptxas_check.run"
        )
        return 1

    config_file = os.environ.get("TRITONBENCH_RUN_CONFIG", None)
    if config_file is None:
        print(
            "[ptxas-check] ERROR: TRITONBENCH_RUN_CONFIG environment variable is not set."
        )
        print(
            "[ptxas-check] Please set TRITONBENCH_RUN_CONFIG before running this benchmark."
        )
        print(
            "[ptxas-check] Example: TRITONBENCH_RUN_CONFIG='benchmarks/run_config/....yaml' python -m benchmarks.ptxas_check.run"
        )
        return 1

    print("[ptxas-check] PTXAS Options Compatibility Check")
    print(f"[ptxas-check] PTXAS_OPTIONS: {ptxas_options}")
    print(f"[ptxas-check] Config file: {config_file if config_file else '<not set>'}")
    print(f"[ptxas-check] Extra args: {extra_args}")
    print()

    timestamp, output_dir = setup_output_dir(bm_name="ptxas_check")

    print("[ptxas-check] === Run 1: WITH PTXAS_OPTIONS ===")
    output_dir_with = os.path.join(output_dir, "with_ptxas_options")
    os.mkdir(output_dir_with)
    env_with = os.environ.copy()
    rc1 = run_tritonbench(config_file, extra_args, output_dir_with, env_with)
    print()

    print("[ptxas-check] === Run 2: WITHOUT PTXAS_OPTIONS ===")
    output_dir_without = os.path.join(output_dir, "without_ptxas_options")
    os.mkdir(output_dir_without)
    env_without = os.environ.copy()
    env_without.pop("PTXAS_OPTIONS", None)
    rc2 = run_tritonbench(config_file, extra_args, output_dir_without, env_without)
    print()

    if rc1 != 0:
        print(f"[ptxas-check] WARNING: Run with PTXAS_OPTIONS exited with code {rc1}")
    if rc2 != 0:
        print(
            f"[ptxas-check] WARNING: Run without PTXAS_OPTIONS exited with code {rc2}"
        )

    print("[ptxas-check] === COMPARISON ===")
    print(f"[ptxas-check] Comparing outputs:")
    print(f"[ptxas-check]   With PTXAS_OPTIONS: {output_dir_with}")
    print(f"[ptxas-check]   Without PTXAS_OPTIONS: {output_dir_without}")
    print()

    match, issues = compare_outputs(output_dir_with, output_dir_without)
    print("[ptxas-check] === REPORT ===")
    if match:
        print("[ptxas-check] ✓ SUCCESS: Outputs and configs match!")
        print("[ptxas-check] PTXAS_OPTIONS does not affect benchmark results.")
        return 0
    else:
        print("[ptxas-check] ✗ FAILURE: Outputs or configs differ!")
        print("[ptxas-check] Issues found:")
        for issue in issues:
            print(f"[ptxas-check]   - {issue}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
