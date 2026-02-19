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
import pickle
import sys
from typing import Any, Dict, List, Tuple

import torch

from ..common import setup_output_dir, setup_tritonbench_cwd


setup_tritonbench_cwd()

from tritonbench.utils.run_utils import run_config, run_in_task


def find_pkl_files(path: str) -> List[str]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return []
    return [
        f
        for f in os.listdir(abs_path)
        if os.path.isfile(os.path.join(abs_path, f)) and f.endswith(".pkl")
    ]


def find_stderr_file(path: str) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Directory {path} does not exist")

    stderr_files = [
        f
        for f in os.listdir(abs_path)
        if os.path.isfile(os.path.join(abs_path, f)) and f.endswith("stderr.log")
    ]
    assert len(stderr_files) == 1, (
        f"Expected exactly one stderr file, found {len(stderr_files)}"
    )
    return stderr_files[0]


def load_pickle(filepath: str) -> Any:
    with open(filepath, "rb") as f:
        return pickle.load(f)


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


def load_best_config_from_stderr(file_path: str):
    config = None
    with open(file_path, "r") as f:
        lines = f.readlines()
        for lineno, line in enumerate(lines):
            if "Autotune Choices Stats:" in line:
                config = lines[lineno + 1].strip()
    if not config:
        return None
    config_dict = json.loads(config)
    return config_dict["best_kernel_desc"]


def compare_configs(config_a: Any, config_b: Any) -> bool:
    print(f"data a: {config_a}")
    print(f"data b: {config_b}")
    if config_a is None and config_b is None:
        return True
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

    stderr_files_a = find_stderr_file(dir_a)
    stderr_files_b = find_stderr_file(dir_b)
    assert stderr_files_b == stderr_files_a, (
        f"Expected same stderr files, found {stderr_files_b} and {stderr_files_a}"
    )

    path_a = os.path.join(dir_a, stderr_files_a)
    path_b = os.path.join(dir_b, stderr_files_a)
    data_a = load_best_config_from_stderr(path_a)
    data_b = load_best_config_from_stderr(path_b)

    if not compare_configs(data_a, data_b):
        issues.append(f"Config mismatch in {stderr_files_a}")
        issues.append(f"  With PTXAS_OPTIONS: {data_a}")
        issues.append(f"  Without PTXAS_OPTIONS: {data_b}")
    else:
        print(f"[ptxas-check] {stderr_files_a}: configs match ✓")

    pkl_files_a = set(find_pkl_files(dir_a))
    pkl_files_b = set(find_pkl_files(dir_b))
    common_pkl = pkl_files_a & pkl_files_b
    only_in_a = pkl_files_a - pkl_files_b
    only_in_b = pkl_files_b - pkl_files_a

    if only_in_a:
        issues.append(f"PKL files only in run with PTXAS_OPTIONS: {sorted(only_in_a)}")
    if only_in_b:
        issues.append(
            f"PKL files only in run without PTXAS_OPTIONS: {sorted(only_in_b)}"
        )

    for pkl_file in sorted(common_pkl):
        path_a = os.path.join(dir_a, pkl_file)
        path_b = os.path.join(dir_b, pkl_file)
        data_a = load_pickle(path_a)
        data_b = load_pickle(path_b)
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
    parser.add_argument("--config-file", type=str, help="Config file to use")
    args, extra_args = parser.parse_known_args()

    if "--" in extra_args:
        extra_args.remove("--")

    ptxas_options = os.environ.get("PTXAS_OPTIONS")
    if ptxas_options is None:
        print("[ptxas-check] ERROR: PTXAS_OPTIONS environment variable is not set.")
        print("[ptxas-check] Please set PTXAS_OPTIONS before running this benchmark.")
        print("[ptxas-check] Example: PTXAS_OPTIONS='-v' buck2 run ... -- --op gemm")
        return 1

    print("[ptxas-check] PTXAS Options Compatibility Check")
    print(f"[ptxas-check] PTXAS_OPTIONS: {ptxas_options}")
    print(
        f"[ptxas-check] Config file: {args.config_file if args.config_file else '<not set>'}"
    )
    print(f"[ptxas-check] Extra args: {extra_args}")
    print()

    timestamp, output_dir = setup_output_dir(bm_name="ptxas_check")

    print("[ptxas-check] === Run 1: WITH PTXAS_OPTIONS ===")
    output_dir_with = os.path.join(output_dir, "with_ptxas_options")
    os.mkdir(output_dir_with)
    env_with = os.environ.copy()
    rc1 = run_tritonbench(args.config_file, extra_args, output_dir_with, env_with)
    print()

    print("[ptxas-check] === Run 2: WITHOUT PTXAS_OPTIONS ===")
    output_dir_without = os.path.join(output_dir, "without_ptxas_options")
    os.mkdir(output_dir_without)
    env_without = os.environ.copy()
    env_without.pop("PTXAS_OPTIONS", None)
    rc2 = run_tritonbench(args.config_file, extra_args, output_dir_without, env_without)
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
