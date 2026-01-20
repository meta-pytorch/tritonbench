#!/usr/bin/env python

import logging
import os
from argparse import ArgumentParser
from typing import Any, Dict


logging.basicConfig(level=logging.INFO)

# This mapping is needed to find out the platform of the runner
RUNNER_TO_PLATFORM_MAPPING = {
    "gcp-h100-runner": "cuda",
    "amd-mi350-runner": "rocm",
}

# TritonBench benchmarks
TRITON_CHANNELS = set(
    [
        "triton-main",
        "meta-triton"
    ]
)


def set_output(name: str, val: Any) -> None:
    """
    Set the output value to be used by other GitHub jobs.

    Args:
        name (str): The name of the output variable.
        val (Any): The value to set for the output variable.

    Example:
        set_output("benchmark_matrix", {"include": [...]})
    """
    github_output = os.getenv("GITHUB_OUTPUT")

    if not github_output:
        print(f"::set-output name={name}::{val}")
        return

    with open(github_output, "a") as env:
        env.write(f"{name}={val}\n")


def parse_args() -> Any:
    parser = ArgumentParser("Generate TritonBench benchmark CI matrix")

    parser.add_argument(
        "--triton-channel",
        type=str,
        choices=TRITON_CHANNELS,
        help="the triton channels to bisect. Choices include triton-main,meta-triton. Required.",
        required=True,
    )
    parser.add_argument(
        "--runner",
        type=str,
        help="the runner to run bisect. Required.",
        required=True,
    )

    return parser.parse_args()

def generate_benchmark_matrix(triton_channel: str, runner: str) -> Dict[str, Any]:
    benchmark_matrix: Dict[str, Any] = {
        "include": [],
    }

    runner_args = None
    for k in RUNNER_TO_PLATFORM_MAPPING.keys():
        if runner.lower() in k:
            runner_args = k

    assert runner_args is not None, f"Unknown runner {runner}"

    # Gather all possible benchmarks
    benchmark_matrix["include"] = [{
        "runner": runner_args,
        "triton_channel": triton_channel,
    }]

    return benchmark_matrix


def main() -> None:
    args = parse_args()
    runner = args.runner
    triton_channel = args.triton_channel
    benchmark_matrix = generate_benchmark_matrix(triton_channel, runner)
    print(benchmark_matrix)
    set_output("benchmark_matrix", benchmark_matrix)


if __name__ == "__main__":
    main()
