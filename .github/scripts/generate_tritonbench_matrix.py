#!/usr/bin/env python3

import argparse
import json
import os
import sys


BENCHMARK_CONFIG = {
    "nightly": {
        "triton_channels": ["meta-triton", "triton-main"],
        "runners": ["h100", "mi350"],
        "manual_only": False,
    },
    "compile_time": {
        "triton_channels": ["triton-main"],
        "runners": ["h100", "mi350"],
        "manual_only": False,
    },
    "tlx": {
        "triton_channels": ["meta-triton"],
        "runners": ["h100"],
        "manual_only": False,
    },
    "timing_accuracy": {
        "triton_channels": ["meta-triton"],
        "runners": ["h100"],
        "manual_only": True,
    },
}

CI_BENCHMARKS = ["nightly", "compile_time", "tlx"]
RUNNER_FULL_NAMES = {
    "h100": "gcp-h100-runner",
    "mi350": "amd-mi350-runner",
}
INFRA_TRIGGER_PATHS = {
    ".github/scripts/generate_tritonbench_matrix.py",
    ".github/workflows/benchmark.yml",
    ".github/workflows/_linux-benchmark.yml",
}
SUPPORTED_RUNNERS = {"h100", "mi350", "all"}
SUPPORTED_TRITON_CHANNELS = {"meta-triton", "triton-main"}


def parse_csv(raw: str) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_runners(runners: list[str]) -> list[str]:
    if not runners or "all" in runners:
        return []
    return runners


def normalize_triton_channels(triton_channels: list[str]) -> list[str]:
    if not triton_channels or "all" in triton_channels:
        return []
    return triton_channels


def validate_requested_values(
    benchmarks: list[str],
    triton_channels: list[str],
    runners: list[str],
) -> None:
    unknown_benchmarks = [name for name in benchmarks if name not in BENCHMARK_CONFIG]
    if unknown_benchmarks:
        raise ValueError(
            f"Unsupported benchmarks: {', '.join(sorted(unknown_benchmarks))}"
        )

    unknown_channels = [
        channel for channel in triton_channels if channel not in SUPPORTED_TRITON_CHANNELS
    ]
    if unknown_channels:
        raise ValueError(
            f"Unsupported triton channels: {', '.join(sorted(unknown_channels))}"
        )

    unknown_runners = [runner for runner in runners if runner not in SUPPORTED_RUNNERS]
    if unknown_runners:
        raise ValueError(f"Unsupported runners: {', '.join(sorted(unknown_runners))}")


def benchmarks_from_pr_changes(changed_files: list[str]) -> list[str]:
    if not changed_files:
        return CI_BENCHMARKS

    changed = set(changed_files)
    if changed & INFRA_TRIGGER_PATHS:
        return CI_BENCHMARKS

    benchmarks: list[str] = []
    for benchmark in CI_BENCHMARKS:
        prefix = f"benchmarks/{benchmark}/"
        if any(path.startswith(prefix) for path in changed):
            benchmarks.append(benchmark)
    return benchmarks


def select_benchmarks(
    requested_benchmarks: list[str], event_name: str, changed_files: list[str]
) -> list[str]:
    if requested_benchmarks:
        benchmarks = requested_benchmarks
    elif event_name == "schedule":
        benchmarks = CI_BENCHMARKS
    elif event_name == "pull_request":
        benchmarks = benchmarks_from_pr_changes(changed_files)
    elif event_name == "workflow_dispatch":
        benchmarks = ["nightly"]
    else:
        benchmarks = CI_BENCHMARKS

    if event_name != "workflow_dispatch":
        benchmarks = [
            benchmark
            for benchmark in benchmarks
            if not BENCHMARK_CONFIG[benchmark]["manual_only"]
        ]

    return benchmarks


def filter_dimensions(
    benchmark: str,
    test_type: str,
    requested_channels: list[str],
    requested_runners: list[str],
) -> list[dict[str, str]]:
    config = BENCHMARK_CONFIG[benchmark]
    triton_channels = list(config["triton_channels"])
    runners = list(config["runners"])

    if benchmark == "timing_accuracy" and test_type != "periodic":
        return []

    if test_type == "abtest":
        if benchmark not in {"nightly", "compile_time"}:
            return []
        triton_channels = ["triton-main"]

    if requested_channels:
        requested_channel_set = set(requested_channels)
        triton_channels = [
            channel for channel in triton_channels if channel in requested_channel_set
        ]
    if requested_runners:
        requested_runner_set = set(requested_runners)
        runners = [runner for runner in runners if runner in requested_runner_set]

    matrix_entries = []
    for triton_channel in triton_channels:
        for runner in runners:
            matrix_entries.append(
                {
                    "benchmark": benchmark,
                    "triton_channel": triton_channel,
                    "runner": runner,
                    "runner_full_name": RUNNER_FULL_NAMES[runner],
                }
            )
    return matrix_entries


def to_matrix(entries: list[dict[str, str]]) -> str:
    return json.dumps({"include": entries}, separators=(",", ":"))


def write_output(name: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")
    else:
        print(f"{name}={value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--triton-channels", default="")
    parser.add_argument("--runners", default="")
    parser.add_argument("--test-type", default="periodic")
    parser.add_argument("--event-name", default="")
    parser.add_argument("--changed-files", default="")
    args = parser.parse_args()

    requested_benchmarks = parse_csv(args.benchmarks)
    requested_triton_channels = normalize_triton_channels(
        parse_csv(args.triton_channels)
    )
    requested_runners = normalize_runners(parse_csv(args.runners))
    changed_files = parse_csv(args.changed_files)

    validate_requested_values(
        requested_benchmarks, requested_triton_channels, requested_runners
    )

    benchmarks = select_benchmarks(
        requested_benchmarks, args.event_name, changed_files
    )

    full_matrix_entries: list[dict[str, str]] = []
    for benchmark in benchmarks:
        full_matrix_entries.extend(
            filter_dimensions(
                benchmark,
                args.test_type,
                requested_triton_channels,
                requested_runners,
            )
        )

    write_output("benchmark_matrix", to_matrix(full_matrix_entries))
    write_output("has_benchmarks", str(bool(full_matrix_entries)).lower())

    if not full_matrix_entries:
        sys.stderr.write("No benchmark matrix entries were generated.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
