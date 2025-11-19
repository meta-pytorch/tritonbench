#!/usr/bin/env python3
"""
Simple tagging coverage analyzer for TritonBench.

Analyzes which operators have tags and which don't, and shows statistics.
"""

import argparse

import yaml


def analyze_coverage(yaml_file, verbose=False):
    """Analyze tagging coverage from YAML metadata file"""

    # Load metadata
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f) or {}

    # Statistics
    total_ops = 0
    total_backends = 0
    tagged_backends = 0
    null_backends = 0

    # Per-operator stats
    op_stats = {}

    # Untagged backends
    untagged = []

    # Process each operator
    for op_name, backends in data.items():
        if backends is None:
            continue

        total_ops += 1
        op_tagged = 0
        op_total = 0
        op_null = 0

        for backend_name, info in backends.items():
            op_total += 1
            total_backends += 1

            if info is None:
                null_backends += 1
                op_null += 1
                untagged.append(f"{op_name}.{backend_name}")
            elif "tags" in info and info["tags"]:
                tagged_backends += 1
                op_tagged += 1
            else:
                untagged.append(f"{op_name}.{backend_name}")

        op_stats[op_name] = {
            "total": op_total,
            "tagged": op_tagged,
            "null": op_null,
            "coverage": (op_tagged / op_total * 100) if op_total > 0 else 0,
        }

    # Print results
    print("=" * 70)
    print("  TritonBench Tagging Coverage Analysis")
    print("=" * 70)
    print()

    # Overall stats
    overall_coverage = (
        (tagged_backends / total_backends * 100) if total_backends > 0 else 0
    )

    print(f"Total Operators:     {total_ops}")
    print(f"Total Backends:      {total_backends}")
    print(f"Tagged Backends:     {tagged_backends}")
    print(f"Untagged Backends:   {null_backends}")
    print(f"Overall Coverage:    {overall_coverage:.1f}%")
    print()

    # Per-operator breakdown
    print("Per-Operator Coverage:")
    print("-" * 70)
    print(f"{'Operator':<30} {'Total':>8} {'Tagged':>8} {'Coverage':>10}")
    print("-" * 70)

    # Sort by coverage (ascending)
    for op_name in sorted(op_stats.keys(), key=lambda x: op_stats[x]["coverage"]):
        stats = op_stats[op_name]
        coverage_str = f"{stats['coverage']:.1f}%"
        print(
            f"{op_name:<30} {stats['total']:>8} {stats['tagged']:>8} {coverage_str:>10}"
        )

    print()

    # Show untagged backends
    if untagged:
        print(f"Untagged Backends ({len(untagged)}):")
        print("-" * 70)

        # Show all if verbose, otherwise show first 20
        if verbose:
            for backend in untagged:
                print(f"  • {backend}")
        else:
            for backend in untagged[:20]:
                print(f"  • {backend}")
            if len(untagged) > 20:
                print(f"  ... and {len(untagged) - 20} more")
                print(f"\n  Use --verbose to see all untagged backends")

    print()
    print("=" * 70)

    return {
        "total_ops": total_ops,
        "total_backends": total_backends,
        "tagged_backends": tagged_backends,
        "coverage": overall_coverage,
        "untagged": untagged,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TorchBench tagging coverage")
    # example: tritonbench/metadata/oss_cuda_kernels.yaml
    parser.add_argument("yaml_file", help="Path to tagging metadata YAML file")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show all untagged backends (default: show first 20)",
    )
    args = parser.parse_args()

    analyze_coverage(args.yaml_file, verbose=args.verbose)
