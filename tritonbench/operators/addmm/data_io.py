import argparse
from typing import List


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TritonBench Addmm operator Benchmark")
    parser.add_argument("--m", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--input", type=str)
    parser.add_argument("--col-major", type=bool, default=False)
    parser.add_argument("--large-k-shapes", type=bool, default=False)
    parser.add_argument("--bias-1D-y", type=bool, default=False)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(args)
    return args
