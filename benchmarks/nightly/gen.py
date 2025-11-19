# Generate the TRITONBENCH_CONFIG autogen.yaml for nightly benchmark
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent

METADATA_PATH = REPO_PATH.joinpath("tritonbench/metadata/")
CURRENT_PATH = Path(os.path.abspath(__file__)).parent
OUTPUT_PATH = CURRENT_PATH.joinpath("autogen.yaml")


def get_metadata(name: str, path: Path = METADATA_PATH) -> Any:
    fpath = os.path.join(path, f"{name}.yaml")
    with open(fpath, "r") as f:
        return yaml.safe_load(f)


def get_triton_ops(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    triton_ops = {}
    for op in metadata:
        for backend in metadata[op]:
            if (
                metadata[op][backend]
                and metadata[op][backend]["tags"]
                and "triton" in metadata[op][backend]["tags"]
            ):
                if op not in triton_ops:
                    triton_ops[op] = []
                triton_ops[op].append(backend)
    print(triton_ops)
    return triton_ops


TRITON_OPS: dict[str, list[str]] = get_triton_ops(get_metadata("oss_cuda_kernels"))
DTYPE_OPS: Dict[str, str] = get_metadata("dtype_operators")
TFLOPS_OPS: List[str] = get_metadata("tflops_operators")
BASELINE_OPS: Dict[str, str] = get_metadata("baseline_operators")
BWD_OPS: List[str] = get_metadata("backward_operators")

# Manually overridden options
MANUAL_OPTIONS = get_metadata("manual", path=CURRENT_PATH)


def _has_meaningful_baseline(op: str):
    return op in BASELINE_OPS and not (
        BASELINE_OPS[op] in TRITON_OPS[op] and len(TRITON_OPS[op]) == 1
    )


def gen_run(operators: Dict[str, List[str]]) -> Dict[str, Any]:
    out = {}
    for op in operators:
        dtype = (
            DTYPE_OPS[op]
            if not DTYPE_OPS[op] == "fp8" and not DTYPE_OPS[op] == "bypass"
            else ""
        )
        run_name = f"{dtype}_{op}_fwd" if dtype else f"{op}_fwd"
        cmd = ["--op", op]
        # add metrics
        metrics = ["latency"]
        if op in TFLOPS_OPS:
            metrics.append("tflops")
        if _has_meaningful_baseline(op):
            cmd.extend(["--baseline", BASELINE_OPS[op]])
            metrics.append("speedup")
        cmd.extend(["--metrics", ",".join(metrics)])
        # add backends
        run_backends = TRITON_OPS[op]
        if _has_meaningful_baseline(op) and BASELINE_OPS[op] not in run_backends:
            run_backends.append(BASELINE_OPS[op])
        cmd.extend(["--only", ",".join(run_backends)])
        out[run_name] = {}
        out[run_name]["args"] = " ".join(cmd)
        # add backward run if applicable
        if op in BWD_OPS:
            bwd_run_name = f"{dtype}_{op}_bwd" if dtype else f"{op}_bwd"
            out[bwd_run_name] = {}
            out[bwd_run_name]["args"] = " ".join(cmd) + " --bwd"
    return out


def add_manual_benchmarks(
    run_configs: Dict[str, Any], options: Dict[str, Any]
) -> Dict[str, Any]:
    disabled = options.get("disabled", [])
    extra_args = options.get("extra_args", {})
    for benchmark in disabled:
        if benchmark in run_configs:
            run_configs[benchmark]["disabled"] = True
    for benchmark in extra_args:
        run_configs[benchmark]["args"] = extra_args[benchmark]["args"]
    for benchmark, benchmark_config in options.get("enabled", {}).items():
        run_configs[benchmark] = benchmark_config.copy()
    return run_configs


def run():
    runs = gen_run(TRITON_OPS)
    # add manual benchmarks and configs
    add_manual_benchmarks(runs, MANUAL_OPTIONS)
    with open(OUTPUT_PATH, "w") as f:
        yaml.safe_dump(runs, f, sort_keys=False)


if __name__ == "__main__":
    run()
