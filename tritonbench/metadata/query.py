import os
from typing import Any, Dict, List, Optional

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

KERNEL_METADATA_PATH = os.path.join(CURRENT_DIR, "oss_cuda_kernels.yaml")
BACKWARD_METADATA_PATH = os.path.join(CURRENT_DIR, "backward_operators.yaml")
DTYPE_METADATA_PATH = os.path.join(CURRENT_DIR, "dtype_operators.yaml")

SKIP_DTYPE = ["bypass", "fp8", "int4"]


def get_benchmark_config_with_tags(
    tags: List[str], runtime_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Return benchmark config dict with any of these tags"""
    with open(KERNEL_METADATA_PATH, "r") as f:
        operators = yaml.safe_load(f)
        if runtime_metadata is not None:
            for op in runtime_metadata:
                if op not in operators:
                    operators.update({op: runtime_metadata[op]})
                else:
                    operators[op].update(runtime_metadata[op])

    with open(BACKWARD_METADATA_PATH, "r") as f:
        backwards = yaml.safe_load(f)
    with open(DTYPE_METADATA_PATH, "r") as f:
        dtype = yaml.safe_load(f)

    result_dict = {}
    for op, backend in operators.items():
        backend_with_wanted_tags = {
            b
            for b in backend
            if "tags" in backend[b] and any(t in backend[b]["tags"] for t in tags)
        }
        backend_names_with_tags = [b for b in backend_with_wanted_tags]
        if not backend_names_with_tags:
            continue
        dtype_prefix = dtype[op] if op in dtype and dtype[op] not in SKIP_DTYPE else ""
        benchmark_prefix = f"{dtype_prefix}_{op}"
        benchmark_name = f"{benchmark_prefix}_fwd"
        result_dict[benchmark_name] = {}
        result_dict[benchmark_name]["args"] = " ".join(
            ["--op", op, "--only"] + [",".join(backend_names_with_tags)]
        )
        if op in backwards:
            benchmark_name = f"{benchmark_prefix}_bwd"
            result_dict[benchmark_name] = {}
            result_dict[benchmark_name]["args"] = " ".join(
                ["--op", op, "--only"] + [",".join(backend_names_with_tags), "--bwd"]
            )
    return result_dict
