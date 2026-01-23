
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

KERNEL_METADATA_PATH = os.path.join(CURRENT_DIR, "oss_cuda_kernels.yaml")
BACKWARD_METADATA_PATH = os.path.join(CURRENT_DIR, "backward_operators.yaml")
DTYPE_METADATA_PATH = os.path.join(CURRENT_DIR, "dtype_operators.yaml")

SKIP_DTYPE = ["bypass", "fp8", "int4"]

def get_benchmark_config_with_tags(tags: List[str]) -> Dict[str, Any]:
    """Return benchmark config dict with any of these tags"""
    with open(KERNEL_METADATA_PATH, "r") as f:
        operators = yaml.safe_load(f)
    with open(BACKWARD_METADATA_PATH, "r") as f:
        backwards = yaml.safe_load(f)
    with open(DTYPE_METADATA_PATH, "r") as f:
        dtype = yaml.safe_load(f)

    result_dict = {}
    for op, backend in operators.items():
        for backend_name, backend_tags in backend.items():
            if "tags" in backend_tags and any(t in backend["tags"] for t in tags):
                dtype_prefix = dtype[op] if op in dtype and dtype[op] not in SKIP_DTYPE else ""
                benchmark_prefix = f"{dtype_prefix}_{op}_{backend_name}"
                benchmark_name = f"{benchmark_prefix}_fwd"
                result_dict[benchmark_name] = {}
                result_dict[benchmark_name]["op"] = op
                result_dict[benchmark_name]["args"] = ["--op", op, "--only", backend_name]
                if op in backwards:
                    benchmark_name = f"{benchmark_prefix}_bwd"
                    result_dict[benchmark_name] = backend_tags
                    result_dict[benchmark_name]["op"] = op
                    result_dict[benchmark_name]["args"] = ["--op", op, "--only", backend_name, "--bwd"]
    return result_dict
