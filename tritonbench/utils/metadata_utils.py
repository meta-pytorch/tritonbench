import yaml
from tritonbench.utils.path_utils import REPO_PATH

METADATA_PATH = REPO_PATH.joinpath("tritonbench/metadata")

METADATA_NAME_MAPPING = {
    "aten": "aten_operators.yaml",
    "backward": "backward_operators.yaml",
    "baseline": "baseline_operators.yaml",
    "dtype": "dtype_operators.yaml",
    "cuda_kernels": "oss_cuda_kernels.yaml",
    "tflops": "tflops_operators.yaml",
}


def get_metadata(name: str):
    assert name in METADATA_NAME_MAPPING, f"Unknown metadata name: {name}"
    with open(METADATA_PATH.joinpath(METADATA_NAME_MAPPING[name])) as f:
        return yaml.safe_load(f)
