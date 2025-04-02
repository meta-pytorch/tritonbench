import argparse
import os
import re
import subprocess
from pathlib import Path

from .torch_utils import install_pytorch_nightly

# defines the default CUDA version to compile against
DEFAULT_CUDA_VERSION = "12.8"

CUDA_VERSION_MAP = {
    "12.8": {
        "pytorch_url": "cu128",
        "jax": "jax[cuda12]",
    },
}


def detect_cuda_version_with_nvcc(env):
    test_nvcc = ["nvcc", "--version"]
    regex = "release (.*),"
    output = subprocess.check_output(test_nvcc, stderr=subprocess.STDOUT).decode()
    version = re.search(regex, output).groups()[0]
    return version


def prepare_cuda_env(cuda_version: str, dryrun=False):
    assert (
        cuda_version in CUDA_VERSION_MAP
    ), f"Required CUDA version {cuda_version} doesn't exist in {CUDA_VERSION_MAP.keys()}."
    env = os.environ.copy()
    # step 1: setup CUDA path and environment variables
    cuda_path = Path("/").joinpath("usr", "local", f"cuda-{cuda_version}")
    assert (
        cuda_path.exists() and cuda_path.is_dir()
    ), f"Expected CUDA Library path {cuda_path} doesn't exist."
    cuda_path_str = str(cuda_path.resolve())
    env["CUDA_ROOT"] = cuda_path_str
    env["CUDA_HOME"] = cuda_path_str
    env["PATH"] = f"{cuda_path_str}/bin:{env['PATH']}"
    env["CMAKE_CUDA_COMPILER"] = str(cuda_path.joinpath("bin", "nvcc").resolve())
    env["LD_LIBRARY_PATH"] = (
        f"{cuda_path_str}/lib64:{cuda_path_str}/extras/CUPTI/lib64:{env['LD_LIBRARY_PATH']}"
    )
    if dryrun:
        print(f"CUDA_HOME is set to {env['CUDA_HOME']}")

    # step 2: test call to nvcc to confirm the version is correct
    nvcc_version = detect_cuda_version_with_nvcc(env=env)
    assert (
        nvcc_version == cuda_version
    ), f"Expected CUDA version {cuda_version}, getting nvcc test result {nvcc_version}"

    return env


def setup_cuda_softlink(cuda_version: str):
    assert (
        cuda_version in CUDA_VERSION_MAP
    ), f"Required CUDA version {cuda_version} doesn't exist in {CUDA_VERSION_MAP.keys()}."
    cuda_path = Path("/").joinpath("usr", "local", f"cuda-{cuda_version}")
    assert (
        cuda_path.exists() and cuda_path.is_dir()
    ), f"Expected CUDA Library path {cuda_path} doesn't exist."
    current_cuda_path = Path("/").joinpath("usr", "local", "cuda")
    if current_cuda_path.exists():
        assert (
            current_cuda_path.is_symlink()
        ), "Expected /usr/local/cuda to be a symlink."
        current_cuda_path.unlink()
    os.symlink(str(cuda_path.resolve()), str(current_cuda_path.resolve()))


def install_torch_deps():
    # install other dependencies
    torch_deps = [
        "requests",
        "ninja",
        "pyyaml",
        "setuptools",
        "gitpython",
        "beautifulsoup4",
        "regex",
    ]
    cmd = ["conda", "install", "-y"] + torch_deps
    subprocess.check_call(cmd)
    # conda forge deps
    # ubuntu 22.04 comes with libstdcxx6 12.3.0
    # we need to install the same library version in conda
    conda_deps = ["libstdcxx-ng=12.3.0"]
    cmd = ["conda", "install", "-y", "-c", "conda-forge"] + conda_deps
    subprocess.check_call(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cudaver",
        default=DEFAULT_CUDA_VERSION,
        help="Specify the default CUDA version",
    )
    parser.add_argument(
        "--setup-cuda-softlink",
        action="store_true",
        help="Setup the softlink to /usr/local/cuda",
    )
    parser.add_argument(
        "--install-torch-deps",
        action="store_true",
        help="Install pytorch runtime dependencies",
    )
    parser.add_argument(
        "--install-torch-build-deps",
        action="store_true",
        help="Install pytorch build dependencies",
    )
    parser.add_argument(
        "--install-torch-nightly", action="store_true", help="Install pytorch nightlies"
    )
    parser.add_argument(
        "--check-torch-nightly-version",
        action="store_true",
        help="Validate pytorch nightly package consistency",
    )
    parser.add_argument(
        "--force-date",
        type=str,
        default=None,
        help="Force Pytorch nightly release date version. Date string format: YYmmdd",
    )
    args = parser.parse_args()
    if args.setup_cuda_softlink:
        setup_cuda_softlink(cuda_version=args.cudaver)
    if args.install_torch_deps:
        install_torch_deps()
    if args.install_torch_build_deps:
        from .torch_utils import install_torch_build_deps

        install_torch_deps()
        install_torch_build_deps()
    if args.install_torch_nightly:
        pytorch_cuda_version = CUDA_VERSION_MAP[args.cudaver]["pytorch_url"]
        install_pytorch_nightly(cuda_version=pytorch_cuda_version, env=os.environ)
    if args.check_torch_nightly_version:
        from .torch_utils import check_torch_nightly_version

        assert not args.install_torch_nightly, "Error: Can't run install torch nightly and check version in the same command."
        check_torch_nightly_version(args.force_date)
