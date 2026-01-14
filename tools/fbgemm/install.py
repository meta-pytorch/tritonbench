import os
import subprocess
import sys
from pathlib import Path

# requires torch
from ..cuda_utils import get_toolkit_version_from_torch
from ..python_utils import get_pip_cmd, pip_install_requirements

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
FBGEMM_INSTALL_PATH = REPO_PATH.joinpath(".install", "FBGEMM")
FBGEMM_REPO = "https://github.com/pytorch/FBGEMM"
FBGEMM_COMMIT = "7cbaff699e784736b5650dfa51d62c251b028717"


def install_fbgemm(prebuilt=True):
    if prebuilt:
        install_prebuilt_fbgemm()
    else:
        install_build_fbgemm()


def install_prebuilt_fbgemm():
    toolkit_version = get_toolkit_version_from_torch()
    cmd = get_pip_cmd() + [
        "install",
        "--pre",
        "fbgemm-gpu",
        "-i",
        f"https://download.pytorch.org/whl/nightly/{toolkit_version}",
    ]
    subprocess.check_call(cmd)


def checkout_fbgemm():
    git_clone_cmd = ["git", "clone", FBGEMM_REPO]
    subprocess.check_call(git_clone_cmd, cwd=FBGEMM_INSTALL_PATH)
    fbgemm_repo_path = FBGEMM_INSTALL_PATH.joinpath("FBGEMM")
    git_checkout_cmd = ["git", "checkout", FBGEMM_COMMIT]
    subprocess.check_call(git_checkout_cmd, cwd=fbgemm_repo_path)
    git_submodule_checkout_cmd = [
        "git",
        "submodules",
        "update",
        "--init",
        "--recursive",
    ]
    subprocess.check_call(git_submodule_checkout_cmd, cwd=fbgemm_repo_path)


def install_build_fbgemm(genai=True):
    fbgemm_repo_path = FBGEMM_INSTALL_PATH.joinpath("FBGEMM")
    if not os.path.exists(fbgemm_repo_path):
        checkout_fbgemm()
    pip_install_requirements(
        "requirements.txt",
        current_dir=str(fbgemm_repo_path.joinpath("fbgemm_gpu").resolve()),
    )
    # Build target H100(9.0, 9.0a) and blackwell (10.0, 12.0)
    extra_envs = os.environ.copy()
    cmd = [
        sys.executable,
        "setup.py",
        "install",
        "--build-target=cuda",
        "-DTORCH_CUDA_ARCH_LIST=9.0;9.0a;10.0;12.0",
    ]
    subprocess.check_call(cmd, cwd=str(fbgemm_repo_path.resolve()), env=extra_envs)


def test_fbgemm():
    print("Checking fbgemm_gpu installation...", end="")
    # test triton
    cmd = [
        sys.executable,
        "-c",
        "from fbgemm_gpu.quantize_utils import fp32_to_mx4",
    ]
    subprocess.check_call(cmd)
    print("OK")
