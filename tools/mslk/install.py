import os
import subprocess
import sys
from pathlib import Path

from tritonbench.utils.env_utils import is_hip

# requires torch
from ..cuda_utils import get_toolkit_version_from_torch

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
MSLK_INSTALL_PATH = REPO_PATH.joinpath(".install", "MSLK")
MSLK_REPO = "https://github.com/meta-pytorch/MSLK"
MSLK_COMMIT = "7ad994918da19fe2d148533670f472ad661c54ce"


def install_mslk(prebuilt=True):
    if prebuilt:
        install_prebuilt_mslk()
    else:
        install_build_mslk()


def install_prebuilt_mslk():
    toolkit_version = get_toolkit_version_from_torch()
    cmd = [
        "pip",
        "install",
        "--pre",
        "mslk",
        "-i",
        f"https://download.pytorch.org/whl/nightly/{toolkit_version}",
    ]
    subprocess.check_call(cmd)


def checkout_mslk():
    git_clone_cmd = ["git", "clone", MSLK_REPO]
    subprocess.check_call(git_clone_cmd, cwd=MSLK_INSTALL_PATH)
    mslk_repo_path = MSLK_INSTALL_PATH.joinpath("MSLK")
    git_checkout_cmd = ["git", "checkout", MSLK_COMMIT]
    subprocess.check_call(git_checkout_cmd, cwd=mslk_repo_path)
    git_submodule_checkout_cmd = [
        "git",
        "submodule",
        "update",
        "--init",
        "--recursive",
    ]
    subprocess.check_call(git_submodule_checkout_cmd, cwd=mslk_repo_path)


def install_build_mslk():
    mslk_repo_path = MSLK_INSTALL_PATH.joinpath("MSLK")
    if not os.path.exists(mslk_repo_path):
        checkout_mslk()
    cmd = ["pip", "install", "-r", "requirements.txt"]
    subprocess.check_call(
        cmd, cwd=str(mslk_repo_path)
    )
    # Build target H100(9.0, 9.0a) and blackwell (10.0, 12.0)
    extra_envs = os.environ.copy()
    if not is_hip():
        cmd = [
            sys.executable,
            "setup.py",
            "install",
            "--build-target=default",
            "-DTORCH_CUDA_ARCH_LIST=9.0;9.0a;10.0;12.0",
        ]
    elif is_hip():
        # build for MI300(gfx942) and MI350(gfx950)
        current_conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        cmd = [
            "bash",
            "-c",
            f'. .github/scripts/setup_env.bash; integration_mslk_build_and_install {current_conda_env} default/rocm "{mslk_repo_path}"',
        ]
        extra_envs["BUILD_ROCM_VERSION"] = "7.0"
        subprocess.check_call(
            cmd, cwd=str(mslk_repo_path.resolve()), env=extra_envs
        )
        return
    subprocess.check_call(cmd, cwd=str(mslk_repo_path.resolve()), env=extra_envs)


def test_mslk():
    print("Checking mslk installation...", end="")
    # test triton
    cmd = [
        sys.executable,
        "-c",
        "import mslk.gemm.triton.fp8_gemm as fp8_gemm",
    ]
    subprocess.check_call(cmd)
    # test mslk
    cmd = [sys.executable, "-c", "import mslk"]
    subprocess.check_call(cmd)
    print("OK")
