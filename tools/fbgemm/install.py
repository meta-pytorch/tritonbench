import os
import subprocess
import sys

from pathlib import Path

# requires torch
from ..cuda_utils import get_toolkit_version_from_torch

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
FBGEMM_INSTALL_PATH = REPO_PATH.joinpath(".install", "FBGEMM")
FBGEMM_REPO = "https://github.com/pytorch/FBGEMM"
FBGEMM_COMMIT = "7cbaff699e784736b5650dfa51d62c251b028717"


def install_fbgemm(genai=True, prebuilt=True):
    if prebuilt:
        install_prebuilt_fbgemm(genai)
    else:
        install_build_fbgemm(genai)


def install_prebuilt_fbgemm(genai=True):
    assert genai, "in prebuilt fbgemm package, we only support genai now"
    toolkit_version = get_toolkit_version_from_torch()
    cmd = [
        "pip",
        "install",
        "--pre",
        "fbgemm-gpu-genai",
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
    cmd = ["pip", "install", "-r", "requirements.txt"]
    subprocess.check_call(
        cmd, cwd=str(fbgemm_repo_path.joinpath("fbgemm_gpu").resolve())
    )
    # Build target H100(9.0, 9.0a) and blackwell (10.0, 12.0)
    extra_envs = os.environ.copy()
    if genai:
        if not is_hip():
            cmd = [
                sys.executable,
                "setup.py",
                "install",
                "--build-target=genai",
                "-DTORCH_CUDA_ARCH_LIST=9.0;9.0a;10.0;12.0",
            ]
        elif is_hip():
            # build for MI300(gfx942) and MI350(gfx950)
            current_conda_env = os.environ.get("CONDA_DEFAULT_ENV")
            cmd = [
                "bash",
                "-c",
                f'. .github/scripts/setup_env.bash; integration_fbgemm_gpu_build_and_install {current_conda_env} genai/rocm "{fbgemm_repo_path}"',
            ]
            extra_envs["BUILD_ROCM_VERSION"] = "7.0"
            subprocess.check_call(
                cmd, cwd=str(fbgemm_repo_path.resolve()), env=extra_envs
            )
            return
    else:
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
        "import fbgemm_gpu.experimental.gemm.triton_gemm.fp8_gemm",
    ]
    subprocess.check_call(cmd)
    # test genai (cutlass or ck)
    cmd = [sys.executable, "-c", "import fbgemm_gpu.experimental.gen_ai"]
    subprocess.check_call(cmd)
    print("OK")
