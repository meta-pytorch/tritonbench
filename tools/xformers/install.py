import os
import subprocess
import sys
from pathlib import Path

from ..python_utils import get_pip_cmd

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
PATCH_DIR = str(REPO_PATH.joinpath("submodules", "xformers").absolute())
PATCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xformers.patch")


def patch_xformers():
    try:
        subprocess.check_output(
            [
                "patch",
                "-p1",
                "--forward",
                "-i",
                PATCH_FILE,
                "-r",
                "/tmp/rej",
            ],
            cwd=PATCH_DIR,
        )
    except subprocess.SubprocessError as e:
        output_str = str(e.output)
        if "previously applied" in output_str:
            return
        else:
            print(str(output_str))
            sys.exit(1)


def install_xformers():
    patch_xformers()
    os_env = os.environ.copy()
    os_env["TORCH_CUDA_ARCH_LIST"] = "8.0;9.0;9.0a"
    # Do not use built-in flash_attn
    # Since it will be incompatible with xformers code
    # Instead, use the flash_attn module built-in as xformers submodule
    os_env["XFORMERS_PT_FLASH_ATTN"] = "0"
    XFORMERS_PATH = REPO_PATH.joinpath("submodules", "xformers")
    cmd = get_pip_cmd() + ["install", "-e", XFORMERS_PATH]
    subprocess.check_call(cmd, env=os_env)
