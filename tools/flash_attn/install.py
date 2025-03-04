import os
import subprocess
import sys

from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CUR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def install_fa3():
    FA3_PATH = REPO_PATH.joinpath("submodules", "flash-attention", "hopper")
    env = os.environ.copy()
    # limit nvcc memory usage on the CI machine
    env["MAX_JOBS"] = "8"
    env["NVCC_THREADS"] = "1"
    cmd = ["pip", "install", "-e", "."]
    subprocess.check_call(cmd, cwd=str(FA3_PATH.resolve()), env=env)
