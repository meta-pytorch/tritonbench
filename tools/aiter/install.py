import os
import subprocess

from pathlib import Path


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
AITER_PATH = REPO_PATH.joinpath("submodules", "aiter")


def pip_install_requirements():
    cmd = ["pip", "install", "-r", "requirements.txt"]
    subprocess.check_call(cmd, cwd=CURRENT_DIR)


def install_aiter():
    pip_install_requirements()
    cmd = ["pip", "install", "-e", "."]
    subprocess.check_call(cmd, cwd=AITER_PATH)
