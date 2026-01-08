import os
import subprocess

from pathlib import Path
from ..python_utils import get_pip_cmd, pip_install_requirements

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
AITER_PATH = REPO_PATH.joinpath("submodules", "aiter")


def install_aiter():
    pip_install_requirements("requirements.txt", current_dir=CURRENT_DIR)
    cmd = get_pip_cmd() + ["install", "-e", "."]
    subprocess.check_call(cmd, cwd=AITER_PATH)
