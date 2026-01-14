import os
import shutil
import subprocess
from pathlib import Path

from ..python_utils import get_pip_cmd


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CURRENT_DIR = Path(os.path.abspath(__file__)).parent

QUACK_REPO = "https://github.com/Dao-AILab/quack.git"
QUACK_SHA = "21f1c053b1c35a3aaf20002107d0704f364de10a"

QUACK_INSTALL_PATH = REPO_PATH.joinpath(".install")
BUILD_CONSTRAINTS_FILE = REPO_PATH.joinpath("build", "constraints.txt")


def install_quack():
    QUACK_INSTALL_PATH.mkdir(parents=True, exist_ok=True)
    constraints_parameters = ["-c", str(BUILD_CONSTRAINTS_FILE.resolve())]
    quack_path = QUACK_INSTALL_PATH.joinpath("quack")
    if quack_path.exists():
        shutil.rmtree(quack_path)
    git_clone_cmd = ["git", "clone", QUACK_REPO]
    subprocess.check_call(git_clone_cmd, cwd=QUACK_INSTALL_PATH)
    git_checkout_cmd = ["git", "checkout", QUACK_SHA]
    subprocess.check_call(git_checkout_cmd, cwd=quack_path)
    install_quack_cmd = (
        get_pip_cmd() + ["install", "-e", ".[dev]"] + constraints_parameters
    )
    subprocess.check_call(install_quack_cmd, cwd=quack_path)
