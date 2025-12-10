import os
import shutil
import subprocess

from pathlib import Path


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
HELION_INSTALL_PATH = REPO_PATH.joinpath(".install")
HELION_REPO = "https://github.com/pytorch/helion.git"
HEION_COMMIT = "51580b43bd65978a28b6e5bcd6f625485f02cba1"
BUILD_CONSTRAINTS_FILE = REPO_PATH.joinpath("build", "constraints.txt")


def install_helion():
    HELION_INSTALL_PATH.mkdir(parents=True, exist_ok=True)
    HELION_PATH = HELION_INSTALL_PATH.joinpath("helion")
    constraints_parameters = ["-c", str(BUILD_CONSTRAINTS_FILE.resolve())]
    if HELION_PATH.exists():
        shutil.rmtree(HELION_PATH)
    git_clone_cmd = ["git", "clone", HELION_REPO]
    subprocess.check_call(git_clone_cmd, cwd=HELION_INSTALL_PATH)
    git_checkout_cmd = ["git", "checkout", HEION_COMMIT]
    subprocess.check_call(git_checkout_cmd, cwd=HELION_PATH)
    install_tritonbench_cmd = ["pip", "install", "--no-deps", "-e", "."] + constraints_parameters
    subprocess.check_call(install_tritonbench_cmd, cwd=REPO_PATH)
    install_helion_cmd = ["pip", "install", "-e", ".[dev]"] + constraints_parameters
    subprocess.check_call(install_helion_cmd, cwd=HELION_PATH)
