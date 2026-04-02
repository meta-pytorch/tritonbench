import os
import shutil
import subprocess
from pathlib import Path

from ..python_utils import get_pip_cmd


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
HSTU_INSTALL_PARENT = REPO_PATH.joinpath(".install")
HSTU_PATH = HSTU_INSTALL_PARENT.joinpath("hstu")
HSTU_REPO = "https://github.com/facebookresearch/generative-recommenders.git"
HSTU_COMMIT = "adfd9bb688237bad6aa88e001e6b0e94a2778478"
BUILD_CONSTRAINTS_FILE = REPO_PATH.joinpath("build", "constraints.txt")


def install_hstu():
    HSTU_INSTALL_PARENT.mkdir(parents=True, exist_ok=True)
    constraints_parameters = ["-c", str(BUILD_CONSTRAINTS_FILE.resolve())]
    if HSTU_PATH.exists():
        shutil.rmtree(HSTU_PATH)
    git_clone_cmd = ["git", "clone", HSTU_REPO, HSTU_PATH.name]
    subprocess.check_call(git_clone_cmd, cwd=HSTU_INSTALL_PARENT)
    git_checkout_cmd = ["git", "checkout", HSTU_COMMIT]
    subprocess.check_call(git_checkout_cmd, cwd=HSTU_PATH)
    git_submodule_update_cmd = ["git", "submodule", "update", "--init", "--recursive"]
    subprocess.check_call(git_submodule_update_cmd, cwd=HSTU_PATH)
