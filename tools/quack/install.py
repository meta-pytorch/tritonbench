import os
import shutil
import subprocess

from pathlib import Path


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CURRENT_DIR = Path(os.path.abspath(__file__)).parent

QUACK_REPO = "https://github.com/Dao-AILab/quack.git"
QUACK_SHA = "bceb632dbac9bb0b55d48a7ed3ad204bd952fcb2"

QUACK_INSTALL_PATH = REPO_PATH.joinpath(".install")


def install_quack():
    cmd = ["pip", "install", "-e", "."]
    subprocess.check_call(cmd, cwd=QUACK_PATH)


def install_quack():
    QUACK_INSTALL_PATH.mkdir(parents=True, exist_ok=True)
    quack_path = QUACK_INSTALL_PATH.joinpath("quack")
    if quack_path.exists():
        shutil.rmtree(quack_path)
    git_clone_cmd = ["git", "clone", QUACK_REPO]
    subprocess.check_call(git_clone_cmd, cwd=QUACK_INSTALL_PATH)
    git_checkout_cmd = ["git", "checkout", QUACK_SHA]
    subprocess.check_call(git_checkout_cmd, cwd=quack_path)
    install_helion_cmd = ["pip", "install", "-e", ".[dev]"]
    subprocess.check_call(install_helion_cmd, cwd=quack_path)
