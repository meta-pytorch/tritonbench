import os
import shutil
import subprocess

from pathlib import Path


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
KRAKEN_INSTALL_PATH = REPO_PATH.joinpath(".install")
KRAKEN_REPO = "https://github.com/meta-pytorch/kraken.git"
KRAKEN_COMMIT = "693f252a3ec39309703e65ae47d0de144adfaeac"


def install_kraken():
    KRAKEN_INSTALL_PATH.mkdir(parents=True, exist_ok=True)
    KRAKEN_PATH = KRAKEN_INSTALL_PATH.joinpath("kraken")
    if KRAKEN_PATH.exists():
        shutil.rmtree(KRAKEN_PATH)
    git_clone_cmd = ["git", "clone", KRAKEN_REPO]
    subprocess.check_call(git_clone_cmd, cwd=KRAKEN_INSTALL_PATH)
    git_checkout_cmd = ["git", "checkout", KRAKEN_COMMIT]
    subprocess.check_call(git_checkout_cmd, cwd=KRAKEN_PATH)
    install_requirements_cmd = [
        "pip",
        "install",
        "-r",
        "requirements.txt",
    ]
    subprocess.check_call(install_requirements_cmd, cwd=KRAKEN_PATH)
    install_helion_cmd = ["pip", "install", "-e", ".", "-r", "requirements.txt"]
    subprocess.check_call(install_helion_cmd, cwd=KRAKEN_PATH)
