import os
import subprocess

from pathlib import Path


REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
HELION_INSTALL_PATH = REPO_PATH.joinpath(".install")
HEION_COMMIT = "1aaba3f33fcbd730ce24a13bcd76d49e0f536ede"

def install_helion():
    HELION_INSTALL_PATH.mkdir(parents=True, exist_ok=True)
    git_clone_cmd = ["git", "clone", "https://github.com/pytorch/helion.git", cwd=HELION_INSTALL_PATH]
    subprocess.check_call(git_clone_cmd)
    HELION_PATH = HELION_INSTALL_PATH.joinpath("helion")
    git_checkout_cmd = ["git", "checkout", HEION_COMMIT, cwd=HELION_PATH]
    subprocess.check_call(git_clone_cmd)
    install_requirements_cmd = ["pip", "install", "-r", "requirements.txt", cwd=HELION_PATH]
    subprocess.check_call(install_requirements_cmd)
    install_helion_cmd = ["pip", "install", "-e", ".'[dev]'", cwd=HELION_PATH]
    subprocess.check_call(install_helion_cmd)
