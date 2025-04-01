import os
import subprocess
import sys

REQUIREMENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

def install_requirements(requirements_txt: str):
    # ignore dependencies to bypass reinstalling pytorch stable version
    cmd = ["pip", "install", "-r", requirements_txt, "--no-deps"]
    subprocess.check_call(cmd)

def check_install():
    cmd = [sys.executable, "-c", "import tilelang"]
    subprocess.check_call(cmd)

def install_tile():
    install_requirements(REQUIREMENTS_FILE)
    check_install()
