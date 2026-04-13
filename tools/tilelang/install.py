import os
import subprocess
import sys

from tritonbench.utils.tilelang_utils import preload_cuda_driver

from ..python_utils import pip_install_requirements

REQUIREMENTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "requirements.txt"
)


def install_requirements(requirements_txt: str):
    # ignore dependencies to bypass reinstalling pytorch stable version
    cmd = ["pip", "install", "-r", requirements_txt, "--no-deps"]
    subprocess.check_call(cmd)


def check_install():
    env = os.environ.copy()
    libcuda = preload_cuda_driver()
    if libcuda:
        ld_preload = env.get("LD_PRELOAD")
        env["LD_PRELOAD"] = f"{libcuda}:{ld_preload}" if ld_preload else libcuda
    cmd = [sys.executable, "-c", "import tilelang"]
    subprocess.check_call(cmd, env=env)


def install_tile():
    pip_install_requirements(REQUIREMENTS_FILE, extra_args=["--no-deps"])
    check_install()
