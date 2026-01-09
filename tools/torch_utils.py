"""
CUDA/ROCM independent pytorch installation helpers.
"""

import importlib
import os
import re
import subprocess
from pathlib import Path

from typing import Optional

from .python_utils import get_pip_cmd

REPO_ROOT = Path(__file__).parent.parent.parent

TORCH_NIGHTLY_PACKAGES = ["torch"]
PIN_CMAKE_VERSION = "3.22.*"
USE_UV = os.path.exists(os.getenv("VIRTUAL_ENV", ""))
BUILD_REQUIREMENTS_FILE = REPO_ROOT.joinpath("utils", "build_requirements.txt")


def install_torch_build_deps():
    # Pin cmake version to stable
    # See: https://github.com/pytorch/builder/pull/1269
    torch_build_deps = [
        "cffi",
        "sympy",
        "typing_extensions",
        "future",
        "six",
        "dataclasses",
        "tabulate",
        "tqdm",
        "mkl",
        "mkl-include",
        f"cmake={PIN_CMAKE_VERSION}",
    ]
    cmd = ["conda", "install", "-y"] + torch_build_deps
    subprocess.check_call(cmd)
    build_deps = ["ffmpeg"]
    cmd = ["conda", "install", "-y"] + build_deps
    subprocess.check_call(cmd)
    # pip build deps
    cmd = ["pip", "install", "-r"] + [str(BUILD_REQUIREMENTS_FILE.resolve())]
    subprocess.check_call(cmd)
    # conda forge deps
    # ubuntu 22.04 comes with libstdcxx6 12.3.0
    # we need to install the same library version in conda to maintain ABI compatibility
    conda_deps = ["libstdcxx-ng=12.3.0"]
    cmd = ["conda", "install", "-y", "-c", "conda-forge"] + conda_deps
    subprocess.check_call(cmd)


def get_torch_nightly_version(pkg_name: str):
    pkg = importlib.import_module(pkg_name)
    version = pkg.__version__
    regex = ".*dev([0-9]+).*"
    date_str = re.match(regex, version).groups()[0]
    pkg_ver = {"version": version, "date": date_str}
    return (pkg_name, pkg_ver)


def check_torch_nightly_version(force_date: Optional[str] = None):
    pkg_versions = dict(map(get_torch_nightly_version, TORCH_NIGHTLY_PACKAGES))
    pkg_dates = [x[1]["date"] for x in pkg_versions.items()]
    if not len(set(pkg_dates)) == 1:
        raise RuntimeError(
            f"Found more than 1 dates in the torch nightly packages: {pkg_versions}."
        )
    if force_date and not pkg_dates[0] == force_date:
        raise RuntimeError(
            f"Force date value {force_date}, but found torch packages {pkg_versions}."
        )
    force_date_str = f"User force date {force_date}" if force_date else ""
    print(
        f"Installed consistent torch nightly packages: {pkg_versions}. {force_date_str}"
    )


def install_pytorch_nightly(toolkit_version: str, env, dryrun=False):
    from .torch_utils import TORCH_NIGHTLY_PACKAGES

    install_cmd = get_pip_cmd()
    uninstall_torch_cmd = install_cmd + ["uninstall"]
    if not USE_UV:
        uninstall_torch_cmd.extend(["-y"])
    uninstall_torch_cmd.extend(TORCH_NIGHTLY_PACKAGES)
    if dryrun:
        print(f"Uninstall pytorch: {uninstall_torch_cmd}")
    else:
        # uninstall multiple times to make sure the env is clean
        for _loop in range(3):
            subprocess.check_call(uninstall_torch_cmd)
    pytorch_nightly_url = f"https://download.pytorch.org/whl/nightly/{toolkit_version}"
    install_torch_cmd = install_cmd + ["install", "--pre", "--no-cache-dir"]
    install_torch_cmd.extend(TORCH_NIGHTLY_PACKAGES)
    install_torch_cmd.extend(["-i", pytorch_nightly_url])
    if dryrun:
        print(f"Install pytorch nightly: {install_torch_cmd}")
    else:
        subprocess.check_call(install_torch_cmd, env=env)
