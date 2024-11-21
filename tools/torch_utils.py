"""
CUDA/ROCM independent pytorch installation helpers.
"""

import importlib
import re
import subprocess
from pathlib import Path

from typing import Optional

REPO_ROOT = Path(__file__).parent.parent.parent

TORCH_NIGHTLY_PACKAGES = ["torch"]
PIN_CMAKE_VERSION = "3.22.*"
BUILD_REQUIREMENTS_FILE = REPO_ROOT.joinpath("utils", "build_requirements.txt")


def is_hip() -> bool:
    import torch

    version = torch.__version__
    return "rocm" in version


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
