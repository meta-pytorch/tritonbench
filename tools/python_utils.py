import logging
import subprocess
import sys
from pathlib import Path

from typing import Dict, List, Optional

DEFAULT_PYTHON_VERSION = "3.12"

PYTHON_VERSION_MAP = {
    "3.11": {
        "pytorch_url": "cp311",
    },
    "3.12": {
        "pytorch_url": "cp312",
    },
}
REPO_DIR = Path(__file__).parent.parent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_conda_env(pyver: str, name: str):
    command = ["conda", "create", "-n", name, "-y", f"python={pyver}"]
    subprocess.check_call(command)


def get_pkg_versions(packages: List[str]) -> Dict[str, str]:
    versions = {}
    for module in packages:
        cmd = [sys.executable, "-c", f"import {module}; print({module}.__version__)"]
        version = subprocess.check_output(cmd).decode().strip()
        versions[module] = version
    return versions


def has_pkg(pkg: str):
    """
    Check if a package is installed
    """
    try:
        cmd = [sys.executable, "-c", f"import {pkg}; {pkg}.__version__"]
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


def generate_build_constraints(package_versions: Dict[str, str]):
    """
    Generate package versions dict and save them to REPO_DIR/build/constraints.txt
    """
    output_dir = REPO_DIR.joinpath("build")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir.joinpath("constraints.txt"), "w") as fp:
        for k, v in package_versions.items():
            fp.write(f"{k}=={v}\n")


def pip_install_requirements(
    requirements_txt="requirements.txt",
    continue_on_fail=False,
    no_build_isolation=False,
    add_build_constraints=True,
    extra_args: Optional[List[str]] = None,
):
    import sys

    constraints_file = REPO_DIR.joinpath("build", "constraints.txt")
    if add_build_constraints:
        if not constraints_file.exists():
            logger.warn(
                "The build/constrants.txt file is not found. "
                "Please consider rerunning the install.py script to generate it."
                "It is recommended to install with the build/constrants.txt file "
                "to prevent unexpected version change of numpy or torch."
            )
            constraints_parameters = []
        else:
            constraints_parameters = ["-c", str(constraints_file.resolve())]
    else:
        constraints_parameters = []

    if no_build_isolation:
        constraints_parameters.append("--no-build-isolation")
    if extra_args and isinstance(extra_args, list):
        constraints_parameters.extend(extra_args)
    if not continue_on_fail:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements_txt]
            + constraints_parameters,
        )
        return True, None
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_txt]
            + constraints_parameters,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        return (False, e.output)
    except Exception as e:
        return (False, e)
    return True, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyver",
        type=str,
        default=DEFAULT_PYTHON_VERSION,
        help="Specify the Python version.",
    )
    parser.add_argument(
        "--create-conda-env",
        type=str,
        default=None,
        help="Create conda environment of the default Python version.",
    )
    args = parser.parse_args()
    if args.create_conda_env:
        create_conda_env(args.pyver, args.create_conda_env)
