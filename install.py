import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from tools.cuda_utils import DEFAULT_TOOLKIT_VERSION, TOOLKIT_MAPPING
from tools.git_utils import checkout_submodules
from tools.python_utils import (
    generate_build_constraints,
    get_pkg_versions,
    has_pkg,
    pip_install_requirements,
    get_pip_cmd,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Install the latest pytorch nightly with default cuda/hip version
# if torch does not exist
if not has_pkg("torch"):
    from tools.torch_utils import install_pytorch_nightly

    env = os.environ
    toolkit_version = TOOLKIT_MAPPING[DEFAULT_TOOLKIT_VERSION]["pytorch_url"]
    install_pytorch_nightly(toolkit_version, env)

# requires torch
from tritonbench.utils.env_utils import is_hip


REPO_PATH = Path(os.path.abspath(__file__)).parent

# Packages we assume to have installed before running this script
# We will use build constraints to assume the version is not changed across the install
TRITONBENCH_DEPS = ["torch", "numpy"]


def install_jax(toolkit_version=DEFAULT_TOOLKIT_VERSION):
    jax_package_name = TOOLKIT_MAPPING[toolkit_version]["jax"]
    jax_nightly_html = (
        "https://storage.googleapis.com/jax-releases/jax_nightly_releases.html"
    )
    # install instruction:
    # https://jax.readthedocs.io/en/latest/installation.html
    # pip install -U --pre jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
    cmd = get_pip_cmd() + ["install", "--pre", jax_package_name, "-f", jax_nightly_html]
    subprocess.check_call(cmd)
    # Test jax installation
    test_cmd = [sys.executable, "-c", "import jax"]
    subprocess.check_call(test_cmd)


def install_fa2(compile=False):
    if compile:
        # compile from source (slow)
        FA2_PATH = REPO_PATH.joinpath("submodules", "flash-attention")
        cmd = get_pip_cmd() + ["install", "-e", "."]
        subprocess.check_call(cmd, cwd=str(FA2_PATH.resolve()))
    else:
        # Install the pre-built binary
        cmd = get_pip_cmd() + ["install", "flash-attn", "--no-build-isolation"]
        subprocess.check_call(cmd)


def install_liger():
    # Liger-kernel has a conflict dependency `triton` with pytorch,
    # so we need to install it without dependencies
    cmd = get_pip_cmd() + ["install", "liger-kernel-nightly", "--no-deps"]
    subprocess.check_call(cmd)


def install_tritonparse():
    # Install tritonparse from GitHub
    cmd = get_pip_cmd() + [
        "install",
        "-e",
        "git+https://github.com/meta-pytorch/tritonparse.git#egg=tritonparse",
        "--no-deps",
    ]
    subprocess.check_call(cmd)


def setup_hip(args: argparse.Namespace):
    # We have to disable all third-parties that donot support hip/rocm
    args.all = False
    args.liger = True
    args.aiter = True
    args.mslk = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--numpy", action="store_true", help="Install suggested numpy")
    parser.add_argument(
        "--fbgemm", action="store_true", help="Install prebuilt FBGEMM"
    )
    parser.add_argument(
        "--mslk", action="store_true", help="Install prebuilt MSLK"
    )
    parser.add_argument(
        "--mslk-compile",
        action="store_true",
        help="Compile and install MSLK",
    )
    parser.add_argument(
        "--fa2", action="store_true", help="Install optional flash_attention 2 kernels"
    )
    parser.add_argument(
        "--fa2-compile",
        action="store_true",
        help="Install optional flash_attention 2 kernels from source.",
    )
    parser.add_argument(
        "--fa3", action="store_true", help="Install optional flash_attention 3 kernels"
    )
    parser.add_argument("--helion", action="store_true", help="Install Helion")
    parser.add_argument("--jax", action="store_true", help="Install jax nightly")
    parser.add_argument("--tk", action="store_true", help="Install ThunderKittens")
    parser.add_argument("--liger", action="store_true", help="Install Liger-kernel")
    parser.add_argument("--quack", action="store_true", help="Install quack")
    parser.add_argument("--xformers", action="store_true", help="Install xformers")
    parser.add_argument("--tile", action="store_true", help="install tile lang")
    parser.add_argument("--aiter", action="store_true", help="install AMD's aiter")
    parser.add_argument(
        "--tritonparse", action="store_true", help="Install tritonparse"
    )
    parser.add_argument(
        "--all", action="store_true", help="Install all custom kernel repos"
    )
    args = parser.parse_args()

    if args.all and is_hip():
        setup_hip(args)

    if args.numpy or not has_pkg("numpy"):
        pip_install_requirements("requirements-numpy.txt", add_build_constraints=False)

    # generate build constraints before installing anything
    deps = get_pkg_versions(TRITONBENCH_DEPS)
    generate_build_constraints(deps)

    # install framework dependencies
    pip_install_requirements("requirements.txt")
    # checkout submodules
    checkout_submodules(REPO_PATH)
    # install submodules
    if args.fa3:
        # we need to install fa3 above all other dependencies
        logger.info("[tritonbench] installing fa3...")
        from tools.flash_attn.install import install_fa3

        install_fa3()
    if args.fbgemm or args.all:
        logger.info("[tritonbench] installing prebuilt FBGEMM GPU...")
        from tools.fbgemm.install import install_fbgemm, test_fbgemm

        install_fbgemm(prebuilt=True)
        test_fbgemm()
    if args.mslk or args.all:
        logger.info("[tritonbench] installing prebuilt MSLK...")
        from tools.mslk.install import install_mslk, test_mslk

        install_mslk(prebuilt=True)
        test_mslk()
    elif args.mslk_compile:
        logger.info("[tritonbench] compiling and installing MSLK...")
        from tools.mslk.install import install_mslk, test_mslk

        install_mslk(prebuilt=False)
        test_mslk()
    if args.fa2:
        logger.info("[tritonbench] installing fa2 from source...")
        install_fa2(compile=True)
    if args.jax:
        logger.info("[tritonbench] installing jax...")
        install_jax()
    if args.tk:
        logger.info("[tritonbench] installing thunderkittens...")
        from tools.tk.install import install_tk

        install_tk()
    if args.tile:
        logger.info("[tritonbench] installing tilelang...")
        from tools.tilelang.install import install_tile

        install_tile()
    if args.liger or args.all:
        logger.info("[tritonbench] installing liger-kernels...")
        install_liger()
    if args.quack or args.all:
        logger.info("[tritonbench] installing quack...")
        from tools.quack.install import install_quack

        install_quack()
    if args.helion:
        logger.info("[tritonbench] installing helion...")
        from tools.helion.install import install_helion

        install_helion()
    if args.xformers:
        logger.info("[tritonbench] installing xformers...")
        from tools.xformers.install import install_xformers

        install_xformers()
    if args.aiter and is_hip():
        logger.info("[tritonbench] installing aiter...")
        from tools.aiter.install import install_aiter

        install_aiter()
    if args.tritonparse:
        logger.info("[tritonbench] installing tritonparse...")
        install_tritonparse()
    logger.info("[tritonbench] installation complete!")
