import argparse
import os
import subprocess

# defines the default ROCM version to compile against
DEFAULT_ROCM_VERSION = "7.0"
ROCM_VERSION_MAP = {
    "6.4": {
        "pytorch_url": "rocm6.4",
    },
    "7.0": {
        "pytorch_url": "rocm7.0",
    },
}


def install_torch_deps():
    # install other dependencies
    torch_deps = [
        "requests",
        "ninja",
        "pyyaml",
        "setuptools",
        "gitpython",
        "beautifulsoup4",
        "regex",
    ]
    cmd = ["conda", "install", "-y"] + torch_deps
    subprocess.check_call(cmd)
    # conda forge deps
    # ubuntu 24.04 comes with libstdcxx6 14.3.0
    # we need to install the same library version in conda
    conda_deps = ["libstdcxx-ng=14.3.0"]
    cmd = ["conda", "install", "-y", "-c", "conda-forge"] + conda_deps
    subprocess.check_call(cmd)


def install_pytorch_nightly(rocm_version: str, env, dryrun=False):
    from .torch_utils import TORCH_NIGHTLY_PACKAGES

    uninstall_torch_cmd = ["pip", "uninstall", "-y"]
    uninstall_torch_cmd.extend(TORCH_NIGHTLY_PACKAGES)
    if dryrun:
        print(f"Uninstall pytorch: {uninstall_torch_cmd}")
    else:
        # uninstall multiple times to make sure the env is clean
        for _loop in range(3):
            subprocess.check_call(uninstall_torch_cmd)
    pytorch_nightly_url = f"https://download.pytorch.org/whl/nightly/{ROCM_VERSION_MAP[rocm_version]['pytorch_url']}"
    install_torch_cmd = ["pip", "install", "--pre", "--no-cache-dir"]
    install_torch_cmd.extend(TORCH_NIGHTLY_PACKAGES)
    install_torch_cmd.extend(["-i", pytorch_nightly_url])
    if dryrun:
        print(f"Install pytorch nightly: {install_torch_cmd}")
    else:
        subprocess.check_call(install_torch_cmd, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rocmver",
        default=DEFAULT_ROCM_VERSION,
        help="Specify rocm version.",
    )
    parser.add_argument(
        "--install-torch-deps",
        action="store_true",
        help="Install pytorch runtime dependencies",
    )
    parser.add_argument(
        "--install-torch-nightly",
        action="store_true",
        help="Install pytorch nightly",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dryrun the commands",
    )
    args = parser.parse_args()
    if args.install_torch_deps:
        install_torch_deps()
    if args.install_torch_nightly:
        install_pytorch_nightly(args.rocmver, env=os.environ, dryrun=args.dryrun)
