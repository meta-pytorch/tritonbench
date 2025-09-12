#!/bin/bash

set -xeuo pipefail

# Print usage
usage() {
    echo "Usage: $0 --repo <repo-path> --commit <commit-hash> --side <a|b|single> --conda-env <env-name> --install-dir <triton-install-dir>"
    exit 1
}

# remove triton installations
remove_triton() {
    # delete the original triton directory
    TRITON_PKG_DIR=$(python -c "import triton; import os; print(os.path.dirname(triton.__file__))")
    # make sure all pytorch triton has been uninstalled
    pip uninstall -y triton
    pip uninstall -y triton
    pip uninstall -y triton
    rm -rf "${TRITON_PKG_DIR}"
}

checkout_triton() {
    REPO=$1
    COMMIT=$2
    TRITON_INSTALL_DIR=$3
    TRITON_INSTALL_DIRNAME=$(basename "${TRITON_INSTALL_DIR}")
    TRITON_INSTALL_BASEDIR=$(dirname "${TRITON_INSTALL_DIR}")
    cd "${TRITON_INSTALL_BASEDIR}"
    git clone "https://github.com/${REPO}.git" "${TRITON_INSTALL_DIRNAME}"
    cd "${TRITON_INSTALL_DIR}"
    git checkout "${COMMIT}"
}

install_triton() {
    TRITON_INSTALL_DIR=$1
    cd "${TRITON_INSTALL_DIR}"
    # install main triton
    pip install ninja cmake wheel pybind11; # build-time dependencies
    pip install -r python/requirements.txt
    # old versions of triton have setup.py in ./python; newer versions in ./
    if [ ! -f setup.py ]; then
      pushd python
    else
      pushd .
    fi
    pip install -e .
    popd
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --conda-env) CONDA_ENV="$2"; shift ;;
        --repo) REPO="$2"; shift ;;
        --commit) COMMIT="$2"; shift ;;
        --side) SIDE="$2"; shift ;;
        --install-dir) TRITON_INSTALL_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

# Validate arguments
if [ -z "${REPO}" ] || [ -z "${COMMIT}" ] || [ -z "${SIDE}" ]; then
    echo "Missing required arguments."
    usage
fi

if [ "${SIDE}" == "single" ]; then
    if [ -z "${CONDA_ENV}" ] || [ -z "${TRITON_INSTALL_DIR}" ]; then
        echo "Must specifify --conda-env and --install-dir with --side single."
        exit 1
    fi
elif [ "${SIDE}" == "a" ] || [ "${SIDE}" == "b" ]; then
    mkdir -p /workspace/abtest
    CONDA_ENV="triton-side-${SIDE}"
    TRITON_INSTALL_DIR=/workspace/abtest/${CONDA_ENV}
else
    echo "Unknown side: ${SIDE}"
    exit 1
fi

CONDA_ENV=pytorch . "${SETUP_SCRIPT}"
# Remove the conda env if exists
conda remove --name "${CONDA_ENV}" -y --all || true
cd /workspace/tritonbench
conda create --name "${CONDA_ENV}" -y --clone pytorch

. "${SETUP_SCRIPT}"

remove_triton

checkout_triton "${REPO}" "${COMMIT}" "${TRITON_INSTALL_DIR}"
install_triton "${TRITON_INSTALL_DIR}"

# export Triton repo related envs
# these envs will be used in nightly runs and other benchmarks
cd "${TRITON_INSTALL_DIR}"
TRITONBENCH_TRITON_COMMIT_HASH=$(git rev-parse --verify HEAD)
TRITONBENCH_TRITON_REPO=$(git config --get remote.origin.url | sed -E 's|.*github.com[:/](.+)\.git|\1|')

# If the current conda env matches the env we just created
# then export all Triton related envs to shell env
cat <<EOF >> /workspace/setup_instance.sh
if [ \${CONDA_DEFAULT_ENV} == "${CONDA_ENV}" ] ; then
    export TRITONBENCH_TRITON_COMMIT_HASH="${TRITONBENCH_TRITON_COMMIT_HASH}"
    export TRITONBENCH_TRITON_REPO="${TRITONBENCH_TRITON_REPO}"
    export TRITONBENCH_TRITON_COMMIT="${COMMIT}"
    export TRITONBENCH_TRITON_INSTALL_DIR="${TRITON_INSTALL_DIR}"
fi
EOF
