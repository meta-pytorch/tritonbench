#!/bin/bash

set -xeuo pipefail

# Print usage
usage() {
    echo "Usage: $0 [--repo <repo-path>] [--commit <commit-hash>] --side <a|b|single> --conda-env <env-name> --install-dir <triton-install-dir> [--nightly] [--no-build] [--no-clone] [--no-checkout] [--skip-conda-reset]"
    exit 1
}


remove_env() {
    CONDA_ENV=$1
    if [ -n "${UV_VENV_DIR:-}" ]; then
        # uv
        rm -r "${UV_VENV_DIR}/${CONDA_ENV}" || true 
    else
        # conda
        conda remove --name "${CONDA_ENV}" -y --all || true
    fi
}

clone_env() {
    DEST_CONDA_ENV=$1
    SRC_CONDA_ENV=$2
    if [ -n "${UV_VENV_DIR:-}" ]; then
        cp -r "${UV_VENV_DIR}/${SRC_CONDA_ENV}" "${UV_VENV_DIR}/${DEST_CONDA_ENV}"
        # replace the activate script to point to the new env
        sed -i "s,${UV_VENV_DIR}/${SRC_CONDA_ENV},${UV_VENV_DIR}/${DEST_CONDA_ENV},g" "${UV_VENV_DIR}/${DEST_CONDA_ENV}/bin/activate"
    else
        # conda
        conda create --name "${DEST_CONDA_ENV}" -y --clone "${SRC_CONDA_ENV}"
    fi
}

# "NIGHTLY" option controls whether to truncate the branch
# to the earliest commit of the current day.
# This is useful for nightly runs across multiple devices.
NIGHTLY="0"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --conda-env) CONDA_ENV="$2"; shift ;;
        --repo) REPO="$2"; shift ;;
        --commit) COMMIT="$2"; shift ;;
        --side) SIDE="$2"; shift ;;
        --nightly) NIGHTLY="1"; ;;
        --no-build) NO_BUILD="1"; ;;
        --no-clone) NO_CLONE="1"; ;;
        --no-checkout) NO_CHECKOUT="1"; ;;
        --skip-conda-reset) SKIP_CONDA_RESET="1"; ;;
        --install-dir) TRITON_INSTALL_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "${WORKSPACE_DIR:-}" ]; then
  echo "ERROR: WORKSPACE_DIR is not set"
  exit 1
fi

if [ -z "${SETUP_SCRIPT:-}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

# Validate arguments
if [ -z "${SIDE:-}" ]; then
    echo "Missing required argument: --side."
    usage
fi
if [ -z "${NO_CHECKOUT:-}" ] && { [ -z "${REPO:-}" ] || [ -z "${COMMIT:-}" ]; }; then
    echo "Missing required arguments: --repo or --commit."
    usage
fi

if [ "${SIDE}" == "single" ]; then
    if [ -z "${CONDA_ENV:-}" ] || [ -z "${TRITON_INSTALL_DIR:-}" ]; then
        echo "Must specifify --conda-env and --install-dir with --side single."
        exit 1
    fi
elif [ "${SIDE}" == "a" ] || [ "${SIDE}" == "b" ]; then
    mkdir -p ${WORKSPACE_DIR}/abtest
    CONDA_ENV="triton-side-${SIDE}"
    TRITON_INSTALL_DIR=${WORKSPACE_DIR}/abtest/${CONDA_ENV}
else
    echo "Unknown side: ${SIDE}"
    exit 1
fi

if [ -n "${NO_CHECKOUT:-}" ] && [ ! -d "${TRITON_INSTALL_DIR}" ]; then
    echo "ERROR: --no-checkout requires an existing TRITON_INSTALL_DIR: ${TRITON_INSTALL_DIR}"
    exit 1
fi


if [ -z "${SKIP_CONDA_RESET:-}" ]; then
    CONDA_ENV=pytorch . "${SETUP_SCRIPT}"
    # Remove the conda env if exists
    remove_env "${CONDA_ENV}"
    clone_env "${CONDA_ENV}" pytorch
fi
. "${SETUP_SCRIPT}"

TRITONBENCH_DIR=$(dirname "$(readlink -f "$0")")/../..
. "${TRITONBENCH_DIR}/.ci/triton/triton_install_utils.sh"

remove_triton

if [ -z "${NO_CHECKOUT:-}" ]; then
    if [ -z "${NO_CLONE:-}" ]; then
        clone_triton "${REPO}" "${TRITON_INSTALL_DIR}"
    else
        update_triton "${TRITON_INSTALL_DIR}"
    fi

    checkout_triton "${COMMIT}" "${TRITON_INSTALL_DIR}" "${NIGHTLY}"
fi

if [ -z "${NO_BUILD:-}" ]; then
    install_triton "${TRITON_INSTALL_DIR}"
fi

# export Triton repo related envs
# these envs will be used in nightly runs and other benchmarks
cd "${TRITON_INSTALL_DIR}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    TRITONBENCH_TRITON_COMMIT_HASH=$(git rev-parse --verify HEAD)
    TRITON_REMOTE_URL=$(git config --get remote.origin.url || true)
    if [ -n "${TRITON_REMOTE_URL}" ]; then
        TRITONBENCH_TRITON_REPO=$(echo "${TRITON_REMOTE_URL}" | sed -E 's|.*github.com[:/](.+)\.git|\1|')
    else
        TRITONBENCH_TRITON_REPO="unknown"
    fi
else
    TRITONBENCH_TRITON_COMMIT_HASH="unknown"
    TRITONBENCH_TRITON_REPO="unknown"
fi
if [ -n "${NO_CHECKOUT:-}" ]; then
    COMMIT="${TRITONBENCH_TRITON_COMMIT_HASH}"
fi

# If the current conda env matches the env we just created
# then export all Triton related envs to shell env
cat <<EOF >> "${SETUP_SCRIPT}"
if [ \${CONDA_ENV} == "${CONDA_ENV}" ] ; then
    export TRITONBENCH_TRITON_COMMIT_HASH="${TRITONBENCH_TRITON_COMMIT_HASH}"
    export TRITONBENCH_TRITON_REPO="${TRITONBENCH_TRITON_REPO}"
    export TRITONBENCH_TRITON_COMMIT="${COMMIT}"
    export TRITONBENCH_TRITON_INSTALL_DIR="${TRITON_INSTALL_DIR}"
fi
EOF
