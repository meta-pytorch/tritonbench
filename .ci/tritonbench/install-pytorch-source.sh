#!/bin/bash

set -ex

if [ -z "${SETUP_SCRIPT:-}" ] || [ ! -e "${SETUP_SCRIPT}" ]; then
    echo "SETUP_SCRIPT is not set or not exist"
    exit 1
fi

if [ -z "${WORKSPACE_DIR:-}" ] || [ ! -e "${WORKSPACE_DIR}" ]; then
    echo "WORKSPACE_DIR is not set or not exist"
    exit 1
fi

PYTORCH_REPO_URL="https://github.com/pytorch/pytorch.git"

# The checkout is only used to resolve the branch and commit time of the
# installed pytorch nightly, so it must be an upstream pytorch/pytorch clone.
is_upstream_pytorch_clone() {
    CHECKOUT_DIR=$1
    REMOTE_URL=$(git -C "${CHECKOUT_DIR}" config --get remote.origin.url 2>/dev/null) || return 1
    case "${REMOTE_URL}" in
        *pytorch/pytorch*) return 0 ;;
        *) return 1 ;;
    esac
}

update_pytorch() {
    PYTORCH_INSTALL_DIR=$1
    cd "${PYTORCH_INSTALL_DIR}"
    # the checkout may be shallow or single-branch, make sure main is fetchable
    if [ "$(git rev-parse --is-shallow-repository)" == "true" ]; then
        git fetch --unshallow origin
    fi
    git remote set-branches origin '*'
    git fetch origin main
    git checkout -B main origin/main
    git submodule sync
    git submodule update --init --recursive
    git fetch origin nightly
}


PYTORCH_INSTALL_DIR="${WORKSPACE_DIR}/pytorch"
# Some base docker images (e.g. rocm/pytorch) ship their own pytorch checkout
# at this path. It can be a fork or a partial clone that we can't update from
# upstream, in which case keep it untouched and use our own directory.
if [ -e "${PYTORCH_INSTALL_DIR}" ] && ! is_upstream_pytorch_clone "${PYTORCH_INSTALL_DIR}"; then
    echo "WARNING: ${PYTORCH_INSTALL_DIR} is not an upstream pytorch checkout, cloning to ${WORKSPACE_DIR}/pytorch-src instead"
    PYTORCH_INSTALL_DIR="${WORKSPACE_DIR}/pytorch-src"
fi

if is_upstream_pytorch_clone "${PYTORCH_INSTALL_DIR}"; then
    # the pytorch repo is already cloned, update it
    update_pytorch "${PYTORCH_INSTALL_DIR}"
else
    rm -rf "${PYTORCH_INSTALL_DIR}"
    git clone "${PYTORCH_REPO_URL}" "${PYTORCH_INSTALL_DIR}"
fi

echo "export TRITONBENCH_PYTORCH_REPO_PATH=${PYTORCH_INSTALL_DIR}" >> "${SETUP_SCRIPT}"
