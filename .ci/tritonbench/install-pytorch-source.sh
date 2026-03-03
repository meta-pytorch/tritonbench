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


update_pytorch() {
    PYTORCH_INSTALL_DIR=$1
    cd "${PYTORCH_INSTALL_DIR}"
    git checkout main
    git pull origin main
    git submodule sync
    git submodule update --init --recursive
    git fetch origin nightly
}


if [ -e "${WORKSPACE_DIR}/pytorch" ]; then
    # the pytorch repo is already cloned, update it
    update_pytorch "${WORKSPACE_DIR}/pytorch"
else
    git clone https://github.com/pytorch/pytorch.git "${WORKSPACE_DIR}/pytorch"
fi

echo "export TRITONBENCH_PYTORCH_REPO_PATH=/workspace/pytorch" >> "${SETUP_SCRIPT}"
