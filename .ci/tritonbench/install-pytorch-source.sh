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

git clone https://github.com/pytorch/pytorch.git "${WORKSPACE_DIR}/pytorch"
echo "export TRITONBENCH_PYTORCH_REPO_PATH=/workspace/pytorch" >> "${SETUP_SCRIPT}"
