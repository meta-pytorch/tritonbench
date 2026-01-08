#!/bin/bash

set -ex

if [ -z "${SETUP_SCRIPT}" ]; then
    echo "SETUP_SCRIPT is not set"
    exit 1
fi

if [ -z "${WORKSPACE_DIR}" ]; then
    echo "WORKSPACE_DIR is not set"
    exit 1
fi

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

