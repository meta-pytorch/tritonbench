#!/bin/bash

set -ex

if [ -z "${WORKSPACE_DIR}" ]; then
    WORKSPACE_DIR=/workspace
fi

cd ${WORKSPACE_DIR}

wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -u -p ${WORKSPACE_DIR}/miniconda3

# Test
. ${WORKSPACE_DIR}/miniconda3/etc/profile.d/conda.sh
conda activate base
conda init
conda tos accept
