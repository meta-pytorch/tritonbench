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

if [ -e "${WORKSPACE_DIR}/miniconda3" ]; then
    rm -r "${WORKSPACE_DIR}/miniconda3"
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

echo "\
. ${WORKSPACE_DIR}/miniconda3/etc/profile.d/conda.sh && \
conda activate base && \
export CONDA_HOME=${WORKSPACE_DIR}/miniconda3 " >> ${SETUP_SCRIPT}
