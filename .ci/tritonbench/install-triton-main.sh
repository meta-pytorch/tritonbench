#!/bin/bash

if [ -z "${BASE_CONDA_ENV}" ]; then
  echo "ERROR: BASE_CONDA_ENV is not set"
  exit 1
fi

if [ -z "${CONDA_ENV}" ]; then
  echo "ERROR: CONDA_ENV is not set"
  exit 1
fi

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

CONDA_ENV=${BASE_CONDA_ENV} . "${SETUP_SCRIPT}"
conda activate "${BASE_CONDA_ENV}"
# Remove the conda env if exists
conda remove --name "${CONDA_ENV}" -y --all || true
conda create --name "${CONDA_ENV}" -y --clone "${BASE_CONDA_ENV}"
conda activate "${CONDA_ENV}"

. "${SETUP_SCRIPT}"

# Install and build triton from source code
cd /workspace
git clone https://github.com/triton-lang/triton.git
cd /workspace/triton
# delete the original triton directory
TRITON_PKG_DIR=$(python -c "import triton; import os; print(os.path.dirname(triton.__file__))")
# make sure all pytorch triton has been uninstalled
pip uninstall -y triton
pip uninstall -y triton
pip uninstall -y triton
rm -rf "${TRITON_PKG_DIR}"

# install main triton
pip install ninja cmake wheel pybind11; # build-time dependencies
pip install -e python

# test main branch installation with importing experimental descriptor
python -c "import triton.tools.experimental_descriptor"
