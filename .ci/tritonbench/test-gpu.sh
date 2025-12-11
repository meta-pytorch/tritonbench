#!/bin/bash
set -x

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

# print pytorch and triton versions for debugging
python -c "import torch; print('torch version: ', torch.__version__); print('torch location: ', torch.__file__)"
python -c "import triton; print('triton version: ', triton.__version__); print('triton location: ', triton.__file__)"

# workaround: add libcublas.so to LD_LIBRARY_PATH
PYTORCH_FILE_PATH=$(python -c "import torch; print(torch.__file__)")
NVIDIA_LIB_PATH=$(realpath $(dirname ${PYTORCH_FILE_PATH})/../nvidia/cublas/lib)

export LD_LIBRARY_PATH=${NVIDIA_LIB_PATH}:${LD_LIBRARY_PATH}

python -m unittest test.test_gpu.main -v
