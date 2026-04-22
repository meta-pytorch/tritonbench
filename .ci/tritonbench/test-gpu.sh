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

# workaround: add NVIDIA shared libraries to LD_LIBRARY_PATH
PYTORCH_FILE_PATH=$(python -c "import torch; print(torch.__file__)")
PYTORCH_DIR=$(dirname "${PYTORCH_FILE_PATH}")
NVIDIA_LIB_PATHS=(
  "$(realpath "${PYTORCH_DIR}/../nvidia/cu13/lib")"
  "$(realpath "${PYTORCH_DIR}/../nvidia/cudnn/lib")"
  "$(realpath "${PYTORCH_DIR}/../nvidia/cusparselt/lib")"
  "$(realpath "${PYTORCH_DIR}/../nvidia/nccl/lib")"
  "$(realpath "${PYTORCH_DIR}/../nvidia/nvshmem/lib")"
)

for NVIDIA_LIB_PATH in "${NVIDIA_LIB_PATHS[@]}"; do
  if [ -e "${NVIDIA_LIB_PATH}" ]; then
    export LD_LIBRARY_PATH=${NVIDIA_LIB_PATH}:${LD_LIBRARY_PATH}
  fi
done

# workaround: disable inductor subprocess compilation to avoid
# "Could not find an active GPU backend" in subprocess workers
export TORCHINDUCTOR_COMPILE_THREADS=1

python -m unittest test.test_gpu.main -v
