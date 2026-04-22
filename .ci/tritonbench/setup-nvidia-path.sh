#/usr/bin bash

set -xeuo pipefail

if [ -z "${WORKSPACE_DIR:-}" ]; then
    export WORKSPACE_DIR=/workspace
fi

if [ -z "${SETUP_SCRIPT:-}" ]; then
    export SETUP_SCRIPT=${WORKSPACE_DIR}/setup_instance.sh
fi

. "${SETUP_SCRIPT}"

export PYTORCH_FILE_PATH=$(python -c "import torch; print(torch.__file__)")

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
        cat <<EOF >> "${SETUP_SCRIPT}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
    fi
done
