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

# setup nvidia cublas library path
export NVIDIA_LIB_PATH=$(realpath $(dirname ${PYTORCH_FILE_PATH})/../nvidia/cublas/lib)

if [ -e ${NVIDIA_LIB_PATH} ]; then
    cd ${NVIDIA_LIB_PATH}
    ln -s libcublas.so.* libcublas.so && ln -s libcublasLt.so.* libcublasLt.so &&  ln -s libnvblas.so.* libnvblas.so
    
    cat <<EOF >> "${SETUP_SCRIPT}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
    cd -
fi

# setup nvidia cuda_nvrtc library path
export NVIDIA_LIB_PATH=$(realpath $(dirname ${PYTORCH_FILE_PATH})/../nvidia/cuda_nvrtc/lib)

if [ -e ${NVIDIA_LIB_PATH} ]; then
    cd ${NVIDIA_LIB_PATH}
    cat <<EOF >> "${SETUP_SCRIPT}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
    cd -
fi
