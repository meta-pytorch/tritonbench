#/usr/bin bash

set -xeuo pipefail

if [ -z "${WORKSPACE_DIR}" ]; then
    export WORKSPACE_DIR=/workspace
fi

if [ -z "${SETUP_SCRIPT}" ]; then
    export SETUP_SCRIPT=${WORKSPACE_DIR}/setup_instance.sh
fi

# Initialize workspace directory
if [ -e "${WORKSPACE_DIR}" ]; then
    rm -r "${WORKSPACE_DIR}"
fi


# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda) USE_CUDA="1";  ;;
        --hip) USE_HIP="1"; ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done


sudo mkdir ${WORKSPACE_DIR}
sudo chmod 777 ${WORKSPACE_DIR}

bash ./.ci/conda/install.sh

echo "\
. ${WORKSPACE_DIR}/miniconda3/etc/profile.d/conda.sh && \
conda activate base && \
export CONDA_HOME=${WORKSPACE_DIR}/miniconda3 " >> ${SETUP_SCRIPT}

echo ". ${SETUP_SCRIPT}" >> ${HOME}/.bashrc

export CONDA_ENV=pytorch

. "${SETUP_SCRIPT}"

python tools/python_utils.py --create-conda-env ${CONDA_ENV} && \
echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> "${SETUP_SCRIPT}" && \
echo "conda activate \${CONDA_ENV}" >> "${SETUP_SCRIPT}"

python -m tools.cuda_utils --install-torch-deps

if [ -n "${USE_CUDA:-}" ]; then
    python -m tools.cuda_utils --install-torch-nightly --cuda
    export PYTORCH_FILE_PATH=$(python -c "import torch; print(torch.__file__)")
    export NVIDIA_LIB_PATH=$(realpath $(dirname ${PYTORCH_FILE_PATH})/../nvidia/cublas/lib)

    if [ -e ${NVIDIA_LIB_PATH} ]; then
        cd ${NVIDIA_LIB_PATH}
        ln -s libcublas.so.* libcublas.so && ln -s libcublasLt.so.* libcublasLt.so &&  ln -s libnvblas.so.* libnvblas.so && \
        echo "export LD_LIBRARY_PATH=${NVIDIA_LIB_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n" >> /workspace/setup_instance.sh
        cd -
    fi
elif [ -n "${USE_HIP:-}" ]; then
    python -m tools.cuda_utils --install-torch-nightly --hip
else
    echo "Unknown backend. Only CUDA and HIP are supported."
    exit 1
fi


bash .ci/tritonbench/install-pytorch-source.sh

bash .ci/tritonbench/install.sh

CONDA_ENV_TRITON_MAIN=triton-main
bash .ci/triton/install.sh --conda-env "${CONDA_ENV_TRITON_MAIN}" \
        --repo triton-lang/triton --commit main --side single --nightly \
        --install-dir ${WORKSPACE_DIR}/triton-main

cat "${SETUP_SCRIPT}"
