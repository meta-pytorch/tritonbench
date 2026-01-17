#/usr/bin bash

set -xeuo pipefail

if [ -z "${WORKSPACE_DIR:-}" ]; then
    export WORKSPACE_DIR=/workspace
fi

if [ -z "${SETUP_SCRIPT:-}" ]; then
    export SETUP_SCRIPT=${WORKSPACE_DIR}/setup_instance.sh
fi

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda) USE_CUDA="1";  ;;
        --hip) USE_HIP="1"; ;;
        --triton-main) USE_TRITON_MAIN="1"; ;;
        --meta-triton) USE_META_TRITON="1"; ;;
        --no-build) NO_BUILD="1"; ;;
        --test-nvidia-driver) TEST_NVIDIA_DRIVER="1"; ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ ! -e ${WORKSPACE_DIR} ]; then
    sudo mkdir -p ${WORKSPACE_DIR}
    sudo chown -R $(whoami):$(id -gn) ${WORKSPACE_DIR}
fi

touch "${SETUP_SCRIPT}"
echo ". ${SETUP_SCRIPT}" >> ${HOME}/.bashrc

if [ -n "${UV_VENV_DIR:-}" ]; then
    bash ./.ci/uv/install.sh
    . $HOME/.local/bin/env 
else
    bash ./.ci/conda/install.sh
    . "${SETUP_SCRIPT}"
fi

export CONDA_ENV=pytorch
echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> "${SETUP_SCRIPT}"

python3 tools/python_utils.py --create-conda-env ${CONDA_ENV}
if [ -n "${UV_VENV_DIR:-}" ]; then
    echo ". ${UV_VENV_DIR}/\${CONDA_ENV}/bin/activate" >> "${SETUP_SCRIPT}"
    . "${SETUP_SCRIPT}"
else
    echo "conda activate \${CONDA_ENV}" >> "${SETUP_SCRIPT}"
    . "${SETUP_SCRIPT}"
    python -m tools.cuda_utils --install-torch-deps
fi

bash .ci/tritonbench/install-pytorch-source.sh

if [ -n "${USE_CUDA:-}" ]; then
    python -m tools.cuda_utils --install-torch-nightly --cuda
    export PYTORCH_FILE_PATH=$(python -c "import torch; print(torch.__file__)")
    export NVIDIA_LIB_PATH=$(realpath $(dirname ${PYTORCH_FILE_PATH})/../nvidia/cublas/lib)

    if [ -e ${NVIDIA_LIB_PATH} ]; then
        cd ${NVIDIA_LIB_PATH}
        ln -s libcublas.so.* libcublas.so && ln -s libcublasLt.so.* libcublasLt.so &&  ln -s libnvblas.so.* libnvblas.so
        
        cat <<EOF >> "${SETUP_SCRIPT}"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
        cd -
    fi
    # Hack: install nvidia compute to get libcuda.so.1
    if [ -n "${TEST_NVIDIA_DRIVER:-}" ]; then
        sudo apt update && sudo apt-get install -y libnvidia-compute-580
    fi
elif [ -n "${USE_HIP:-}" ]; then
    python -m tools.cuda_utils --install-torch-nightly --hip
else
    echo "Unknown backend. Only CUDA and HIP are supported."
    exit 1
fi

bash .ci/tritonbench/install.sh

if [ -n "${USE_CUDA:-}" ] && [ -n "${TEST_NVIDIA_DRIVER:-}" ]; then
    sudo apt-get purge -y '^libnvidia-'
    sudo apt-get purge -y '^nvidia-'
fi

if [ -n "${NO_BUILD:-}" ]; then
    CMD_SUFFIX="--no-build"
else
    CMD_SUFFIX=""
fi

if [ -n "${USE_TRITON_MAIN:-}" ]; then
    bash ./.ci/triton/install-triton-main.sh ${CMD_SUFFIX}
fi
if [ -n "${USE_META_TRITON:-}" ]; then
    bash ./.ci/triton/install-meta-triton.sh ${CMD_SUFFIX}
fi

cat "${SETUP_SCRIPT}"
