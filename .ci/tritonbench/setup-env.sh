#/usr/bin bash

set -xeuo pipefail

usage() {
    echo "Usage: $0 [--cuda|--hip] [--triton-main] [--meta-triton] [--no-build] [--test-nvidia-driver] [--triton-main-commit <hash-or-ref>] [--meta-triton-commit <hash-or-ref>]"
    exit 1
}

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
        --triton-main-commit)
            if [ -z "${2:-}" ]; then
                echo "ERROR: --triton-main-commit requires a value"
                usage
            fi
            TRITON_MAIN_COMMIT="$2"
            shift
            ;;
        --meta-triton-commit)
            if [ -z "${2:-}" ]; then
                echo "ERROR: --meta-triton-commit requires a value"
                usage
            fi
            META_TRITON_COMMIT="$2"
            shift
            ;;
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

if [ -n "${CONDA_ENV:-}" ]; then
    export CONDA_ENV=pytorch
fi
echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> "${SETUP_SCRIPT}"

python3 tools/python_utils.py --create-conda-env ${CONDA_ENV}
if [ -n "${UV_VENV_DIR:-}" ]; then
    echo ". ${UV_VENV_DIR}/\${CONDA_ENV}/bin/activate" >> "${SETUP_SCRIPT}"
    . "${SETUP_SCRIPT}"
else
    echo "conda activate \${CONDA_ENV}" >> "${SETUP_SCRIPT}"
    . "${SETUP_SCRIPT}"
fi
python -m tools.cuda_utils --install-torch-deps

bash .ci/tritonbench/install-pytorch-source.sh

if [ -n "${USE_CUDA:-}" ]; then
    python -m tools.cuda_utils --install-torch-nightly --cuda

    bash ./.ci/tritonbench/setup-nvidia-path.sh

    # Hack: install nvidia compute to get libcuda.so.1
    if [ -n "${TEST_NVIDIA_DRIVER:-}" ]; then
        sudo apt update && sudo apt-get install -y libnvidia-compute-580
    fi

elif [ -n "${USE_HIP:-}" ]; then
    python -m tools.cuda_utils --install-torch-nightly --hip
    bash ./.ci/tritonbench/setup-rocm-path.sh
else
    echo "Unknown backend. Only CUDA and HIP are supported."
    exit 1
fi

bash .ci/tritonbench/install.sh

if [ -n "${USE_CUDA:-}" ] && [ -n "${TEST_NVIDIA_DRIVER:-}" ]; then
    sudo apt-get purge -y '^libnvidia-'
    sudo apt-get purge -y '^nvidia-'
fi

COMMON_INSTALL_ARGS=()
if [ -n "${NO_BUILD:-}" ]; then
    COMMON_INSTALL_ARGS+=(--no-build)
fi

if [ -n "${USE_TRITON_MAIN:-}" ]; then
    TRITON_MAIN_INSTALL_ARGS=("${COMMON_INSTALL_ARGS[@]}")
    if [ -n "${TRITON_MAIN_COMMIT:-}" ]; then
        TRITON_MAIN_INSTALL_ARGS+=(--commit "${TRITON_MAIN_COMMIT}")
    fi
    bash ./.ci/triton/install-triton-main.sh "${TRITON_MAIN_INSTALL_ARGS[@]}"
fi
if [ -n "${USE_META_TRITON:-}" ]; then
    META_TRITON_INSTALL_ARGS=("${COMMON_INSTALL_ARGS[@]}")
    if [ -n "${META_TRITON_COMMIT:-}" ]; then
        META_TRITON_INSTALL_ARGS+=(--commit "${META_TRITON_COMMIT}")
    fi
    bash ./.ci/triton/install-meta-triton.sh "${META_TRITON_INSTALL_ARGS[@]}"
fi

cat "${SETUP_SCRIPT}"
