#!/usr/bin/env bash

set -xeuo pipefail

usage() {
    echo "Usage: $0 [--cuda|--hip] [--triton-main] [--meta-triton] [--custom-triton <triton-dir>] [--no-build] [--test-nvidia-driver] [--install-torch-wheel <wheel-url>] [--triton-main-commit <hash-or-ref>] [--meta-triton-commit <hash-or-ref>]"
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
        --custom-triton)
            if [ -z "${2:-}" ]; then
                echo "ERROR: --custom-triton requires a value"
                usage
            fi
            CUSTOM_TRITON_DIR="$2"
            shift
            ;;
        --install-torch-wheel)
            if [ -z "${2:-}" ]; then
                echo "ERROR: --install-torch-wheel requires a value"
                usage
            fi
            INSTALL_TORCH_WHEEL="$2"
            shift
            ;;
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

if [ ! -e "${WORKSPACE_DIR}" ]; then
    sudo mkdir -p "${WORKSPACE_DIR}"
    sudo chown -R "$(whoami):$(id -gn)" "${WORKSPACE_DIR}"
fi

touch "${SETUP_SCRIPT}"
echo ". ${SETUP_SCRIPT}" >> "${HOME}/.bashrc"

if [ -n "${UV_VENV_DIR:-}" ]; then
    bash ./.ci/uv/install.sh
    . "${HOME}/.local/bin/env"
else
    bash ./.ci/conda/install.sh
    . "${SETUP_SCRIPT}"
fi

if [ -z "${CONDA_ENV:-}" ]; then
    export CONDA_ENV=pytorch
fi
echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> "${SETUP_SCRIPT}"

if [ -z "${CUSTOM_TRITON_DIR:-}" ]; then
    export BOOTSTRAP_CONDA_ENV=pytorch
else
    export BOOTSTRAP_CONDA_ENV=$CONDA_ENV
fi    
python3 tools/python_utils.py \
        --create-conda-env "${BOOTSTRAP_CONDA_ENV}"
if [ -n "${UV_VENV_DIR:-}" ]; then
    echo ". ${UV_VENV_DIR}/\${CONDA_ENV}/bin/activate" >> "${SETUP_SCRIPT}"
    # use pytorch conda env to bootstrap
    CONDA_ENV=${BOOTSTRAP_CONDA_ENV} . "${SETUP_SCRIPT}"
else
    echo "conda activate \${CONDA_ENV}" >> "${SETUP_SCRIPT}"
    # use pytorch conda env to bootstrap
    CONDA_ENV=${BOOTSTRAP_CONDA_ENV} . "${SETUP_SCRIPT}"
fi

bash .ci/tritonbench/install-pytorch-source.sh

if [ -n "${USE_CUDA:-}" ]; then
    TORCH_INSTALL_ARGS=(--cuda)
    if [ -n "${INSTALL_TORCH_WHEEL:-}" ]; then
        TORCH_INSTALL_ARGS+=(--install-torch-wheel "${INSTALL_TORCH_WHEEL}")
        python -m tools.cuda_utils --install-torch-deps
    else
        TORCH_INSTALL_ARGS+=(--install-torch-nightly)
    fi
    python -m tools.cuda_utils "${TORCH_INSTALL_ARGS[@]}"

    bash ./.ci/tritonbench/setup-nvidia-path.sh

    # Hack: install nvidia compute to get libcuda.so.1
    if [ -n "${TEST_NVIDIA_DRIVER:-}" ]; then
        sudo apt update && sudo apt-get install -y libnvidia-compute-580
    fi

elif [ -n "${USE_HIP:-}" ]; then
    TORCH_INSTALL_ARGS=(--hip)
    if [ -n "${INSTALL_TORCH_WHEEL:-}" ]; then
        TORCH_INSTALL_ARGS+=(--install-torch-wheel "${INSTALL_TORCH_WHEEL}")
        python -m tools.cuda_utils --install-torch-deps
    else
        TORCH_INSTALL_ARGS+=(--install-torch-nightly)
    fi
    python -m tools.cuda_utils "${TORCH_INSTALL_ARGS[@]}"
    bash ./.ci/tritonbench/setup-rocm-path.sh
else
    echo "Unknown backend. Only CUDA and HIP are supported in the CI."
    exit 1
fi


COMMON_INSTALL_ARGS=()
if [ -n "${NO_BUILD:-}" ]; then
    COMMON_INSTALL_ARGS+=(--no-build)
fi

# when there is no custom triton or install specific pytorch wheel
# it means we have pytorch-triton installed already
# install tritonbench in this case
if [ -z "${CUSTOM_TRITON_DIR:-}" ] && [ -z "${INSTALL_TORCH_WHEEL:-}" ]; then
    bash .ci/tritonbench/install.sh
fi

if [ -n "${CUSTOM_TRITON_DIR:-}" ]; then
    CUSTOM_TRITON_INSTALL_ARGS=("${COMMON_INSTALL_ARGS[@]}" --no-checkout --skip-conda-reset)
    bash ./.ci/triton/install.sh --conda-env "${CONDA_ENV}" \
        --side single --install-dir "${CUSTOM_TRITON_DIR}" \
        "${CUSTOM_TRITON_INSTALL_ARGS[@]}"
else
    if [ -n "${USE_TRITON_MAIN:-}" ]; then
        TRITON_MAIN_INSTALL_ARGS=("${COMMON_INSTALL_ARGS[@]}")
        if [ -n "${TRITON_MAIN_COMMIT:-}" ]; then
            TRITON_MAIN_INSTALL_ARGS+=(--commit "${TRITON_MAIN_COMMIT}")
        else
            TRITON_MAIN_INSTALL_ARGS+=(--commit main --nightly)
        fi
        export CONDA_ENV="triton-main"
        bash ./.ci/triton/install.sh --conda-env "${CONDA_ENV}" \
             --repo triton-lang/triton --side single \
             --install-dir "${WORKSPACE_DIR}/triton-main" "${TRITON_MAIN_INSTALL_ARGS[@]}"
    fi
    if [ -n "${USE_META_TRITON:-}" ]; then
        META_TRITON_INSTALL_ARGS=("${COMMON_INSTALL_ARGS[@]}")
        if [ -n "${META_TRITON_COMMIT:-}" ]; then
            META_TRITON_INSTALL_ARGS+=(--commit "${META_TRITON_COMMIT}")
        else
            META_TRITON_INSTALL_ARGS+=(--commit main --nightly)
        fi
        export CONDA_ENV="meta-triton"
        bash ./.ci/triton/install.sh --conda-env "${CONDA_ENV}" \
             --repo facebookexperimental/triton --side single \
             --install-dir "${WORKSPACE_DIR}/meta-triton" "${META_TRITON_INSTALL_ARGS[@]}"
    fi
fi

if [ -n "${CUSTOM_TRITON_DIR:-}" ] || [ -n "${INSTALL_TORCH_WHEEL:-}" ]; then
    # when using custom triton or using custom pytorch wheel,
    # install tritonbench after installing triton
    # because it will skip conda clone or ignore triton dependency
    bash .ci/tritonbench/install.sh
fi

if [ -n "${USE_CUDA:-}" ] && [ -n "${TEST_NVIDIA_DRIVER:-}" ]; then
    sudo apt-get purge -y '^libnvidia-'
    sudo apt-get purge -y '^nvidia-'
fi

cat "${SETUP_SCRIPT}"
