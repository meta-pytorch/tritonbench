#!/bin/bash
set -x

if [ -z "${SETUP_SCRIPT:-}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --refresh-pytorch) INSTALL_PYTORCH_NIGHTLY="1";  ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

. "${SETUP_SCRIPT}"

tritonbench_dir=$(dirname "$(readlink -f "$0")")/../..
cd ${tritonbench_dir}

if [ -n "${INSTALL_PYTORCH_NIGHTLY}" ]; then
  uv pip uninstall pytorch torchvision
  python -m tools.cuda_utils --install-torch-nightly --cuda
  bash ./.ci/tritonbench/install-pytorch-source.sh
fi

if [ -n "${UV_VENV_DIR:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
  mkdir -p build
  export TRITONBENCH_UV_QUACK_OVERRIDE_FILE="${tritonbench_dir}/build/uv-quack-overrides.txt"
  python - <<'PY' > "${TRITONBENCH_UV_QUACK_OVERRIDE_FILE}"
import importlib.metadata as metadata

for package in ("torch", "triton"):
    print(f"{package}=={metadata.version(package)}")
PY
fi

# Install Tritonbench and all its customized packages
LD_LIBRARY_PATH=${LD_LIBRARY_PATH} python install.py --all
