#!/bin/bash

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
fi

# Install Tritonbench and all its customized packages
python install.py --all
