#!/bin/bash

if [ -z "${SETUP_SCRIPT:-}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

tritonbench_dir=$(dirname "$(readlink -f "$0")")/../..
cd ${tritonbench_dir}

# Install Tritonbench and all its customized packages
python install.py --all
