#!/bin/bash

if [ -z "${SETUP_SCRIPT:-}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

tritonbench_dir=$(dirname "$(readlink -f "$0")")/../..
cd ${tritonbench_dir}

# Hack: install nvidia compute to get libcuda.so.1
sudo apt update && sudo apt-get install -y libnvidia-compute-580

# Install Tritonbench and all its customized packages
python install.py --all

sudo apt-get purge -y '^libnvidia-'
sudo apt-get purge -y '^nvidia-'
