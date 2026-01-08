#!/bin/bash



git clone https://github.com/pytorch/pytorch.git /workspace/pytorch
echo "export TRITONBENCH_PYTORCH_REPO_PATH=/workspace/pytorch" >> "${SETUP_SCRIPT}"
