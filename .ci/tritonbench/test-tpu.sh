#!/bin/bash
set -x

# Smoke-test the TPU (torch_tpu) path. Unlike the GPU suites, this does not run
# the full operator tests -- most operators are Triton kernels, which do not run
# on TPU. It exercises `--device tpu` end-to-end on a non-Triton baseline.

if [ -n "${SETUP_SCRIPT}" ]; then
  source "${SETUP_SCRIPT}"
fi

# print versions for debugging
python -c "import torch; print('torch version:', torch.__version__, torch.__file__)"
python -c "import torch_tpu, torch; print('torch.tpu available:', torch.tpu.is_available())"

python run.py --op test_op --device tpu --metrics latency --num-inputs 1
