#!/bin/bash
set -x

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

source "${SETUP_SCRIPT}"

# if cuda, add cuda libraries to LD_LIBRARY_PATH
if python -c "import re, sys, torch; sys.exit(0 if re.search(r'\+cu[0-9]+', torch.__version__) else 1)"; then
  source .ci/tritonbench/setup-nvidia-path.sh
fi

# print pytorch and triton versions for debugging
python -c "import torch; print('torch version: ', torch.__version__); print('torch location: ', torch.__file__)"
python -c "import triton; print('triton version: ', triton.__version__); print('triton location: ', triton.__file__)"

# workaround: disable inductor subprocess compilation to avoid
# "Could not find an active GPU backend" in subprocess workers
export TORCHINDUCTOR_COMPILE_THREADS=1
# best effort disable autotune to speedup test
export TORCHINDUCTOR_MAX_AUTOTUNE=0

python -m unittest test.test_gpu.main -v
