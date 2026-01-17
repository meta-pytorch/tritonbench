#!/bin/bash
set -xeuo pipefail

if [ -z "${SETUP_SCRIPT:-}" ]; then
    echo "ERROR: SETUP_SCRIPT is not set"
    exit 1
fi

if [ -z "${WORKSPACE_DIR:-}" ]; then
    echo "ERROR: WORKSPACE_DIR is not set"
    exit 1
fi

if [ -z "${CONDA_ENV:-}" ]; then
    echo "ERROR: CONDA_ENV is not set"
    exit 1
fi

if [ -z "${GOOD_COMMIT:-}" ]; then
    echo "ERROR: GOOD_COMMIT is not set"
    exit 1
fi

if [ -z "${BAD_COMMIT:-}" ]; then
    echo "ERROR: BAD_COMMIT is not set"
    exit 1
fi

. "${SETUP_SCRIPT}"

if [ -z "${TRITONBENCH_TRITON_REPO:-}" ]; then
    echo "ERROR: TRITONBENCH_TRITON_REPO is not set"
    exit 1
fi

if [ -z "${TRITONBENCH_TRITON_INSTALL_DIR:-}" ]; then
    echo "ERROR: TRITONBENCH_TRITON_INSTALL_DIR is not set"
    exit 1
fi

TRITON_REPO=${TRITONBENCH_TRITON_REPO}
TRITON_SRC_DIR=${TRITONBENCH_TRITON_INSTALL_DIR}
REGRESSION_THRESHOLD="${REGRESSION_THRESHOLD:-0.1}"

TRITONBENCH_DIR=$(dirname "$(readlink -f "$0")")/../..

echo "===== TritonBench Bisect Driver Script START ====="
echo "Good commit: ${GOOD_COMMIT}"
echo "Bad commit: ${BAD_COMMIT}"
echo "Triton repo: ${TRITON_REPO}"
echo "Triton installation dir: ${TRITON_SRC_DIR}"
echo "Regression threshold: ${REGRESSION_THRESHOLD}"
echo "Repo command line: ${REPRO_CMDLINE}"
echo "=================================================="

# Checkout tritonparse
TRITONPARSE_DIR="${WORKSPACE_DIR}/tritonparse"
git clone https://github.com/meta-pytorch/tritonparse.git ${TRITONPARSE_DIR}

cd ${WORKSPACE_DIR}/tritonparse
git checkout -t origin/xz9/pr11-uv

# install tritonparse
uv pip install -e .

# switch back to tritonbench dir
cd "${TRITONBENCH_DIR}"

# kick off the bisect!
USE_UV=1 CONDA_DIR="${WORKSPACE_DIR}/uv_venvs/${CONDA_ENV}" \
BASELINE_LOG="${PWD}/bisect_logs/baseline.log" \
tritonparseoss bisect --triton-dir "${TRITON_SRC_DIR}" --test-script ./.ci/bisect/regression_detector.py \
--good ${GOOD_COMMIT} --bad ${BAD_COMMIT}
