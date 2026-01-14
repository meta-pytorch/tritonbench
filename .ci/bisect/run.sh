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

if [ -z "${BENCHMARK_NAME:-}" ]; then
    echo "ERROR: BENCHMARK_NAME is not set"
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

cd "${TRITONBENCH_DIR}"


git config --global --add safe.directory '*'

BISECT_DIR="${WORKSPACE_DIR}/bisect"
OUTPUT_DIR="${BISECT_DIR}/bisect-output"
BASELINE_RESULTS="${OUTPUT_DIR}/baseline_results.json"
BISECT_LOG="${OUTPUT_DIR}/bisect.log"

mkdir -p "${BISECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "===== TritonBench Bisect Driver Script START =====" | tee "${BISECT_LOG}"
echo "Good commit: ${GOOD_COMMIT}" | tee -a "${BISECT_LOG}"
echo "Bad commit: ${BAD_COMMIT}" | tee -a "${BISECT_LOG}"
echo "Triton repo: ${TRITON_REPO}" | tee -a "${BISECT_LOG}"
echo "Benchmark: ${BENCHMARK_NAME}" | tee -a "${BISECT_LOG}"
echo "Benchmark command-line: ${OPERATOR:-all}" | tee -a "${BISECT_LOG}"
echo "==================================================" | tee -a "${BISECT_LOG}"

# Checkout tritonparse
TRITONPARSE_DIR="${WORKSPACE_DIR}/tritonparse"
git clone https://github.com/meta-pytorch/tritonparse.git ${TRITONPARSE_DIR}

cd ${WORKSPACE_DIR}/tritonparse
