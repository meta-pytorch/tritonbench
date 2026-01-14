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

if [ -z "${COMMIT:-}" ]; then
    echo "ERROR: GOOD_COMMIT is not set"
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
TRITON_DIR=${TRITONBENCH_TRITON_INSTALL_DIR}
REGRESSION_THRESHOLD="${REGRESSION_THRESHOLD:-0.1}"

NOWTIME=$(date +%Y%m%d%H%M%S)

BISECT_DIR="${WORKSPACE_DIR}/bisect-${NOWTIME}"
OUTPUT_DIR="${BISECT_DIR}/bisect-output"
BISECT_LOG="${OUTPUT_DIR}/bisect.log"

# helper functions for triton installation
TRITONBENCH_DIR=$(dirname "$(readlink -f "$0")")/../..
. "${TRITONBENCH_DIR}/.ci/triton/triton_install_utils.sh"

# helper function for logging output
log_output() {
  if [ -n "$COMMIT_LOG" ]; then
    tee -a "$COMMIT_LOG"
  else
    cat
  fi
}

# ================= Setup =================
cd "$TRITON_DIR" || exit 128
# Get current commit info
COMMIT_HASH=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=9 HEAD)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create per-commit log file (optional, controlled by PER_COMMIT_LOG)
COMMIT_LOG=""
if [ "$PER_COMMIT_LOG" = "1" ]; then
  COMMIT_LOG="$LOG_DIR/${TIMESTAMP}_bisect_triton_${SHORT_COMMIT}.log"
fi

{
echo "===== TritonBench Bisect Script START =====" 
echo "Commit: ${COMMIT_HASH}"
echo "Short: ${SHORT_COIMMIT}"
echo "Triton dir: ${TRITON_REPO}"
echo "Test script: ${BENCHMARK_NAME}"
echo "Test args: ${OPERATOR:-all}"
echo "Venv: ${METRIC}"
echo "==========================================="
} | log_output


run_benchmark() {
    local env_name=$1
    local output_file=$2
    
    echo "Running benchmark with env ${env_name}..." | tee -a "${BISECT_LOG}"
    
    cd "${TRITONBENCH_DIR}"
    . "${SETUP_SCRIPT}"
    
    if [ -n "${OPERATOR}" ]; then
        python "benchmarks/${BENCHMARK_NAME}/run.py" \
            --op "${OPERATOR}" \
            --metrics "${METRIC}" \
            --output-json "${output_file}" \
            --ci || true
    else
        python "benchmarks/${BENCHMARK_NAME}/run.py" \
            --ci \
            --output-json "${output_file}" || true
    fi
}


# Update git submodules to match the current commit
echo "Updating git submodules..." | log_output
git submodule update --init --recursive 2>&1 | log_output
echo "" | log_output

# Build Triton
echo "Building Triton..." | log_output
BUILD_START=$(date +%s)

if [ -n "$COMMIT_LOG" ]; then
  eval "$BUILD_COMMAND" 2>&1 | tee -a "$COMMIT_LOG"
  BUILD_CODE=${PIPESTATUS[0]}
else
  eval "$BUILD_COMMAND" 2>&1
  BUILD_CODE=$?
fi

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))
echo "Build completed in ${BUILD_TIME}s, exit code: $BUILD_CODE" | log_output

if [ $BUILD_CODE -ne 0 ]; then
  echo "Build FAILED" | log_output
  exit 128
fi

echo "" | log_output

# Run test
echo "Running test..." | log_output
TEST_START=$(date +%s)

if [ -n "$COMMIT_LOG" ]; then
  TRITON_ALWAYS_COMPILE=1 python "$TEST_SCRIPT" $TEST_ARGS 2>&1 | tee -a "$COMMIT_LOG"
  TEST_CODE=${PIPESTATUS[0]}
else
  TRITON_ALWAYS_COMPILE=1 python "$TEST_SCRIPT" $TEST_ARGS 2>&1
  TEST_CODE=$?
fi

TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))
echo "Test completed in ${TEST_TIME}s, exit code: $TEST_CODE" | log_output

# Report result
if [ $TEST_CODE -eq 0 ]; then
  RESULT="GOOD"
  echo "✅ Passed" | log_output
else
  RESULT="BAD"
  echo "❌ Failed" | log_output
fi

echo "" | log_output
{
  echo "===== TritonBench Bisect Script Summary =====" 
  echo "Commit: $SHORT_COMMIT"
  echo "Build: ${BUILD_TIME}s (exit $BUILD_CODE)"
  echo "Test: ${TEST_TIME}s (exit $TEST_CODE)"
  echo "Result: $RESULT"
  if [ -n "$COMMIT_LOG" ]; then
    echo "Log: $COMMIT_LOG"
  fi
  echo "=============================================="
} | log_output

# Exit with test code for git bisect
exit $TEST_CODE
