#!/bin/bash
set -xu

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

if [ -z "${REPRO_CMDLINE:-}" ]; then
    echo "ERROR: REPRO_CMDLINE is not set"
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

NOWTIME=$(date +%Y%m%d%H%M%S)

BISECT_DIR="${WORKSPACE_DIR}/bisect-${NOWTIME}"
OUTPUT_DIR="${BISECT_DIR}/bisect-output"
BISECT_LOG="${OUTPUT_DIR}/bisect.log"
BASELINE_LOG="${OUTPUT_DIR}/baseline.log"

if [ -n "${BISECT_FUNCTIONAL:-}" ]; then
  REPRO_SUFFIX="--functional"
fi

if [ -n "${REGRESSION_THRESHOLD:-}" ]; then
  REPRO_SUFFIX="${REPRO_SUFFIX} --threshold ${REGRESSION_THRESHOLD}"
fi

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
echo "[tritonbench bisect] ===== TritonBench Bisect Script START =====" 
echo "[tritonbench bisect] Commit: ${COMMIT_HASH}"
echo "[tritonbench bisect] Short: ${SHORT_COIMMIT}"
echo "[tritonbench bisect] Triton dir: ${TRITON_REPO}"
echo "[tritonbench bisect] Test script: ${BENCHMARK_NAME}"
echo "[tritonbench bisect] Test args: ${OPERATOR:-all}"
echo "[tritonbench bisect] Venv: ${METRIC}"
echo "[tritonbench bisect] ==========================================="
} | log_output


# Update git submodules to match the current commit
echo "[tritonbench bisect] Updating git submodules..." | log_output
git submodule update --init --recursive 2>&1 | log_output
echo "" | log_output

# Build Triton
echo "[tritonbench bisect] Building Triton..." | log_output
BUILD_START=$(date +%s)

if [ -n "$COMMIT_LOG" ]; then
  remove_triton 2>&1 | tee -a "$COMMIT_LOG"
  install_triton "${TRITON_DIR}" 2>&1 | tee -a "$COMMIT_LOG"
  BUILD_CODE=${PIPESTATUS[0]}
else
  remove_triton 2>&1
  install_triton "${TRITON_DIR}" 2>&1
  BUILD_CODE=$?
fi

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))
echo "[tritonbench bisect] Build completed in ${BUILD_TIME}s, exit code: $BUILD_CODE" | log_output

if [ $BUILD_CODE -ne 0 ]; then
  echo "Build FAILED" | log_output
  exit 128
fi

echo "" | log_output

# Run test
echo "[tritonbench bisect] Running test..." | log_output
TEST_START=$(date +%s)


# detect baseline output exists
if [ -f "${BASELINE_LOG}" ]; then
  # run the baseline with logs saved in the baseline log file
  python ./.ci/bisect/regression_detector.py --repro \"${REPRO_CMDLINE}\" ${REPRO_SUFFIX} 2>&1 | tee -a "${BASELINE_LOG}"
  TEST_CODE=${PIPESTATUS[0]}
  if [ -n "$COMMIT_LOG" ]; then
    cat "${BASELINE_LOG}" | tee -a "$COMMIT_LOG"
  fi
else
  if [ -n "$COMMIT_LOG" ]; then
    python ./.ci/bisect/regression_detector.py --repro \"${REPRO_CMDLINE}\" --baseline "${BASELINE_LOG}" ${REPRO_SUFFIX} 2>&1 | tee -a "$COMMIT_LOG"
    TEST_CODE=${PIPESTATUS[0]}
  else
    python ./.ci/bisect/regression_detector.py --repro \"${REPRO_CMDLINE}\" --baseline "${BASELINE_LOG}" ${REPRO_SUFFIX} 2>&1
    TEST_CODE=$?
  fi
fi

TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))
echo "[tritonbench bisect] Test completed in ${TEST_TIME}s, exit code: $TEST_CODE" | log_output

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
  echo "[tritonbench bisect] ===== TritonBench Bisect Script Summary =====" 
  echo "[tritonbench bisect] Commit: $SHORT_COMMIT"
  echo "[tritonbench bisect] Build: ${BUILD_TIME}s (exit $BUILD_CODE)"
  echo "[tritonbench bisect] Test: ${TEST_TIME}s (exit $TEST_CODE)"
  echo "[tritonbench bisect] Result: $RESULT"
  if [ -n "$COMMIT_LOG" ]; then
    echo "[tritonbench bisect] Log: $COMMIT_LOG"
  fi
  echo "[tritonbench bisect] =============================================="
} | log_output


# Exit with test code for git bisect
exit $TEST_CODE
