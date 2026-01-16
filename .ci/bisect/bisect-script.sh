#!/bin/bash
set -xu

if [ -z "${REPRO_CMDLINE:-}" ]; then
    echo "ERROR: REPRO_CMDLINE is not set"
    exit 1
fi

BASELINE_LOG="${OUTPUT_DIR}/baseline.log"

if [ -n "${BISECT_FUNCTIONAL:-}" ]; then
  REPRO_SUFFIX="--functional"
fi

if [ -n "${REGRESSION_THRESHOLD:-}" ]; then
  REPRO_SUFFIX="${REPRO_SUFFIX} --threshold ${REGRESSION_THRESHOLD}"
fi

if [ -f "${BASELINE_LOG}" ]; then
  REPRO_SUFFIX="${REPRO_SUFFIX} --baseline"
fi

# helper functions for triton installation
TRITONBENCH_DIR=$(dirname "$(readlink -f "$0")")/../..
cd "${TRITONBENCH_DIR}"
python ./.ci/bisect/regression_detector.py --repro \"${REPRO_CMDLINE}\" --baseline-log "${BASELINE_LOG}" ${REPRO_SUFFIX}
TEST_CODE=$?

# Exit with test code for git bisect
exit $TEST_CODE
