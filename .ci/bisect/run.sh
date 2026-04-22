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
REGRESSION_THRESHOLD="${REGRESSION_THRESHOLD:-10}"

TRITONBENCH_DIR=$(dirname "$(readlink -f "$0")")/../..

parse_repro_cmdline() {
    local token

    eval "REPRO_CMD_TOKENS=(${REPRO_CMDLINE})"
    REPRO_CMD_ENV_ASSIGNMENTS=()
    REPRO_CMD=()

    local parsing_env_assignments=1
    for token in "${REPRO_CMD_TOKENS[@]}"; do
        if [ "${parsing_env_assignments}" -eq 1 ] && [[ "${token}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
            REPRO_CMD_ENV_ASSIGNMENTS+=("${token}")
        else
            parsing_env_assignments=0
            REPRO_CMD+=("${token}")
        fi
    done

    if [ "${#REPRO_CMD[@]}" -eq 0 ]; then
        echo "ERROR: REPRO_CMDLINE does not contain an executable command: ${REPRO_CMDLINE}"
        exit 1
    fi
}

export_repro_cmd_env() {
    local assignment

    for assignment in "${REPRO_CMD_ENV_ASSIGNMENTS[@]}"; do
        export "${assignment}"
    done
}

parse_repro_cmdline
export_repro_cmd_env
REPRO_CMDLINE="${REPRO_CMD[*]}"

echo "===== TritonBench Bisect Driver Script START ====="
echo "Good commit: ${GOOD_COMMIT}"
echo "Bad commit: ${BAD_COMMIT}"
echo "Virtual Env: ${CONDA_ENV}"
echo "Triton repo: ${TRITON_REPO}"
echo "Triton installation dir: ${TRITON_SRC_DIR}"
echo "Regression threshold: ${REGRESSION_THRESHOLD}"
echo "Functional bisect: ${FUNCTIONAL}"
echo "Repo command line: ${REPRO_CMDLINE}"
echo "Exported repro env: ${REPRO_CMD_ENV_ASSIGNMENTS[*]:-<none>}"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "=================================================="

# refresh triton repo to the latest commit
cd "${TRITON_SRC_DIR}"
git checkout main
git pull origin main
git submodule update --init --recursive

# switch back to tritonbench dir
cd "${TRITONBENCH_DIR}"

# Run the baseline commit first!
BISECT_LOG_DIR="${WORKSPACE_DIR}/bisect_logs"
BASELINE_LOG="${BISECT_LOG_DIR}/baseline.log"
mkdir -p "${BISECT_LOG_DIR}"
. .ci/triton/triton_install_utils.sh
# install triton of the good commit
checkout_triton_commit "${TRITON_SRC_DIR}" "${GOOD_COMMIT}"
install_triton "${TRITON_SRC_DIR}"
sudo ldconfig
cd "${TRITONBENCH_DIR}"
"${REPRO_CMD[@]}" 2>&1 | tee "${BASELINE_LOG}"

# pre-flight check: install and run on the bad commit to validate regression exists
checkout_triton_commit "${TRITON_SRC_DIR}" "${BAD_COMMIT}"
install_triton "${TRITON_SRC_DIR}"
sudo ldconfig
cd "${TRITONBENCH_DIR}"
# allow the regression detector to exit with error code
set +e
BASELINE_LOG="${BASELINE_LOG}" python ./.ci/bisect/regression_detector.py
PREFLIGHT_RC=$?
set -e
# if no regression, exit early and report error: this shouldn't happen
if [ ${PREFLIGHT_RC} -eq 0 ]; then
    echo "ERROR: No regression detected on bad commit (${BAD_COMMIT}) relative to good commit (${GOOD_COMMIT})."
    echo "The regression detector exited with 0, meaning the bad commit behaves the same as the good commit."
    echo "Please verify that your good_commit and bad_commit are correct, or adjust the REGRESSION_THRESHOLD (currently ${REGRESSION_THRESHOLD}%)."
    exit 1
elif [ ${PREFLIGHT_RC} -ne 1 ] && [ ${FUNCTIONAL} -ne 1 ]; then
    echo "WARNING: Pre-flight regression check exited with unexpected code ${PREFLIGHT_RC}."
    echo "This may indicate a build or environment issue. Proceeding with bisect anyway."
fi

# kick off the bisect!
TRITONPARSE_REPO_ARG=()
if [ "${CONDA_ENV}" = "meta-triton" ]; then
    TRITONPARSE_REPO_ARG=(--triton-repo meta)
fi

BASELINE_LOG="${BASELINE_LOG}" PER_COMMIT_LOG=1 USE_UV=1 CONDA_DIR="${WORKSPACE_DIR}/uv_venvs/${CONDA_ENV}" \
tritonparseoss bisect --triton-dir "${TRITON_SRC_DIR}" --test-script ./.ci/bisect/regression_detector.py \
    "${TRITONPARSE_REPO_ARG[@]}" --good ${GOOD_COMMIT} --bad ${BAD_COMMIT} --per-commit-log --log-dir "${BISECT_LOG_DIR}"
