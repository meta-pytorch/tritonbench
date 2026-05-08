usage() {
  echo "Usage: $0 [--commit <hash-or-ref>] [--no-build] [--no-clone]"
  exit 1
}

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

if [ -z "${WORKSPACE_DIR:-}" ]; then
  echo "ERROR: WORKSPACE_DIR is not set"
  exit 1
fi

COMMIT=main
COMMIT_SPECIFIED=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --commit)
            if [ -z "${2:-}" ]; then
              echo "ERROR: --commit requires a value"
              usage
            fi
            COMMIT="$2"
            COMMIT_SPECIFIED=1
            shift
            ;;
        --no-build) NO_BUILD="1"; ;;
        --no-clone) NO_CLONE="1"; ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done


CMD_SUFFIX=""
if [ -n "${NO_BUILD:-}" ]; then
  CMD_SUFFIX="--no-build $CMD_SUFFIX"
fi
if [ -n "${NO_CLONE:-}" ]; then
  CMD_SUFFIX="--no-clone $CMD_SUFFIX"
fi

NIGHTLY_ARG="--nightly"
if [ "${COMMIT_SPECIFIED}" -eq 1 ]; then
  NIGHTLY_ARG=""
fi


VENV_NAME=meta-triton
bash .ci/triton/install.sh --conda-env "${VENV_NAME}" \
        --repo facebookexperimental/triton --commit "${COMMIT}" --side single ${NIGHTLY_ARG} \
        --install-dir ${WORKSPACE_DIR}/meta-triton ${CMD_SUFFIX}
