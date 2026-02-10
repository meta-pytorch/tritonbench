if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

if [ -z "${WORKSPACE_DIR:-}" ]; then
  echo "ERROR: WORKSPACE_DIR is not set"
  exit 1
fi

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
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

VENV_NAME=triton-main
bash .ci/triton/install.sh --conda-env "${VENV_NAME}" \
        --repo triton-lang/triton --commit main --side single --nightly \
        --install-dir ${WORKSPACE_DIR}/triton-main ${CMD_SUFFIX}
