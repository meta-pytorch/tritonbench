if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

if [ -z "${WORKSPACE_DIR:-}" ]; then
  echo "ERROR: WORKSPACE_DIR is not set"
  exit 1
fi

VENV_NAME=triton-main
bash .ci/triton/install.sh --conda-env "${VENV_NAME}" \
        --repo triton-lang/triton --commit main --side single --nightly \
        --install-dir ${WORKSPACE_DIR}/triton-main
