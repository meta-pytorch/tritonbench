# remove triton installations
remove_triton() {
    # delete the original triton directory
    TRITON_PKG_DIR=$(python -c "import triton; import os; print(os.path.dirname(triton.__file__))")
    # make sure all pytorch triton has been uninstalled
    if [ -n "${UV_VENV_DIR:-}" ]; then
        uv pip uninstall triton
        uv pip uninstall triton
        uv pip uninstall triton
    else
        pip uninstall -y triton
        pip uninstall -y triton
        pip uninstall -y triton
    fi
    rm -rf "${TRITON_PKG_DIR}"
}

checkout_triton() {
    REPO=$1
    COMMIT=$2
    TRITON_INSTALL_DIR=$3
    NIGHTLY=$4
    TRITON_INSTALL_DIRNAME=$(basename "${TRITON_INSTALL_DIR}")
    TRITON_INSTALL_BASEDIR=$(dirname "${TRITON_INSTALL_DIR}")
    cd "${TRITON_INSTALL_BASEDIR}"
    git clone "https://github.com/${REPO}.git" "${TRITON_INSTALL_DIRNAME}"
    cd "${TRITON_INSTALL_DIR}"
    git checkout "${COMMIT}"
    if [ "${NIGHTLY}" == "1" ]; then
        # truncate the branch to the earliest commit of the current day
        git checkout $(git rev-list --reverse --since=midnight HEAD | head -n 1)
    fi
}

install_triton() {
    TRITON_INSTALL_DIR=$1
    cd "${TRITON_INSTALL_DIR}"
    # install main triton
    if [ -n "${UV_VENV_DIR:-}" ]; then
        uv pip install ninja cmake wheel pybind11; # build-time dependencies
        uv pip install -r python/requirements.txt
        uv pip install -e .
    else
        pip install ninja cmake wheel pybind11; # build-time dependencies
        pip install -r python/requirements.txt
        pip install -e .
    fi
    cd -
}

checkout_triton_commit() {
    TRITON_INSTALL_DIR=$1
    COMMIT=$2
    cd "${TRITON_INSTALL_DIR}"
    git checkout "${COMMIT}"
    git submodule update --init --recursive
}
