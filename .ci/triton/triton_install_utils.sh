# remove triton installations
remove_triton() {
    # delete the original triton directory
    TRITON_PKG_DIR=$(
        python - <<'PY'
import os

try:
    import triton
except ModuleNotFoundError:
    raise SystemExit(0)

print(os.path.dirname(triton.__file__))
PY
    )
    if [ -z "${TRITON_PKG_DIR}" ]; then
        return 0
    fi
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

clone_triton() {
    REPO=$1
    TRITON_INSTALL_DIR=$2
    TRITON_INSTALL_DIRNAME=$(basename "${TRITON_INSTALL_DIR}")
    TRITON_INSTALL_BASEDIR=$(dirname "${TRITON_INSTALL_DIR}")
    cd "${TRITON_INSTALL_BASEDIR}"
    # clone submodules (e.g., third_party/sleef required by the cpu backend)
    git clone --recurse-submodules "https://github.com/${REPO}.git" "${TRITON_INSTALL_DIRNAME}"
}

update_triton() {
    TRITON_INSTALL_DIR=$1
    cd "${TRITON_INSTALL_DIR}"
    git reset --hard
    git checkout main
    git pull origin main
    git submodule update --init --recursive
}

checkout_triton() {
    COMMIT=$1
    TRITON_INSTALL_DIR=$2
    NIGHTLY=$3
    cd "${TRITON_INSTALL_DIR}"
    git checkout "${COMMIT}"
    if [ "${NIGHTLY}" == "1" ]; then
        # truncate the branch to the earliest commit of the current day
        git checkout $(git rev-list --reverse --since=midnight HEAD | head -n 1)
    fi
    # the checked-out commit may point to different submodule revisions
    git submodule update --init --recursive
    cd -
}

# Some base images (e.g. rocm/pytorch) export LLVM_DIR or put their own LLVM on
# CMAKE_PREFIX_PATH. Both outrank the HINTS that MLIRConfig.cmake uses to find the
# prebuilt LLVM Triton downloads, so cmake silently loads the system LLVM instead.
# When that LLVM is built without the NVPTX target, find_package(MLIR) then fails with
# "The following imported targets are referenced, but are missing: LLVMNVPTXCodeGen
# LLVMNVPTXDesc LLVMNVPTXInfo".
drop_system_llvm_from_cmake_env() {
    unset LLVM_DIR
    unset LLVM_ROOT
    if [ -z "${CMAKE_PREFIX_PATH:-}" ]; then
        return 0
    fi
    KEPT_PREFIX_PATH=""
    IFS=':' read -r -a CMAKE_PREFIXES <<< "${CMAKE_PREFIX_PATH}"
    for CMAKE_PREFIX in "${CMAKE_PREFIXES[@]}"; do
        if [ -e "${CMAKE_PREFIX}/lib/cmake/llvm/LLVMConfig.cmake" ] || \
           [ -e "${CMAKE_PREFIX}/LLVMConfig.cmake" ]; then
            echo "Dropping ${CMAKE_PREFIX} from CMAKE_PREFIX_PATH: it would shadow Triton's LLVM"
            continue
        fi
        KEPT_PREFIX_PATH="${KEPT_PREFIX_PATH:+${KEPT_PREFIX_PATH}:}${CMAKE_PREFIX}"
    done
    if [ -z "${KEPT_PREFIX_PATH}" ]; then
        unset CMAKE_PREFIX_PATH
    else
        export CMAKE_PREFIX_PATH="${KEPT_PREFIX_PATH}"
    fi
}

install_triton() {
    TRITON_INSTALL_DIR=$1
    cd "${TRITON_INSTALL_DIR}"
    drop_system_llvm_from_cmake_env
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
