#!/bin/bash

set -xeuo pipefail

# Remove Triton build-time downloads that are not needed at runtime.
#
# Triton caches the prebuilt LLVM, the NVIDIA toolchain and nlohmann/json under
# ${TRITON_HOME:-${HOME}}/.triton. These are only read while cmake configures and
# builds: ptxas & friends are copied into third_party/*/backend/ and libtriton.so is
# written to python/triton/_C/, both inside the Triton checkout. Dropping them saves
# a few GB per LLVM pin in the docker images.
#
# Run this in the same docker layer as the build, otherwise the space is not reclaimed.

# Mirrors get_triton_cache_path() in Triton's setup.py
TRITON_CACHE_PATH="${TRITON_HOME:-${HOME:?}}/.triton"

if [ ! -e "${TRITON_CACHE_PATH}" ]; then
    echo "No Triton cache at ${TRITON_CACHE_PATH}, nothing to clean up"
    exit 0
fi

# Keep the "cache" subdirectory: that is the runtime JIT cache, not a download.
for SUBDIR in llvm json nvidia; do
    rm -rf "${TRITON_CACHE_PATH:?}/${SUBDIR}"
done

du -sh "${TRITON_CACHE_PATH}"
