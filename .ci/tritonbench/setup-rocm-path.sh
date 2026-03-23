#/usr/bin bash

set -xeuo pipefail

if [ -z "${WORKSPACE_DIR:-}" ]; then
    export WORKSPACE_DIR=/workspace
fi

if [ -z "${SETUP_SCRIPT:-}" ]; then
    export SETUP_SCRIPT=${WORKSPACE_DIR}/setup_instance.sh
fi

# add rocm lib to setup script, if exists
# some pytorch nightly version require rocprofiler-sdk.so
if [ -e "/opt/rocm/lib" ]; then
   echo "export LD_LIBRARY_PATH=/opt/rocm/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> "${SETUP_SCRIPT}"
   if [ ! -e "/opt/rocm/lib/librocprofiler-sdk.so.1" ]; then
       sudo cp /opt/rocm/lib/librocprofiler-sdk.so /opt/rocm/lib/librocprofiler-sdk.so.1
   fi
fi
