#!/bin/bash

set -xeuo pipefail

# Print usage
usage() {
    echo "Usage: $0 --conda-env <env-name>"
    exit 1
}


# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --conda-env) CONDA_ENV="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

# Validate arguments
if [ -z "${CONDA_ENV}" ];  then
    echo "Missing required arguments: CONDA_ENV."
    usage
fi


. "${SETUP_SCRIPT}"

tritonbench_dir=$(dirname "$(readlink -f "$0")")/../..
cd ${tritonbench_dir}

python install.py --helion
