#!/bin/bash
set -xeuo pipefail

if [ -z "${SETUP_SCRIPT}" ]; then
  echo "ERROR: SETUP_SCRIPT is not set"
  exit 1
fi

if [ -z "$1" ]; then
  echo "ERROR: BENCHMARK_NAME must be set as the first argument."
  exit 1
fi

BENCHMARK_NAME=$1
shift

if [ -z "${CONDA_ENV}" ]; then
  echo "ERROR: CONDA_ENV is not set"
  exit 1
fi

tritonbench_dir=$(dirname "$(readlink -f "$0")")/../..
cd "${tritonbench_dir}"

# check if the current repo has "dubious ownership" issue
git config --global --add safe.directory '*'
git rev-parse --verify HEAD

echo "Running ${BENCHMARK_NAME} benchmark under conda env ${CONDA_ENV}"

. "${SETUP_SCRIPT}"

CONDA_ENV=${CONDA_ENV} python -m "benchmarks.${BENCHMARK_NAME}.run" --ci $@
