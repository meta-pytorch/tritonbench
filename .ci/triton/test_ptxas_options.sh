set -xeuo pipefail

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

if [ -z "${CONDA_ENV}" ]; then
  echo "ERROR: CONDA_ENV is not set"
  exit 1
fi

. "${SETUP_SCRIPT}"

cd /workspace/tritonbench

python -c "from triton import knobs; assert hasattr(knobs.nvidia, 'ptxas_options')" || \
  (echo "ERROR: Triton does not have ptxas_options knob" && exit 1)
