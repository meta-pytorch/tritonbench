set -x


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

PTXAS_OPTIONS="--apply-controls non-exist.bin" TRITONBENCH_RUN_CONFIG=$PWD/benchmarks/run_config/example_config.yaml python run.py &> /workspace/ptxas_run.log || true
PTXAS_ERROR_MESSAGE='ptxas fatal   : File 'non-exist.bin' could not be opened'

# Test that ptxas options are passed correctly to ptxas
# If so, the output file should contain an error message

if ! grep -q "$PTXAS_ERROR_MESSAGE" /workspace/ptxas_run.log; then
    echo "============== Error: ptxas options not passed correctly =============="
    echo ">>> Output file: "
    cat /workspace/ptxas_run.log
    rm /workspace/ptxas_run.log
    exit 1
else
    echo "============== PTXAS_OPTIONS config test passes  =============="
    echo ">>> Output file: "
    cat /workspace/ptxas_run.log
    rm /workspace/ptxas_run.log
    exit 0
fi 
