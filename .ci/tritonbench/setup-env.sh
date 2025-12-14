#/usr/bin bash

set -xeuo pipefail

bash ./.ci/conda/install.sh

echo "\
. /workspace/miniconda3/etc/profile.d/conda.sh && \
conda activate base && \
export CONDA_HOME=/workspace/miniconda3 " >> ${SETUP_SCRIPT}

echo ". \${SETUP_SCRIPT}\n" >> ${HOME}/.bashrc

export CONDA_ENV=pytorch

. ${SETUP_SCRIPT}

python tools/python_utils.py --create-conda-env ${CONDA_ENV} && \
echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> /workspace/setup_instance.sh && \
echo "conda activate \${CONDA_ENV}" >> /workspace/setup_instance.sh

python -m tools.cuda_utils --install-torch-deps

python -m tools.cuda_utils --install-torch-nightly --hip

export PYTORCH_FILE_PATH=$(python -c "import torch; print(torch.__file__)")
export NVIDIA_LIB_PATH=$(realpath $(dirname ${PYTORCH_FILE_PATH})/../nvidia/cublas/lib)

if [ -e ${NVIDIA_LIB_PATH} ]; then
    cd ${NVIDIA_LIB_PATH}
    ln -s libcublas.so.* libcublas.so && ln -s libcublasLt.so.* libcublasLt.so &&  ln -s libnvblas.so.* libnvblas.so && \
    echo "export LD_LIBRARY_PATH=${NVIDIA_LIB_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n" >> /workspace/setup_instance.sh
    cd -
fi

pip install ninja

bash .ci/tritonbench/install-pytorch-source.sh

bash .ci/tritonbench/install.sh

CONDA_ENV_TRITON_MAIN=triton-main
bash .ci/triton/install.sh --conda-env "${CONDA_ENV_TRITON_MAIN}" \
        --repo triton-lang/triton --commit main --side single --nightly \
        --install-dir /workspace/triton-main

cat "${SETUP_SCRIPT}"

