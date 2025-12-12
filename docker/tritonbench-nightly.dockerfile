ARG BASE_IMAGE=ghcr.io/actions/actions-runner:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV CONDA_ENV_META_TRITON=meta-triton
ENV SETUP_SCRIPT=/workspace/setup_instance.sh
ARG OVERRIDE_GENCODE="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a"
ARG OVERRIDE_GENCODE_CUDNN="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a"
ARG TRITONBENCH_BRANCH=${TRITONBENCH_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

RUN sudo apt-get -y update && sudo apt -y update
RUN sudo apt-get install -y git jq gcc g++ \
                            vim wget curl ninja-build cmake \
                            libgl1-mesa-glx libsndfile1-dev kmod libxml2-dev libxslt1-dev \
                            libsdl2-dev libsdl2-2.0-0 \
                            zlib1g-dev patch patchelf

# get switch-cuda utility
RUN sudo wget -q https://raw.githubusercontent.com/phohenecker/switch-cuda/master/switch-cuda.sh -O /usr/bin/switch-cuda.sh
RUN sudo chmod +x /usr/bin/switch-cuda.sh

# Create workspace
RUN sudo mkdir -p /workspace; sudo chown runner:runner /workspace

# We assume that the host NVIDIA driver binaries and libraries are mapped to the docker filesystem
# Install CUDA 12.8 build toolchains (only useful for bisection)
# RUN cd /workspace; mkdir -p pytorch-ci; cd pytorch-ci; wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cuda.sh
# RUN cd /workspace/pytorch-ci; wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cudnn.sh || true && \
#     wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_nccl.sh && \
#     wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cusparselt.sh && \
#     mkdir ci_commit_pins && cd ci_commit_pins && \
#     wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/ci_commit_pins/nccl-cu12.txt
# RUN sudo bash -c "set -x;export OVERRIDE_GENCODE=\"${OVERRIDE_GENCODE}\" OVERRIDE_GENCODE_CUDNN=\"${OVERRIDE_GENCODE_CUDNN}\"; cd /workspace/pytorch-ci; bash install_cuda.sh 12.8"

# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/meta-pytorch/tritonbench /workspace/tritonbench

# Install and setup miniconda
RUN cd /workspace/tritonbench && bash ./.ci/conda/install.sh

RUN echo "\
. /workspace/miniconda3/etc/profile.d/conda.sh\n\
conda activate base\n\
export CONDA_HOME=/workspace/miniconda3\n\
export PATH=/home/runner/bin\${PATH:+:\${PATH}}\n" >> /workspace/setup_instance.sh

RUN echo ". /workspace/setup_instance.sh\n" >> ${HOME}/.bashrc

# Setup conda env
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python tools/python_utils.py --create-conda-env ${CONDA_ENV} && \
    echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> /workspace/setup_instance.sh && \
    echo "conda activate \${CONDA_ENV}" >> /workspace/setup_instance.sh

# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python -m tools.cuda_utils --install-torch-deps && \
    python -m tools.cuda_utils --install-torch-nightly --cuda

# Setup libcublas path to LD_LIBRARY_PATH
RUN . ${SETUP_SCRIPT} && \
    export PYTORCH_FILE_PATH=$(python -c "import torch; print(torch.__file__)") && \
    export NVIDIA_LIB_PATH=$(realpath $(dirname ${PYTORCH_FILE_PATH})/../nvidia/cublas/lib) && \
    cd ${NVIDIA_LIB_PATH} && \
    ln -s libcublas.so.* libcublas.so && ln -s libcublasLt.so.* libcublasLt.so &&  ln -s libnvblas.so.* libnvblas.so && \
    echo "export LD_LIBRARY_PATH=${NVIDIA_LIB_PATH}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n" >> /workspace/setup_instance.sh

# Check the installed version of nightly if needed
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    if [ "${FORCE_DATE}" = "skip_check" ]; then \
        echo "torch version check skipped"; \
    elif [ -z "${FORCE_DATE}" ]; then \
        FORCE_DATE=$(date '+%Y%m%d') \
        python -m tools.cuda_utils --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    else \
        python -m tools.cuda_utils --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    fi

# Workaround: installing Ninja from setup.py hits "Failed to decode METADATA with UTF-8" error
RUN . ${SETUP_SCRIPT} && pip install ninja

# Install PyTorch source
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install-pytorch-source.sh

# Install Tritonbench
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install.sh

# Build triton-main conda env
RUN cd /workspace/tritonbench && \
    bash .ci/triton/install.sh --conda-env "${CONDA_ENV_TRITON_MAIN}" \
        --repo triton-lang/triton --commit main --side single --nightly \
        --install-dir /workspace/triton-main

# Build meta-triton conda env
RUN cd /workspace/tritonbench && \
    bash .ci/triton/install.sh --conda-env "${CONDA_ENV_META_TRITON}" \
        --repo facebookexperimental/triton --commit b939601a9a376342985ab27bc649e02d4288afc6 --side single \
        --install-dir /workspace/meta-triton

# Install Helion in the meta-triton conda env
RUN cd /workspace/tritonbench && \
    bash .ci/helion/install.sh --conda-env "${CONDA_ENV_META_TRITON}"

# Output setup script for inspection
RUN cat "${SETUP_SCRIPT}"

# Set entrypoint
CMD ["bash", "/workspace/tritonbench/docker/entrypoint.sh"]
