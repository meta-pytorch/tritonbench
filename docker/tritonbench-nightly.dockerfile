ARG BASE_IMAGE=ghcr.io/actions/actions-runner:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
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
                            zlib1g-dev patch

# get switch-cuda utility
RUN sudo wget -q https://raw.githubusercontent.com/phohenecker/switch-cuda/master/switch-cuda.sh -O /usr/bin/switch-cuda.sh
RUN sudo chmod +x /usr/bin/switch-cuda.sh

# Create workspace
RUN sudo mkdir -p /workspace; sudo chown runner:runner /workspace

# We assume that the host NVIDIA driver binaries and libraries are mapped to the docker filesystem
# Install CUDA 12.8 build toolchains
RUN cd /workspace; mkdir -p pytorch-ci; cd pytorch-ci; wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cuda.sh
RUN cd /workspace/pytorch-ci; wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cudnn.sh || true && \
    wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_nccl.sh && \
    wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cusparselt.sh && \
    mkdir ci_commit_pins && cd ci_commit_pins && \
    wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/ci_commit_pins/nccl-cu12.txt
RUN sudo bash -c "set -x;export OVERRIDE_GENCODE=\"${OVERRIDE_GENCODE}\" OVERRIDE_GENCODE_CUDNN=\"${OVERRIDE_GENCODE_CUDNN}\"; cd /workspace/pytorch-ci; bash install_cuda.sh 12.8"

# Install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/Miniconda3-latest-Linux-x86_64.sh
RUN cd /workspace && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -u -p /workspace/miniconda3

# Test activate miniconda
RUN . /workspace/miniconda3/etc/profile.d/conda.sh && \
    conda activate base && \
    conda init && conda tos accept

RUN echo "\
. /workspace/miniconda3/etc/profile.d/conda.sh\n\
conda activate base\n\
export CONDA_HOME=/workspace/miniconda3\n\
export CUDA_HOME=/usr/local/cuda\n\
export PATH=\${CUDA_HOME}/bin:/home/runner/bin\${PATH:+:\${PATH}}\n\
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\n\
export LIBRARY_PATH=\${CUDA_HOME}/lib64\${LIBRARY_PATHPATH:+:\${LIBRARY_PATHPATH}}\n" >> /workspace/setup_instance.sh

RUN echo ". /workspace/setup_instance.sh\n" >> ${HOME}/.bashrc

# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/meta-pytorch/tritonbench /workspace/tritonbench

# Setup conda env and CUDA
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python tools/python_utils.py --create-conda-env ${CONDA_ENV} && \
    echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> /workspace/setup_instance.sh && \
    echo "conda activate \${CONDA_ENV}" >> /workspace/setup_instance.sh

RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    sudo python -m tools.cuda_utils --setup-cuda-softlink

# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python -m tools.cuda_utils --install-torch-deps && \
    python -m tools.cuda_utils --install-torch-nightly

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

# Tritonbench library build and test require libcuda.so.1
# which is from NVIDIA driver
RUN sudo apt update && sudo apt-get install -y libnvidia-compute-550 patchelf patch

# Workaround: installing Ninja from setup.py hits "Failed to decode METADATA with UTF-8" error
RUN . ${SETUP_SCRIPT} && pip install ninja

# Install PyTorch source
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install-pytorch-source.sh

# Install Tritonbench
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install.sh

# Remove NVIDIA driver library - they are supposed to be mapped at runtime
RUN sudo apt-get purge -y libnvidia-compute-550

# Clone the pytorch env as triton-main env, then compile triton main from source
RUN cd /workspace/tritonbench && \
    BASE_CONDA_ENV=${CONDA_ENV} CONDA_ENV=${CONDA_ENV_TRITON_MAIN} bash .ci/tritonbench/install-triton-main.sh

# Set run command
CMD ["bash", "/workspace/tritonbench/docker/entrypoint.sh"]
