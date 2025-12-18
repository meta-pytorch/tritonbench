ARG BASE_IMAGE=ghcr.io/actions/actions-runner:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV CONDA_ENV_META_TRITON=meta-triton
ENV WORKSPACE_DIR=/workspace
ENV SETUP_SCRIPT=/workspace/setup_instance.sh

# ARG OVERRIDE_GENCODE="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a"
# ARG OVERRIDE_GENCODE_CUDNN="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a"
ARG TRITONBENCH_BRANCH=${TRITONBENCH_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

RUN sudo apt-get -y update && sudo apt -y update
RUN sudo apt-get install -y git jq gcc g++ \
                            vim wget curl ninja-build cmake \
                            libgl1-mesa-glx libsndfile1-dev kmod libxml2-dev libxslt1-dev \
                            libsdl2-dev libsdl2-2.0-0 \
                            zlib1g-dev patch patchelf

# get switch-cuda utility
# RUN sudo wget -q https://raw.githubusercontent.com/phohenecker/switch-cuda/master/switch-cuda.sh -O /usr/bin/switch-cuda.sh
# RUN sudo chmod +x /usr/bin/switch-cuda.sh

# Create workspace
# RUN sudo mkdir -p /workspace; sudo chown runner:runner /workspace

# We assume that the host NVIDIA driver binaries and libraries are mapped to the docker filesystem
# Install CUDA 12.8 build toolchains (only useful for bisection)
# RUN cd /workspace; mkdir -p pytorch-ci; cd pytorch-ci; wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cuda.sh
# RUN cd /workspace/pytorch-ci; wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cudnn.sh || true && \
#     wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_nccl.sh && \
#     wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cusparselt.sh && \
#     mkdir ci_commit_pins && cd ci_commit_pins && \
#     wget https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/ci_commit_pins/nccl-cu12.txt
# RUN sudo bash -c "set -x;export OVERRIDE_GENCODE=\"${OVERRIDE_GENCODE}\" OVERRIDE_GENCODE_CUDNN=\"${OVERRIDE_GENCODE_CUDNN}\"; cd /workspace/pytorch-ci; bash install_cuda.sh 12.8"

# Create workspace and permission check
RUN sudo mkdir -p /workspace; sudo chown $(whoami):$(id -gn) /workspace; touch "${SETUP_SCRIPT}"

# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/meta-pytorch/tritonbench "${WORKSPACE_DIR}/tritonbench"

# Install and setup env
RUN cd /workspace/tritonbench && bash ./.ci/tritonbench/setup-env.sh --cuda

RUN echo "\
export PATH=/home/runner/bin\${PATH:+:\${PATH}}\n" >> "${SETUP_SCRIPT}"

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

# Build meta-triton conda env
RUN cd /workspace/tritonbench && \
    bash .ci/triton/install.sh --conda-env "${CONDA_ENV_META_TRITON}" \
        --repo facebookexperimental/triton --commit 969e1f50c38c09f679f2e054511fe74da51c3eb3 --side single \
        --install-dir /workspace/meta-triton

# Test the install of meta-triton respects PTXAS_OPTIONS env var
RUN cd /workspace/tritonbench && \
    bash .ci/triton/test_ptxas_options.sh --conda-env "${CONDA_ENV_META_TRITON}"

# Install Helion in the meta-triton conda env
RUN cd /workspace/tritonbench && \
    bash .ci/helion/install.sh --conda-env "${CONDA_ENV_META_TRITON}"

# Output setup script for inspection
RUN cat "${SETUP_SCRIPT}"

# Set entrypoint
CMD ["bash", "/workspace/tritonbench/docker/entrypoint.sh"]
