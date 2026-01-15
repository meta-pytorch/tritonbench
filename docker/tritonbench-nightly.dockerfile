ARG BASE_IMAGE=ghcr.io/actions/actions-runner:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV CONDA_ENV_META_TRITON=meta-triton
ENV WORKSPACE_DIR=/workspace
ENV SETUP_SCRIPT=${WORKSPACE_DIR}/setup_instance.sh

# Use UV for Python venv
ENV UV_VENV_DIR=${WORKSPACE_DIR}/uv_venvs

# ARG OVERRIDE_GENCODE="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a"
# ARG OVERRIDE_GENCODE_CUDNN="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90a,code=sm_90a"
ARG TRITONBENCH_BRANCH=${TRITONBENCH_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

RUN sudo apt-get -y update && sudo apt -y update
RUN sudo apt-get install -y git jq gcc g++ \
                            vim wget curl ninja-build cmake \
                            libsndfile1-dev kmod libxml2-dev libxslt1-dev \
                            zlib1g-dev patch patchelf


# Create workspace and permission check
RUN sudo mkdir -p ${WORKSPACE_DIR}; sudo chown $(whoami):$(id -gn) ${WORKSPACE_DIR}; touch "${SETUP_SCRIPT}"
RUN echo "\
export WORKSPACE_DIR=${WORKSPACE_DIR}\n\
export PATH=/home/runner/bin:/home/runner/.local/bin\${PATH:+:\${PATH}}\n" >> "${SETUP_SCRIPT}"

# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/meta-pytorch/tritonbench "${WORKSPACE_DIR}/tritonbench"

# Install and setup env
RUN cd ${WORKSPACE_DIR}/tritonbench && bash ./.ci/tritonbench/setup-env.sh --cuda --triton-main --meta-triton --test-nvidia-driver

# Check the installed version of nightly if needed
RUN cd ${WORKSPACE_DIR}/tritonbench && \
    . ${SETUP_SCRIPT} && \
    if [ "${FORCE_DATE}" = "skip_check" ]; then \
        echo "torch version check skipped"; \
    elif [ -z "${FORCE_DATE}" ]; then \
        FORCE_DATE=$(date '+%Y%m%d') \
        python -m tools.cuda_utils --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    else \
        python -m tools.cuda_utils --check-torch-nightly-version --force-date "${FORCE_DATE}"; \
    fi

# Test the install of meta-triton respects PTXAS_OPTIONS env var
RUN cd "${WORKSPACE_DIR}"/tritonbench && \
    bash .ci/triton/test_ptxas_options.sh --conda-env "${CONDA_ENV_META_TRITON}"

# Install Helion in the triton-meta venv
RUN cd "${WORKSPACE_DIR}"/tritonbench && \
    bash .ci/helion/install.sh --conda-env "${CONDA_ENV_META_TRITON}"

# Output setup script for inspection
RUN cat "${SETUP_SCRIPT}"

# Set entrypoint
CMD ["bash", "/workspace/tritonbench/docker/entrypoint.sh"]
