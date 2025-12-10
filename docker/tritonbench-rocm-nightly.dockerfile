# Build ROCM base docker file
# Base image is rocm/pytorch:latest (on top of ubuntu 24.04)
ARG BASE_IMAGE=rocm/pytorch:latest

FROM ${BASE_IMAGE}

ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV SETUP_SCRIPT=/workspace/setup_instance.sh
ARG TRITONBENCH_BRANCH=${TRITONBENCH_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

RUN mkdir -p /workspace; touch "${SETUP_SCRIPT}"


# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/meta-pytorch/tritonbench /workspace/tritonbench


# Install and setup miniconda
RUN cd /workspace/tritonbench && bash ./.ci/conda/install.sh


# Setup SETUP_SCRIPT
RUN echo "\
. /workspace/miniconda3/etc/profile.d/conda.sh\n\
conda activate base\n\
export CONDA_HOME=/workspace/miniconda3\n" > "${SETUP_SCRIPT}"

RUN echo ". /workspace/setup_instance.sh\n" >> ${HOME}/.bashrc


# Setup conda env
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python tools/python_utils.py --create-conda-env ${CONDA_ENV} && \
    echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> "${SETUP_SCRIPT}" && \
    echo "conda activate \${CONDA_ENV}" >> "${SETUP_SCRIPT}"


# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python -m tools.cuda_utils --install-torch-deps && \
    python -m tools.cuda_utils --install-torch-nightly --hip


# Install Tritonbench
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install.sh


# Install PyTorch source
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install-pytorch-source.sh


# Build triton-main conda env
RUN cd /workspace/tritonbench && \
    bash .ci/triton/install.sh --conda-env "${CONDA_ENV_TRITON_MAIN}" \
        --repo triton-lang/triton --commit main --side single --nightly \
        --install-dir /workspace/triton-main


# Output setup script for inspection
RUN cat "${SETUP_SCRIPT}"

# Set entrypoint
CMD ["bash", "/workspace/tritonbench/docker/entrypoint.sh"]
