# Build ROCM base docker file
# Base image is rocm/pytorch:latest (on top of ubuntu 24.04)
ARG BASE_IMAGE=rocm/pytorch:latest

FROM ${BASE_IMAGE}

ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV WORKSPACE_DIR=/workspace
ENV SETUP_SCRIPT=/workspace/setup_instance.sh
ARG TRITONBENCH_BRANCH=${TRITONBENCH_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

# Create workspace and permission check
RUN sudo mkdir -p /workspace; sudo chown $(whoami):$(id -gn) /workspace; touch "${SETUP_SCRIPT}"

# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/meta-pytorch/tritonbench /workspace/tritonbench

# Install and setup env
RUN cd /workspace/tritonbench && bash ./.ci/tritonbench/setup-env.sh --hip

# Output setup script for inspection
RUN cat "${SETUP_SCRIPT}"

# Set entrypoint
CMD ["bash", "/workspace/tritonbench/docker/entrypoint.sh"]
