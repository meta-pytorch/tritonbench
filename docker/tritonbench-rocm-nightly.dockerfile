# Build ROCM base docker file
# Base image is rocm/pytorch:latest (on top of ubuntu 24.04)
ARG BASE_IMAGE=rocm/pytorch:latest

FROM ${BASE_IMAGE}

ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV WORKSPACE_DIR=/workspace
ENV SETUP_SCRIPT=/workspace/setup_instance.sh
# Use UV for Python venv
ENV UV_VENV_DIR=${WORKSPACE_DIR}/uv_venvs
ARG TRITONBENCH_BRANCH=${TRITONBENCH_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

# Create workspace and permission check
RUN sudo mkdir -p /workspace; sudo chown $(whoami):$(id -gn) /workspace; touch "${SETUP_SCRIPT}"

# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/meta-pytorch/tritonbench /workspace/tritonbench

# Install and setup env
# AMD will segfault on MI350 due to an LLVM bug: [llvm/llvm-project#193499](https://github.com/llvm/llvm-project/pull/193499)
# commit 90cd5e2abb74c on llvm-head branch fixes the segfault
RUN cd /workspace/tritonbench && bash ./.ci/tritonbench/setup-env.sh --hip --triton-main --meta-triton --triton-main-commit f7c1d69401e9f09050451f30776562954b05e850

# Output setup script for inspection
RUN cat "${SETUP_SCRIPT}"

# Set entrypoint
CMD ["bash", "/workspace/tritonbench/docker/entrypoint.sh"]
