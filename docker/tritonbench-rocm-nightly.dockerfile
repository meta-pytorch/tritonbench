# Build ROCM base docker file
# We are not building AMD CI in a short term, but this could be useful
# for sharing benchmark results with AMD.
ARG BASE_IMAGE=rocm/pytorch:latest

FROM ${BASE_IMAGE}

ENV CONDA_ENV=pytorch
ENV CONDA_ENV_TRITON_MAIN=triton-main
ENV SETUP_SCRIPT=/workspace/setup_instance.sh
ARG TRITONBENCH_BRANCH=${TRITONBENCH_BRANCH:-main}
ARG FORCE_DATE=${FORCE_DATE}

RUN mkdir -p /workspace; touch "${SETUP_SCRIPT}"

RUN echo "\
. /opt/conda/etc/profile.d/conda.sh\n\
conda activate base\n\
export CONDA_HOME=/opt/conda\n" > "${SETUP_SCRIPT}"

RUN echo ". /workspace/setup_instance.sh\n" >> ${HOME}/.bashrc

# Checkout TritonBench and submodules
RUN git clone --recurse-submodules -b "${TRITONBENCH_BRANCH}" --single-branch \
    https://github.com/meta-pytorch/tritonbench /workspace/tritonbench

# Setup conda env
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python tools/python_utils.py --create-conda-env ${CONDA_ENV} && \
    echo "if [ -z \${CONDA_ENV} ]; then export CONDA_ENV=${CONDA_ENV}; fi" >> "${SETUP_SCRIPT}" && \
    echo "conda activate \${CONDA_ENV}" >> "${SETUP_SCRIPT}"


# Install PyTorch nightly and verify the date is correct
RUN cd /workspace/tritonbench && \
    . ${SETUP_SCRIPT} && \
    python -m tools.rocm_utils --install-torch-deps && \
    python -m tools.rocm_utils --install-torch-nightly


# Install Tritonbench
RUN cd /workspace/tritonbench && \
    bash .ci/tritonbench/install.sh
