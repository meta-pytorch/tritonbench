#!/bin/bash
# Install the TPU (torch_tpu) stack: JAX/libtpu + a source build of torch_tpu
# pinned by .github/ci_commit_pins/torch_tpu.txt. torch_tpu is a private repo,
# so cloning needs an SSH key from GCP Secret Manager (same approach as the
# PyTorch CI). Mirrors pytorch/helion's TPU install.
#
# Infra-specific values (provision before enabling TPU CI):
#   TORCH_TPU_SSH_SECRET  - GCP Secret Manager secret holding the read-only
#                           deploy key for torch_tpu
#   TORCH_TPU_GCP_PROJECT - GCP project the secret lives in
set -euxo pipefail

TORCH_TPU_SSH_SECRET="${TORCH_TPU_SSH_SECRET:-torchtpu-read-key}"
TORCH_TPU_GCP_PROJECT="${TORCH_TPU_GCP_PROJECT:-ml-velocity-actions-testing}"
TORCH_TPU_COMMIT="$(cat .github/ci_commit_pins/torch_tpu.txt)"

# JAX + libtpu (Pallas/TPU runtime). libtpu 0.0.40 matches the pinned torch_tpu.
uv pip install \
  --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
  --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
  --pre \
  'jax==0.10.1' 'jaxlib==0.10.1' 'libtpu==0.0.40' 'tpu-info==0.7.1'

# Bazel (to build the torch_tpu wheel).
if ! command -v bazel &>/dev/null; then
  sudo curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel
  sudo chmod +x /usr/local/bin/bazel
fi

# Clone torch_tpu at the pin via the Secret Manager SSH key.
set +x
gcloud secrets versions access latest \
  --secret="${TORCH_TPU_SSH_SECRET}" \
  --project="${TORCH_TPU_GCP_PROJECT}" >/tmp/torch_tpu_ssh_key
set -x
chmod 600 /tmp/torch_tpu_ssh_key
GIT_SSH_COMMAND="ssh -i /tmp/torch_tpu_ssh_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=no" \
  git clone git@github.com:google-pytorch/torch_tpu.git /tmp/torch_tpu
rm -f /tmp/torch_tpu_ssh_key
pushd /tmp/torch_tpu
git checkout "${TORCH_TPU_COMMIT}"

# Build + install the torch_tpu wheel against the installed torch.
export TORCH_SOURCE=$(python -c "import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))")
export SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
bazel build -c opt //ci/wheel:torch_tpu_wheel \
  --define WHEEL_VERSION=0.1.0 \
  --define TORCH_SOURCE=local \
  --config=no_rbe \
  --action_env=PYTHONPATH="$TORCH_SOURCE:$SITE_PACKAGES" \
  --action_env=JAX_PLATFORMS=cpu
uv pip install bazel-bin/ci/wheel/*.whl
# torch_tpu pins libtpu==0.0.37; re-pin to 0.0.40 to match the JAX install above.
uv pip install libtpu==0.0.40
popd
rm -rf /tmp/torch_tpu

python -c "import torch, torch_tpu, sys; sys.exit(0 if torch.tpu.is_available() else 1)" \
  && echo "torch_tpu OK" || { echo "TPU not available"; exit 1; }
