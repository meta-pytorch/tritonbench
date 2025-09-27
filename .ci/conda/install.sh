#!/bin/bash

set -ex

wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/Miniconda3-latest-Linux-x86_64.sh
cd /workspace
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -u -p /workspace/miniconda3

# Test
. /workspace/miniconda3/etc/profile.d/conda.sh
conda activate base
conda init
conda tos accept
