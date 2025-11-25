#!/usr/bin/env bash
# Script to tune NVIDIA B200 GPU
# To stablize performance

set -ex

GPU_ID=0
POWER_CAP=750

MAX_POWER=$(nvidia-smi --query-gpu=power.max_limit  --format=csv,noheader,nounits -i $GPU_ID)
MAX_SM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits  -i $GPU_ID)
MAX_MEM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits  -i $GPU_ID)
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_ID | head -n1 | awk '{print $2}')

if [[ "$GPU_MODEL" == "H100" ]]; then
    DESIRED_POWER=500
elif [[ "$GPU_MODEL" == "GB200" ]]; then
    DESIRED_POWER=1200
elif [[ "$GPU_MODEL" == "B200" ]]; then
    DESIRED_POWER=750
else
    DESIRED_POWER=500
fi

echo "→ Locking power cap to $POWER_CAP W, SM clock to $MAX_SM_CLOCK MHz, and memory clock to $MAX_MEM_CLOCK MHz on GPU $GPU_ID"

(
    sudo nvidia-smi -i "$GPU_ID" -pm 1
    sudo nvidia-smi --power-limit=$POWER_CAP -i "$GPU_ID"
    sudo nvidia-smi -lgc $MAX_SM_CLOCK -i "$GPU_ID"
    sudo nvidia-smi -lmc $MAX_MEM_CLOCK -i "$GPU_ID"
    sudo nvidia-smi -ac $MAX_MEM_CLOCK,$MAX_SM_CLOCK -i "$GPU_ID"
) >/dev/null

