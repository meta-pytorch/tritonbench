#!/usr/bin/env bash
# Script to tune NVIDIA B200 GPU
# To reset GPU status

# Reset GPU and Memory clocks
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc

# Restore the default power limit (750W)
sudo nvidia-smi -pl 750

# Disable persistent mode
sudo nvidia-smi -pm 0
