from typing import List, Tuple

import torch


def validate_device_ids(device_ids: List[int]) -> None:
    """Validate that all specified CUDA device IDs are available on the current node.

    Raises RuntimeError if CUDA is not available or if any device ID is invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is not available on this node, cannot use devices: {device_ids}"
        )
    num_gpus = torch.cuda.device_count()
    unavailable = [d for d in device_ids if d >= num_gpus]
    if unavailable:
        raise RuntimeError(
            f"CUDA device(s) {unavailable} not available. "
            f"This node has {num_gpus} GPU(s) (IDs 0-{num_gpus - 1})."
        )


def parse_device_range(device_str: str) -> List[int]:
    """Parse a CUDA_VISIBLE_DEVICES-style device string into a list of device IDs.

    Supports individual IDs and ranges separated by commas.
    Examples:
        "0"       -> [0]
        "0-2,5"   -> [0, 1, 2, 5]
        "7,3,1"   -> [7, 3, 1]
        "0-0"     -> [0]
    """
    if not device_str or not device_str.strip():
        raise ValueError("Device string must not be empty")

    device_ids = []
    for part in device_str.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid device string: '{device_str}' (empty segment)")
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise ValueError(
                    f"Invalid range '{part}' in device string '{device_str}'"
                )
            try:
                start = int(bounds[0])
                end = int(bounds[1])
            except ValueError:
                raise ValueError(
                    f"Invalid range '{part}' in device string '{device_str}': "
                    "bounds must be integers"
                )
            if start < 0 or end < 0:
                raise ValueError(
                    f"Invalid range '{part}' in device string '{device_str}': "
                    "device IDs must be non-negative"
                )
            if start > end:
                raise ValueError(
                    f"Invalid range '{part}' in device string '{device_str}': "
                    "start must be <= end"
                )
            device_ids.extend(range(start, end + 1))
        else:
            try:
                device_id = int(part)
            except ValueError:
                raise ValueError(
                    f"Invalid device ID '{part}' in device string '{device_str}': "
                    "must be an integer"
                )
            if device_id < 0:
                raise ValueError(
                    f"Invalid device ID '{part}' in device string '{device_str}': "
                    "device IDs must be non-negative"
                )
            device_ids.append(device_id)

    if not device_ids:
        raise ValueError(f"No device IDs parsed from '{device_str}'")

    return device_ids


MIN_INPUTS_PER_DEVICE = 10


def compute_input_shards(
    total_inputs: int,
    num_devices: int,
    min_inputs_per_device: int = MIN_INPUTS_PER_DEVICE,
) -> List[Tuple[int, int]]:
    """Compute per-device input shards for evenly distributing inputs.

    Returns a list of (input_id_start, num_inputs) tuples, one per device.
    Remainder inputs are distributed to the first devices.

    If total_inputs / num_devices < min_inputs_per_device, the number of
    devices is reduced so each device gets at least min_inputs_per_device
    inputs (with at least 1 device always used).

    Examples:
        compute_input_shards(100, 4) -> [(0,25), (25,25), (50,25), (75,25)]
        compute_input_shards(10, 3)  -> [(0,10)]  # 3 devices capped to 1
        compute_input_shards(25, 4)  -> [(0,13), (13,12)]  # 4 devices capped to 2
    """
    if total_inputs <= 0:
        raise ValueError(f"total_inputs must be positive, got {total_inputs}")
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")

    if min_inputs_per_device > 0:
        max_devices = max(1, total_inputs // min_inputs_per_device)
        num_devices = min(num_devices, max_devices)

    base_size = total_inputs // num_devices
    remainder = total_inputs % num_devices

    shards = []
    offset = 0
    for i in range(num_devices):
        shard_size = base_size + (1 if i < remainder else 0)
        shards.append((offset, shard_size))
        offset += shard_size

    return shards
