import logging
import os
import subprocess
from contextlib import contextmanager
from typing import Dict, List, Optional

import triton
import triton.language as tl

try:
    from tritonbench.utils.env_utils import is_hip, is_mtia
except ModuleNotFoundError:
    is_hip = lambda: False
    is_mtia = lambda: False

AMD_SLEEP_NS_PER_ITERATION = 3870

# Defer MTIA check to avoid triggering Triton driver initialization at import time
try:
    _is_mtia = is_mtia()
except Exception:
    _is_mtia = False

if _is_mtia:
    from tritonbench.utils.fb.mtia_utils import MTIA_COMPUTE_SPECS, MTIA_MEMORY_SPECS
else:
    MTIA_COMPUTE_SPECS = {}
    MTIA_MEMORY_SPECS = {}


# NVIDIA A100 GPU Spec:
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
NV_A100 = {
    "fp32": 19.5,
    "tf32": 156,
    "bf16": 312,
    "fp16": 312,
}

# NVIDIA H100 GPU Datasheet:
# "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
NV_H100 = {
    "fp32": 989 // 2,
    "tf32": 989 // 2,
    "bf16": 1979 // 2,
    "fp16": 1979 // 2,
    "fp8": 3958 // 2,
    "int8": 3958 // 2,
}

# https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf
AMD_MI300X = {
    "fp32": 1300 // 8,
    "tf32": 5200 // 8,
    "bf16": 10500 // 8,
    "fp16": 10500 // 8,
    "fp8": 20900 // 8,
    "int8": 20900 // 8,
}

# https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf
# individual blackwell gpu specs
NV_B200 = {
    "fp32": 2200 // 2,
    "tf32": 2200 // 2,
    "bf16": 4500 // 2,
    "fp16": 4500 // 2,
    "fp8": 9000 // 2,
    "int8": 9000 // 2,
}


HW_ROOFLINE_SPECS: Dict[
    bool, Dict[str, Dict[str, float]]
] = {  # true is compute bound false would be memory bound
    True: {
        "NVIDIA A100-SXM4-40GB": NV_A100,
        "NVIDIA A100-PG509-200": NV_A100,
        "NVIDIA H100": NV_H100,
        "NVIDIA H100 80GB HBM3": NV_H100,
        "AMD MI300X": AMD_MI300X,
        "NVIDIA B200": NV_B200,
        **MTIA_COMPUTE_SPECS,
    },
    False: {
        # https://www.nvidia.com/en-gb/data-center/h100
        # values in gbps
        "NVIDIA H100": 3350,
        "NVIDIA H100 80GB HBM3": 3350,
        # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf
        "AMD MI300X": 5300,
        # https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet
        "NVIDIA B200": 8000,
        **MTIA_MEMORY_SPECS,
    },
}

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

logger = logging.getLogger(__name__)


def get_gpu_id() -> int:
    """Get the first GPU ID from CUDA_VISIBLE_DEVICES."""
    return int(CUDA_VISIBLE_DEVICES.split(",")[0])


def get_gpu_device_name() -> str:
    try:
        import torch

        if is_hip():
            return get_amd_device_name()
        return torch.cuda.get_device_name()
    except Exception:
        return "None"


# =============================================================================
# NVIDIA Primitives
# =============================================================================

POWER_LIMIT = {
    "NVIDIA PG509-210": "330",
    "NVIDIA A100": "330",
    "NVIDIA H100": "650",
}
GRAPHIC_FREQ_LIMIT = {
    "NVIDIA PG509-210": "1410",
    "NVIDIA A100": "1410",
    "NVIDIA H100": "1980",
}
MEMORY_FREQ_LIMIT = {
    "NVIDIA H100": "1593",
}


def _get_gpu_name() -> str:
    import pynvml  # @manual=fbsource//third-party/pypi/nvidia-ml-py:nvidia-ml-py

    pynvml.nvmlInit()
    gpu_id = CUDA_VISIBLE_DEVICES.split(",")[0]
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
    return pynvml.nvmlDeviceGetName(handle)


def _set_pm():
    command = ["sudo", "nvidia-smi", "-i", CUDA_VISIBLE_DEVICES, "-pm", "1"]
    subprocess.check_call(command)


def _set_power(gpu_info: str):
    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        CUDA_VISIBLE_DEVICES,
        "--power-limit",
        POWER_LIMIT[gpu_info],
    ]
    subprocess.check_call(command)


def _set_clock(gpu_info: str):
    # lgc: lock gpu clocks
    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        CUDA_VISIBLE_DEVICES,
        "-lgc",
        GRAPHIC_FREQ_LIMIT[gpu_info],
    ]
    subprocess.check_call(command)
    # lmc: lock memory clocks
    if gpu_info in MEMORY_FREQ_LIMIT:
        command = [
            "sudo",
            "nvidia-smi",
            "-i",
            CUDA_VISIBLE_DEVICES,
            "-lmc",
            MEMORY_FREQ_LIMIT[gpu_info],
        ]
        subprocess.check_call(command)


def _set_clock_mhz(target_mhz: int) -> None:
    command = [
        "sudo",
        "nvidia-smi",
        "-i",
        CUDA_VISIBLE_DEVICES,
        "-lgc",
        str(target_mhz),
    ]
    subprocess.check_call(command)


def _maybe_set_app_clocks(gpu_info: str):
    graphic_freq = GRAPHIC_FREQ_LIMIT.get(gpu_info, None)
    memory_freq = MEMORY_FREQ_LIMIT.get(gpu_info, None)
    if graphic_freq and memory_freq:
        command = [
            "sudo",
            "nvidia-smi",
            "-i",
            CUDA_VISIBLE_DEVICES,
            "-ac",
            f"{memory_freq},{graphic_freq}",
        ]
        subprocess.check_call(command)


def _reset_clock(gpu_info: str):
    # rgc: reset gpu clocks
    command = ["sudo", "nvidia-smi", "-i", CUDA_VISIBLE_DEVICES, "-rgc"]
    subprocess.check_call(command)


def _nvidia_smi_query(query: str, device_ids: Optional[List[str]] = None) -> List[str]:
    if device_ids:
        device_ids = [str(id) for id in device_ids]
        device_ids = ",".join(device_ids)
    id_selector = f"-i {device_ids}" if device_ids else ""
    values = (
        subprocess.check_output(
            f'nvidia-smi --query-gpu="{query}" {id_selector} --format=csv,noheader,nounits',
            shell=True,
        )
        .strip()
        .decode()
        .split("\n")
    )
    return values


def get_nvidia_gpu_states() -> Dict[str, List[str]]:
    results = {}
    device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    # get power
    raw_metrics = _nvidia_smi_query(
        "power.draw.average,power.draw.instant,temperature.gpu,temperature.memory,"
        "clocks.current.sm,clocks.current.memory,"
        "clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown",
        device_ids,
    )
    results["power.draw.average"] = ",".join(
        metric.split(",")[0].strip() for metric in raw_metrics
    )
    results["power.draw.instant"] = ",".join(
        metric.split(",")[1].strip() for metric in raw_metrics
    )
    results["temperature.gpu"] = ",".join(
        metric.split(",")[2].strip() for metric in raw_metrics
    )
    results["temperature.memory"] = ",".join(
        metric.split(",")[3].strip() for metric in raw_metrics
    )
    results["clocks.current.sm"] = ",".join(
        metric.split(",")[4].strip() for metric in raw_metrics
    )
    results["clocks.current.memory"] = ",".join(
        metric.split(",")[5].strip() for metric in raw_metrics
    )
    results["hw_thermal_slowdown"] = ",".join(
        metric.split(",")[6].strip() for metric in raw_metrics
    )
    results["sw_thermal_slowdown"] = ",".join(
        metric.split(",")[7].strip() for metric in raw_metrics
    )
    return results


def has_nvidia_smi() -> bool:
    try:
        subprocess.check_output("nvidia-smi")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


# =============================================================================
# AMD Primitives
# =============================================================================

AMD_DEVICE_NAME_MAPPING = {
    (9, 4): "AMD MI300X",
    (9, 5): "AMD MI350X",
}

AMD_POWER_LIMIT = {
    "AMD MI300X": 750,
    "AMD Instinct MI300X": 750,
    "AMD MI350X": 900,
    "AMD Instinct MI350X": 900,
}

AMD_GRAPHIC_FREQ_LIMIT = {
    "AMD MI300X": 2100,
    "AMD Instinct MI300X": 2100,
    "AMD MI350X": 2200,
    "AMD Instinct MI350X": 2200,
}


def _match_amd_device_name(gpu_name: str, lookup: Dict) -> Optional[str]:
    """Match a raw AMD device name to a key in a lookup dict using substring matching.

    E.g. 'AMD Instinct MI300X' matches 'AMD MI300X'.
    """
    if gpu_name in lookup:
        return gpu_name
    for key in lookup:
        if key in gpu_name or gpu_name in key:
            return key
    return None


def _get_amd_gpu_id() -> int:
    """Get the AMD GPU ID from environment variables."""
    return int(
        os.environ.get(
            "HIP_VISIBLE_DEVICES",
            os.environ.get("ROCR_VISIBLE_DEVICES", "0"),
        ).split(",")[0]
    )


def get_amd_device_name() -> str:
    import torch
    from tritonbench.utils.env_utils import is_hip

    assert is_hip(), "get_amd_device_name() is only supported on AMD GPUs"
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name()
    if not device_name == "AMD Radeon Graphics":
        return device_name

    # if device name is "AMD Radeon Graphics", we need to infer the actual device name from gfx arch
    gcn_arch_major = torch.cuda.get_device_properties(current_device).major
    gcn_arch_minor = torch.cuda.get_device_properties(current_device).minor
    assert (gcn_arch_major, gcn_arch_minor) in AMD_DEVICE_NAME_MAPPING, (
        f"Unsupported AMD GCN Arch {gcn_arch_major}.{gcn_arch_minor}"
    )
    return AMD_DEVICE_NAME_MAPPING[(gcn_arch_major, gcn_arch_minor)]


def _amd_lock_clock(gpu_id: int, target_mhz: int) -> None:
    """Lock AMD GPU clock using perf determinism mode via rocm-smi.

    Equivalent to: sudo rocm-smi -d <gpu_id> --setperfdeterminism <mhz>
    """
    command = [
        "sudo",
        "rocm-smi",
        "-d",
        str(gpu_id),
        "--setperfdeterminism",
        str(target_mhz),
    ]
    subprocess.check_call(command)
    logger.info(
        f"[tritonbench] AMD GPU {gpu_id}: locked clock to {target_mhz} MHz (perf determinism)"
    )


def _amd_set_power_cap(gpu_id: int, power_watts: int) -> None:
    """Set AMD GPU power cap in watts via rocm-smi.

    Equivalent to: sudo rocm-smi -d <gpu_id> --setpoweroverdrive <watts>
    """
    command = [
        "sudo",
        "rocm-smi",
        "-d",
        str(gpu_id),
        "--setpoweroverdrive",
        str(power_watts),
    ]
    subprocess.check_call(command)
    logger.info(f"[tritonbench] AMD GPU {gpu_id}: set power cap to {power_watts} W")


def _amd_reset_clocks(gpu_id: int) -> None:
    """Reset AMD GPU clocks and power overdrive via rocm-smi."""
    subprocess.check_call(
        ["sudo", "rocm-smi", "-d", str(gpu_id), "--resetperfdeterminism"]
    )
    subprocess.check_call(
        ["sudo", "rocm-smi", "-d", str(gpu_id), "--resetpoweroverdrive"]
    )
    logger.info(f"[tritonbench] AMD GPU {gpu_id}: reset clocks and power")


@contextmanager
def gpu_lockdown(enabled=True, target_clock_mhz: Optional[int] = None):
    try:
        if enabled:
            if is_hip():
                gpu_id = _get_amd_gpu_id()
                gpu_name = get_amd_device_name()
                matched_name = _match_amd_device_name(gpu_name, AMD_POWER_LIMIT)
                logger.info(f"[tritonbench] Locking down AMD GPU {gpu_id} ({gpu_name})")
                assert matched_name is not None, (
                    f"Unsupported AMD GPU {gpu_name}. "
                    f"Supported: {list(AMD_POWER_LIMIT.keys())}"
                )
                _amd_set_power_cap(gpu_id, AMD_POWER_LIMIT[matched_name])
                clock_mhz = target_clock_mhz or AMD_GRAPHIC_FREQ_LIMIT[matched_name]
                _amd_lock_clock(gpu_id, clock_mhz)
            else:
                logger.info(f"[tritonbench] Locking down GPU {CUDA_VISIBLE_DEVICES}")
                gpu_name = _get_gpu_name()
                assert gpu_name in POWER_LIMIT, f"Unsupported GPU {gpu_name}"
                _set_pm()
                _set_power(gpu_name)
                if target_clock_mhz and target_clock_mhz > 0:
                    _set_clock_mhz(target_clock_mhz)
                else:
                    _set_clock(gpu_name)
                _maybe_set_app_clocks(gpu_name)
        yield
    finally:
        if enabled:
            if is_hip():
                gpu_id = _get_amd_gpu_id()
                _amd_reset_clocks(gpu_id)
            else:
                gpu_name = _get_gpu_name()
                _reset_clock(gpu_name)


@triton.jit
def sleep_amd(sleep_ns: tl.constexpr = 1000000):
    """
    AMD GPU sleep using s_sleep instruction.

    Each iteration of s_sleep 127 sleeps for ~127*64 = 8,128 clock cycles.
    On MI300X @ 2.1 GHz, this is approximately 3.87 μs per iteration.
    On MI350X @ 2.2 GHz, this is approximately 3.69 μs per iteration.

    Args:
        sleep_ns: Target sleep duration in nanoseconds.
                 Default 1000000 (1ms).

    Note:
        Timing is approximate and varies with GPU clock frequency.
    """
    # Calculate iterations: sleep_ns / 3870 ns per iteration
    num_iterations: tl.constexpr = max(1, sleep_ns // AMD_SLEEP_NS_PER_ITERATION)
    for _ in range(num_iterations):
        tl.inline_asm_elementwise(
            "s_sleep 127",
            "=r",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
