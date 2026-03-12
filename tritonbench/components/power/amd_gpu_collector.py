import logging
import os
import time

from tritonbench.components.power.power_manager import PowerEvent

logger = logging.getLogger(__name__)

DEFAULT_QUERY_INTERVAL = 0.01


class AMDGPUCollectorThread:
    """Background collector thread for AMD GPUs using amdsmi.

    Mirrors GPUCollectorThread interface: .start() blocking loop,
    .continue_monitoring flag, .events list of PowerEvent, .sampling_interval.
    """

    def __init__(self, gpu_id=None, query_interval=DEFAULT_QUERY_INTERVAL) -> None:
        self.gpu_id = (
            int(gpu_id)
            if gpu_id is not None
            else int(
                os.environ.get(
                    "HIP_VISIBLE_DEVICES",
                    os.environ.get("ROCR_VISIBLE_DEVICES", "0"),
                ).split(",")[0]
            )
        )
        self.continue_monitoring = True
        self.sampling_interval = query_interval
        self.events = []
        self.iter = []

        import amdsmi

        amdsmi.amdsmi_init()
        devices = amdsmi.amdsmi_get_processor_handles()
        if self.gpu_id < 0 or self.gpu_id >= len(devices):
            raise IndexError(
                f"GPU ID {self.gpu_id} is out of range. "
                f"Available AMD GPU devices: 0-{len(devices) - 1} "
                f"({len(devices)} total)"
            )
        self._device = devices[self.gpu_id]

    def start(self):
        import amdsmi

        device = self._device

        while self.continue_monitoring:
            loop_start = time.monotonic()
            timestamp = int(time.time_ns() / 1e3)

            # SM (GFX) clock
            try:
                clock_info = amdsmi.amdsmi_get_clock_info(
                    device, amdsmi.AmdSmiClkType.GFX
                )
                sm_clock = (
                    int(clock_info["cur_clk"])
                    if "cur_clk" in clock_info and clock_info["cur_clk"] != "N/A"
                    else (
                        int(clock_info["clk"]) if clock_info.get("clk") != "N/A" else 0
                    )
                )
            except Exception:
                sm_clock = 0

            # Memory clock
            try:
                mem_clock_info = amdsmi.amdsmi_get_clock_info(
                    device, amdsmi.AmdSmiClkType.MEM
                )
                mem_clock = (
                    int(mem_clock_info["cur_clk"])
                    if "cur_clk" in mem_clock_info
                    and mem_clock_info["cur_clk"] != "N/A"
                    else (
                        int(mem_clock_info["clk"])
                        if mem_clock_info.get("clk") != "N/A"
                        else 0
                    )
                )
            except Exception:
                mem_clock = 0

            # Power: get both draw and limit from a single call
            try:
                power_info = amdsmi.amdsmi_get_power_info(device)
                power_draw = power_info.get("current_socket_power", 0)
                power_limit = power_info.get("power_limit", 0)
            except Exception:
                power_draw = 0.0
                power_limit = 0.0

            # Temperature - use JUNCTION (not EDGE) for AMD
            try:
                gpu_temp = amdsmi.amdsmi_get_temp_metric(
                    device,
                    amdsmi.AmdSmiTemperatureType.JUNCTION,
                    amdsmi.AmdSmiTemperatureMetric.CURRENT,
                )
            except Exception:
                gpu_temp = 0.0

            # GPU and memory utilization from a single call
            try:
                activity = amdsmi.amdsmi_get_gpu_activity(device)
                gpu_util = activity.get("gfx_activity", 0)
                mem_util = activity.get("umc_activity", 0)
            except Exception:
                gpu_util = 0
                mem_util = 0

            # Memory used (convert from bytes to MB)
            try:
                mem_used_bytes = amdsmi.amdsmi_get_gpu_memory_usage(
                    device, amdsmi.AmdSmiMemoryType.VRAM
                )
                mem_used_mb = mem_used_bytes // (1024 * 1024)
            except Exception:
                mem_used_mb = 0

            self.events.append(
                PowerEvent(
                    timestamp=timestamp,
                    sm_clock=sm_clock,
                    mem_clock=mem_clock,
                    power_draw_instant=power_draw,
                    power_draw_current_limit=power_limit,
                    gpu_temp=gpu_temp,
                    gpu_utilization_pct=gpu_util,
                    memory_utilization_pct=mem_util,
                    memory_used_mb=mem_used_mb,
                )
            )

            # Adaptive sleep: subtract the time spent on API calls from the
            # target interval so the effective sample rate stays close to the
            # configured value.  amdsmi calls are ~0.375 ms each (vs ~0.03 ms
            # for NVML), so without compensation the actual interval overshoots
            # by ~3 ms on AMD hardware.
            elapsed = time.monotonic() - loop_start
            remaining = self.sampling_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        # Cleanup
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            pass
