import csv
import dataclasses
import os
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from typing import Dict, Optional

from pynvml import (
    NVML_CLOCK_ID_CURRENT,
    NVML_CLOCK_MEM,
    NVML_CLOCK_SM,
    NVML_FI_DEV_POWER_CURRENT_LIMIT,
    NVML_FI_DEV_POWER_INSTANT,
    NVML_SUCCESS,
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetClock,
    nvmlDeviceGetFieldValues,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPerformanceState,
    nvmlDeviceGetTemperature,
    nvmlInit,
    nvmlShutdown,
)
from tritonbench.components.power.charts import plot_latencies, plot_power_charts
from tritonbench.components.tasks.base import run_in_worker
from tritonbench.components.tasks.manager import ManagerTask

# query every 10 ms
DEFAULT_QUERY_INTERVAL = 0.01


@dataclass
class PowerEvent:
    timestamp: float
    sm_clock: float
    mem_clock: float
    power_draw_instant: float
    power_draw_current_limit: float
    gpu_temp: float


def check_nvml_status(nvml_status):
    if nvml_status:
        raise RuntimeError("NVML initialization failed")


class GPUCollectorThread:
    def __init__(self, gpu_id=None, query_interval=DEFAULT_QUERY_INTERVAL) -> None:
        self.gpu_id = (
            int(gpu_id) if gpu_id else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        )
        # Assume Python GIL so not protecting this using Atomics
        self.continue_monitoring = True
        # Sampling interval in seconds
        self.sampling_interval = query_interval
        self.events = []
        self.iter = []
        check_nvml_status(nvmlInit())

    def start(self):
        handle = nvmlDeviceGetHandleByIndex(int(self.gpu_id))
        while self.continue_monitoring:
            # check gpu power event
            sm_clock = nvmlDeviceGetClock(handle, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT)
            mem_clock = nvmlDeviceGetClock(
                handle, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT
            )
            power_info = nvmlDeviceGetFieldValues(
                handle, [NVML_FI_DEV_POWER_INSTANT, NVML_FI_DEV_POWER_CURRENT_LIMIT]
            )
            gpu_temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
            self.events.append(
                PowerEvent(
                    timestamp=int(time.time_ns() / 1e3),
                    sm_clock=sm_clock,
                    mem_clock=mem_clock,
                    power_draw_instant=power_info[0].value.uiVal / 1000.0,
                    power_draw_current_limit=power_info[1].value.uiVal / 1000.0,
                    gpu_temp=gpu_temp,
                )
            )
            time.sleep(self.sampling_interval)
        nvmlShutdown()


class PowerManager:
    def __init__(self) -> None:
        self.gpu_id = None
        self.output_dir = None
        self.query_interval = None

    def start(self) -> None:
        self.collector = GPUCollectorThread(self.gpu_id, self.query_interval)
        self._t = threading.Thread(target=self.collector.start)
        self._t.start()

    def stop(self) -> None:
        self.collector.continue_monitoring = False
        self._t.join()

    def finalize(self) -> None:
        # flush results to file
        result_file = os.path.join(self.output_dir, "power.csv")
        with open(result_file, "w", newline="") as csvfile:
            # Get the field names from the dataclass to use as CSV header
            fieldnames = [field.name for field in fields(PowerEvent)]

            # Create a DictWriter object
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")

            # Write the header row
            writer.writeheader()

            # Write each dataclass instance as a row in the CSV
            for event in self.collector.events:
                writer.writerow(asdict(event))


class PowerManagerTask(ManagerTask):
    def __init__(
        self,
        benchmark_name: str,
        gpu_id: int,
        output_dir: str,
        query_interval: float,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(timeout, extra_env)
        self.gpu_id = gpu_id
        self.benchmark_name = benchmark_name
        assert output_dir, "output_dir must be specified for the power chart."
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.query_interval = query_interval

    def start(self) -> None:
        self.make_instance(
            "tritonbench.components.power.power_manager",
            None,
            "PowerManager",
        )
        self.set_manager_attribute("gpu_id", self.gpu_id)
        self.set_manager_attribute("output_dir", str(self.output_dir))
        self.set_manager_attribute("query_interval", self.query_interval)
        self._start()

    def stop(self) -> None:
        self._stop()

    @run_in_worker(scoped=True)
    @staticmethod
    def _start() -> None:
        pm = globals()["manager"]
        pm.start()

    @run_in_worker(scoped=True)
    @staticmethod
    def _stop() -> None:
        pm = globals()["manager"]
        pm.stop()

    @run_in_worker(scoped=True)
    @staticmethod
    def _finalize() -> None:
        pm = globals()["manager"]
        pm.finalize()

    @staticmethod
    def create(
        benchmark_name, gpu_id, output_dir, query_interval=DEFAULT_QUERY_INTERVAL
    ) -> None:
        return PowerManagerTask(benchmark_name, gpu_id, output_dir, query_interval)

    def finalize(self, metrics) -> None:
        # finalize the power manager task
        self._finalize()
        # draw the latency charts
        plot_latencies(self.output_dir, self.gpu_id, metrics)
        # draw the power charts
        plot_power_charts(
            self.benchmark_name,
            self.gpu_id,
            self.output_dir,
            os.path.join(self.output_dir, "power.csv"),
        )
