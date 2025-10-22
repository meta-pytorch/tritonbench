import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from typing import Dict, Optional

from pynvml import NVML_SUCCESS, nvmlDeviceGetHandleByIndex, nvmlInit, nvmlShutdown
from tritonbench.components.tasks.base import run_in_worker
from tritonbench.components.tasks.manager import ManagerTask

# query every 100 ms
DEFAULT_QUERY_INTERVAL = 0.1


@dataclass
class PowerEvent:
    timestamp: float
    power_limit: float
    power_draw_average: float
    power_draw_instant: float
    power_draw_max_limit: float
    temp_gpu: float
    temp_memory: float
    clock_current_sm: float
    clock_current_memory: float
    hw_thermal_slowdown: str
    sw_thermal_slowdown: str


@dataclass
class BenchmarkEvent:
    op: str
    backend: str
    event_type: str
    metrics: Dict[str, float]


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
        check_nvml_status(nvmlInit())
        self.handle = nvmlDeviceGetHandleByIndex(int(self.gpu_id))

    def start(self):
        while self.continue_monitoring:
            # check gpu power event
            time.sleep(self.sampling_interval)

    def output(self) -> str:
        pass
        # header = PowerEvent.fields()
        # for event in self.events:
        #     pass


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
        # with open(result_file, "w") as fp:
        #     fp.write(self.collector.output())


class PowerManagerTask(ManagerTask):
    def __init__(
        self,
        output_dir: str,
        query_interval: float,
        timeout: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(timeout, extra_env)
        assert output_dir, "output_dir must be specified for the power chart."
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.query_interval = query_interval

    def start_monitor(self) -> None:
        self.make_instance(
            "tritonbench.components.power.power_manager",
            None,
            "PowerManager",
        )
        self.set_manager_attribute("gpu_id", 0)
        self.set_manager_attribute("output_dir", str(self.output_dir))
        self.set_manager_attribute("query_interval", self.query_interval)
        self.start()

    def stop_monitor(self) -> None:
        self.stop()

    @run_in_worker(scoped=True)
    @staticmethod
    def start() -> None:
        pm = globals()["manager"]
        pm.start()

    @run_in_worker(scoped=True)
    @staticmethod
    def stop() -> None:
        pm = globals()["manager"]
        pm.stop()

    @run_in_worker(scoped=True)
    @staticmethod
    def pm_finalize() -> None:
        pm = globals()["manager"]
        pm.finalize()

    @staticmethod
    def create(output_dir) -> None:
        return PowerManagerTask(output_dir, DEFAULT_QUERY_INTERVAL)

    def finalize(self, metrics) -> None:
        # finalize the metrics
        # finalize the power manager task, and draw the charts
        self.pm_finalize()
