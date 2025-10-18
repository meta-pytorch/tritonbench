import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlShutdown

class PowerEvent(dataclass):
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

class BenchmarkEvent(dataclass):
    op: str
    backend: str
    event_type: str
    metrics: Dict[str, float]

    
def check_nvml_status(nvml_status):
    if nvml_status != 0:
        raise RuntimeError("NVML initialization failed")
    
class GPUCollectorThread():
    def __init__(self, gpu_id=None) -> None:
        self.gpu_id = int(gpu_id) if gpu_id else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        # Assume Python GIL so not protecting this using Atomics
        self.continue_monitoring = True
        # Sampling interval in seconds 
        self.sampling_interval = 0.1
        self.events = []
        check_nvml_status(nvmlInit())
        check_nvml_status(nvmlDeviceGetHandleByIndex(self.gpu_id))

    def start(self) :
        while self.continue_monitoring:
            # check gpu power event
            time.sleep(self.sampling_interval)
    
    def output(self) -> str:
        header = PowerEvent.fields()
        for event in self.events:
            pass

class PowerManager():
    def __init__(self, gpu_id=None, output_dir=None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collector = GPUCollectorThread(gpu_id)

    def start(self) -> None:
        self._t = threading.Thread(target=self.collector.start)
        self._t.start()

    def stop(self) -> None:
        self.collector.continue_monitoring = False
        self._t.join()
        # flush results to file
        result_file = self.output_dir / "power_events.csv"
        with open(result_file, "w") as fp:
            fp.write(self.collector.output())
