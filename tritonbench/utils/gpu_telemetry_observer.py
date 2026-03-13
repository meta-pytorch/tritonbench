"""
GPU Telemetry Observer for real-time monitoring of GPU clock and power metrics.

Uses a dedicated background collector thread per GPU vendor (NVIDIA via
GPUCollectorThread, AMD via AMDGPUCollectorThread) that produces PowerEvent
objects, converted to GPUSample in the observer layer.

Usage:
    from tritonbench.utils.gpu_telemetry_observer import TelemetryContext

    with TelemetryContext(gpu_id=0, sample_interval_ms=10) as ctx:
        ctx.annotate("benchmark_start")
        run_benchmark()
        ctx.annotate("benchmark_end")

    ctx.save_csv("/tmp/telemetry.csv")
    ctx.plot("/tmp/telemetry.png")
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TELEMETRY_INTERVAL_MS = 10


@dataclass
class GPUSample:
    """A single GPU telemetry sample."""

    timestamp_ms: float
    clock_mhz: int
    memory_clock_mhz: int
    power_watts: float
    temperature_c: int
    memory_used_mb: int
    utilization_pct: int
    memory_utilization_pct: int


@dataclass
class GPUTelemetryData:
    """Collection of GPU telemetry samples with metadata."""

    gpu_id: int
    sample_interval_ms: float
    samples: List[GPUSample] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    annotations: List[Tuple[float, str]] = field(default_factory=list)

    def add_annotation(self, label: str) -> None:
        if self.start_time > 0:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.annotations.append((elapsed_ms, label))

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time > 0 else 0.0

    @property
    def timestamps(self) -> List[float]:
        return [s.timestamp_ms for s in self.samples]

    @property
    def clock_values(self) -> List[int]:
        return [s.clock_mhz for s in self.samples]

    @property
    def memory_clock_values(self) -> List[int]:
        return [s.memory_clock_mhz for s in self.samples]

    @property
    def power_values(self) -> List[float]:
        return [s.power_watts for s in self.samples]

    @property
    def temperature_values(self) -> List[int]:
        return [s.temperature_c for s in self.samples]

    @property
    def utilization_values(self) -> List[int]:
        return [s.utilization_pct for s in self.samples]

    @property
    def memory_utilization_values(self) -> List[int]:
        return [s.memory_utilization_pct for s in self.samples]

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        import statistics

        stats: Dict[str, Dict[str, float]] = {}
        for name, values in [
            ("clock_mhz", [float(v) for v in self.clock_values]),
            ("memory_clock_mhz", [float(v) for v in self.memory_clock_values]),
            ("power_watts", self.power_values),
            ("temperature_c", [float(v) for v in self.temperature_values]),
            ("utilization_pct", [float(v) for v in self.utilization_values]),
            (
                "memory_utilization_pct",
                [float(v) for v in self.memory_utilization_values],
            ),
        ]:
            if values:
                stats[name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
        return stats


class GPUTelemetryObserver:
    """Observer that periodically samples GPU telemetry via a collector thread.

    Lazily selects the appropriate collector: AMDGPUCollectorThread for AMD
    (is_hip()), GPUCollectorThread for NVIDIA.
    """

    def __init__(
        self,
        gpu_id: int = 0,
        sample_interval_ms: float = DEFAULT_TELEMETRY_INTERVAL_MS,
        on_sample: Optional[Callable[[GPUSample], None]] = None,
    ) -> None:
        self.gpu_id = gpu_id
        self.sample_interval_ms = sample_interval_ms
        self.on_sample = on_sample

        self._data = GPUTelemetryData(
            gpu_id=gpu_id,
            sample_interval_ms=sample_interval_ms,
        )
        self._collector = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            logger.warning("Observer is already running")
            return

        from tritonbench.utils.env_utils import is_hip

        self._data = GPUTelemetryData(
            gpu_id=self.gpu_id,
            sample_interval_ms=self.sample_interval_ms,
        )
        self._data.start_time = time.perf_counter()
        self._running = True

        query_interval_sec = self.sample_interval_ms / 1000.0

        if is_hip():
            from tritonbench.components.power.amd_gpu_collector import (
                AMDGPUCollectorThread,
            )

            self._collector = AMDGPUCollectorThread(
                gpu_id=self.gpu_id,
                query_interval=query_interval_sec,
            )
        else:
            from tritonbench.components.power.power_manager import GPUCollectorThread

            self._collector = GPUCollectorThread(
                gpu_id=self.gpu_id,
                query_interval=query_interval_sec,
            )

        self._thread = threading.Thread(target=self._collector.start, daemon=True)
        self._thread.start()

        logger.info(
            "GPU telemetry observer started (GPU %d, interval %.1f ms, backend=%s)",
            self.gpu_id,
            self.sample_interval_ms,
            "AMD" if is_hip() else "NVIDIA",
        )

    def stop(self) -> GPUTelemetryData:
        if not self._running:
            logger.warning("Observer is not running")
            return self._data

        if self._collector is not None:
            self._collector.continue_monitoring = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self._data.end_time = time.perf_counter()
        self._running = False

        if self._collector is not None:
            start_timestamp_us = None
            for event in self._collector.events:
                if start_timestamp_us is None:
                    start_timestamp_us = event.timestamp
                elapsed_ms = (event.timestamp - start_timestamp_us) / 1000.0

                sample = GPUSample(
                    timestamp_ms=elapsed_ms,
                    clock_mhz=int(event.sm_clock),
                    memory_clock_mhz=int(event.mem_clock),
                    power_watts=event.power_draw_instant,
                    temperature_c=int(event.gpu_temp),
                    memory_used_mb=event.memory_used_mb,
                    utilization_pct=event.gpu_utilization_pct,
                    memory_utilization_pct=event.memory_utilization_pct,
                )
                self._data.samples.append(sample)

                if self.on_sample is not None:
                    self.on_sample(sample)

        logger.info(
            "GPU telemetry observer stopped: %d samples collected over %.1f ms",
            len(self._data.samples),
            self._data.duration_ms,
        )
        return self._data

    def add_annotation(self, label: str) -> None:
        self._data.add_annotation(label)

    def get_data(self) -> GPUTelemetryData:
        return self._data


def save_telemetry_csv(data: GPUTelemetryData, output_path: str) -> None:
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_ms",
                "clock_mhz",
                "memory_clock_mhz",
                "power_watts",
                "temperature_c",
                "memory_used_mb",
                "utilization_pct",
                "memory_utilization_pct",
            ]
        )
        for sample in data.samples:
            writer.writerow(
                [
                    f"{sample.timestamp_ms:.2f}",
                    sample.clock_mhz,
                    sample.memory_clock_mhz,
                    f"{sample.power_watts:.2f}",
                    sample.temperature_c,
                    sample.memory_used_mb,
                    sample.utilization_pct,
                    sample.memory_utilization_pct,
                ]
            )

    if data.annotations:
        annotations_path = output_path.replace(".csv", "_annotations.csv")
        with open(annotations_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms", "label"])
            for ts, label in data.annotations:
                writer.writerow([f"{ts:.2f}", label])

    logger.info(
        "Telemetry data saved to %s (%d samples)", output_path, len(data.samples)
    )


def plot_telemetry(
    data: GPUTelemetryData,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    show_annotations: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
        return

    if not data.samples:
        logger.warning("No samples to plot")
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    timestamps = data.timestamps

    # Panel 1: Clock frequency (SM + Memory)
    ax1 = axes[0]
    ax1.plot(timestamps, data.clock_values, "b-", linewidth=1.5, label="SM Clock")
    ax1.set_ylabel("SM Clock (MHz)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        timestamps,
        data.memory_clock_values,
        "cyan",
        linewidth=1.5,
        linestyle="--",
        label="Memory Clock",
    )
    ax1_twin.set_ylabel("Memory Clock (MHz)", color="cyan")
    ax1_twin.tick_params(axis="y", labelcolor="cyan")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1t, labels1t = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines1t, labels1 + labels1t, loc="upper right")

    # Panel 2: Power
    ax2 = axes[1]
    ax2.plot(timestamps, data.power_values, "r-", linewidth=1.5, label="Power")
    ax2.set_ylabel("Power (W)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Panel 3: GPU Utilization + Memory Utilization
    ax3 = axes[2]
    ax3.plot(
        timestamps,
        data.utilization_values,
        "g-",
        linewidth=1.5,
        label="GPU Utilization",
    )
    ax3.set_ylabel("GPU Utilization (%)", color="green")
    ax3.tick_params(axis="y", labelcolor="green")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        timestamps,
        data.memory_utilization_values,
        "purple",
        linewidth=1.5,
        linestyle="--",
        label="Memory Utilization",
    )
    ax3_twin.set_ylabel("Memory Utilization (%)", color="purple")
    ax3_twin.tick_params(axis="y", labelcolor="purple")
    ax3_twin.set_ylim(0, 100)

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3t, labels3t = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines3t, labels3 + labels3t, loc="upper right")

    # Panel 4: Temperature
    ax4 = axes[3]
    ax4.plot(
        timestamps,
        data.temperature_values,
        "orange",
        linewidth=1.5,
        label="Temperature",
    )
    ax4.set_ylabel("Temperature (°C)", color="orange")
    ax4.tick_params(axis="y", labelcolor="orange")
    ax4.set_xlabel("Time (ms)")
    ax4.grid(True, alpha=0.3)

    # Annotation markers
    if show_annotations and data.annotations:
        for ax in axes:
            for ts, label in data.annotations:
                ax.axvline(x=ts, color="purple", linestyle=":", alpha=0.7)
                ax.text(
                    ts,
                    ax.get_ylim()[1] * 0.95,
                    label,
                    rotation=90,
                    fontsize=8,
                    color="purple",
                    va="top",
                )

    if title:
        fig.suptitle(title, fontsize=14)
    else:
        fig.suptitle(
            f"GPU {data.gpu_id} Telemetry ({len(data.samples)} samples, {data.duration_ms:.0f} ms)",
            fontsize=14,
        )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", output_path)
    else:
        plt.show()

    plt.close()


class TelemetryContext:
    """Context manager for easy telemetry collection.

    Usage:
        with TelemetryContext(gpu_id=0) as ctx:
            ctx.annotate("warmup_start")
            warmup()
            ctx.annotate("benchmark_end")

        ctx.save_csv("telemetry.csv")
        ctx.plot("telemetry.png")
    """

    def __init__(
        self,
        gpu_id: int = 0,
        sample_interval_ms: float = DEFAULT_TELEMETRY_INTERVAL_MS,
    ) -> None:
        self.observer = GPUTelemetryObserver(
            gpu_id=gpu_id,
            sample_interval_ms=sample_interval_ms,
        )
        self.data: Optional[GPUTelemetryData] = None

    def __enter__(self) -> "TelemetryContext":
        self.observer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.data = self.observer.stop()

    def annotate(self, label: str) -> None:
        self.observer.add_annotation(label)

    def plot(
        self, output_path: Optional[str] = None, title: Optional[str] = None
    ) -> None:
        if self.data is not None:
            plot_telemetry(self.data, output_path=output_path, title=title)

    def save_csv(self, output_path: str) -> None:
        if self.data is not None:
            save_telemetry_csv(self.data, output_path)
