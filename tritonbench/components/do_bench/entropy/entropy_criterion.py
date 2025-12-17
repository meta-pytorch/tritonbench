import math
from collections import deque
from typing import Dict

from .online_linear_regression import OnlineLinearRegression


class EntropyCriterion:
    """
    Entropy-based stopping criterion for adaptive benchmarking.

    This criterion tracks the Shannon entropy of the measurement distribution
    and stops when the entropy shows a stable or decreasing trend, indicating
    that the distribution has converged.

    Uses incremental algorithms for both entropy calculation and linear regression.

    Args:
        Tuning Parameters

        `max_angle` (default: 0.048°)
        - Controls how flat the entropy slope must be for convergence
        - Lower values = stricter convergence requirement
        - Higher values = faster convergence, potentially less stable

        `min_r2` (default: 0.36)
        - Quality of linear fit required for convergence
        - Higher values = require better fit, more samples
        - Lower values = accept noisier data, fewer samples

        `window_size` (default: 299)
        - Number of recent entropy values to consider
        - Larger window = smoother but slower to detect convergence
        - Smaller window = faster response but more sensitive to noise

        `entropy_window_size` (default: 500)
        - Controls how many recent measurements affect entropy
        - Larger window = more stable entropy, slower response to changes
        - Smaller window = faster response but more sensitive to noise

    """

    def __init__(
        self,
        max_angle: float = 0.048,
        min_r2: float = 0.36,
        window_size: int = 299,
        min_warmup_samples: int = 20,
        entropy_window_size: int = 500,
    ) -> None:
        self.max_angle = max_angle
        self.min_r2 = min_r2
        self.window_size = window_size
        self.min_warmup_samples = min_warmup_samples
        self.entropy_window_size = entropy_window_size

        # State tracking
        self.total_samples = 0
        self.total_time = 0.0

        # Incremental Formula: H = log2(n) - S/n where S = Σ(count * log2(count))
        self.measurement_window: deque[float] = deque(maxlen=entropy_window_size)
        self.freq_tracker: Dict[float, int] = {}
        self._sum_count_log_count = 0.0  # S = Σ(count * log2(count))

        # Entropy tracking with online linear regression
        self.entropy_tracker: deque[float] = deque(maxlen=window_size)
        self._regression = OnlineLinearRegression(window_size=window_size)

    def reset(self) -> None:
        """Reset the criterion state."""
        self.total_samples = 0
        self.total_time = 0.0
        self.measurement_window.clear()
        self.freq_tracker.clear()
        self._sum_count_log_count = 0.0
        self.entropy_tracker.clear()
        self._regression.reset()

    def _update_entropy_sum(self, old_count: int, new_count: int) -> None:
        """
        Update the entropy sum S = Σ(count * log2(count)) in O(1).

        Args:
            old_count: Previous count (0 if new unique value)
            new_count: New count (0 if removing unique value)
        """
        # Remove old contribution: S -= old_count * log2(old_count)
        # Optimization: nlog(n) - olog(o) = nlog(1+(n-o)/o) + (n - o)log(o)
        if old_count > 0 and new_count > 0:
            delta = new_count - old_count
            self._sum_count_log_count += new_count * math.log2(
                1 + delta / old_count
            ) + delta * math.log2(old_count)
        else:
            if old_count > 0:
                self._sum_count_log_count -= old_count * math.log2(old_count)
            if new_count > 0:
                self._sum_count_log_count += new_count * math.log2(new_count)

    def _compute_entropy(self) -> float:
        """
        Compute Shannon entropy using the maintained sum in O(1).

        Formula: H = log2(n) - S/n where S = Σ(count * log2(count))

        Returns:
            Shannon entropy in bits.
        """
        n = len(self.measurement_window)
        if n == 0:
            return 0.0

        # Entropy formula: H = log2(n) - S/n
        entropy = math.log2(n) - (self._sum_count_log_count / n)
        return max(0.0, entropy)

    def add_measurement(self, measurement: float) -> None:
        """
        Add a new measurement and update entropy tracking with O(1) complexity.

        Args:
            measurement: Time measurement in milliseconds.
        """
        self.total_samples += 1
        self.total_time += measurement

        # Sliding window entropy update
        # Remove oldest value if window is full
        if len(self.measurement_window) == self.entropy_window_size:
            old_value = self.measurement_window[0]
            old_count = self.freq_tracker[old_value]

            # Update entropy sum: count decreases by 1
            self._update_entropy_sum(old_count, old_count - 1)
            self.freq_tracker[old_value] -= 1
            if self.freq_tracker[old_value] == 0:
                del self.freq_tracker[old_value]

        # Add new value
        old_count = self.freq_tracker.get(measurement, 0)

        # Update entropy sum: count increases by 1
        self._update_entropy_sum(old_count, old_count + 1)
        self.freq_tracker[measurement] = old_count + 1
        self.measurement_window.append(measurement)

        entropy = self._compute_entropy()

        # Update entropy tracker and regression
        self.entropy_tracker.append(entropy)
        self._regression.add_value(entropy)

    def is_finished(self) -> bool:
        """
        Check if the benchmark should stop based on entropy convergence.

        Returns:
            True if convergence criteria are met, False otherwise.
        """
        # Require minimum warmup samples to avoid premature convergence
        # This ensures GPU has time to reach optimal thermal/frequency state
        if self.total_samples < self.min_warmup_samples:
            return False

        # Need at least 2 entropy samples for regression
        if len(self._regression) < 2:
            return False

        # Only check on even samples to reduce overhead
        if self.total_samples % 2 != 0:
            return False

        # Get regression stats
        stats = self._regression.get_stats()
        slope_degrees = self._regression.get_slope_degrees()

        self._last_convergence_check = {
            "slope": stats.slope,
            "slope_degrees": slope_degrees,
            "r2": stats.r2,
            "mean_entropy": self._regression.mean_y(),
            "window_samples": stats.n,
        }

        # Check convergence criteria
        if not math.isfinite(slope_degrees) or not math.isfinite(stats.r2):
            return False

        if slope_degrees > self.max_angle:
            return False

        if stats.r2 < self.min_r2:
            return False

        return True

    def get_convergence_info(self) -> dict[str, float]:
        """Get the last convergence check information."""
        return getattr(self, "_last_convergence_check", {})

    def get_regression_stats(self) -> dict[str, float]:
        stats = self._regression.get_stats()
        return {
            "slope": stats.slope,
            "intercept": stats.intercept,
            "r2": stats.r2,
            "n": stats.n,
        }

    def get_stats(self) -> dict[str, float]:
        return {
            "total_samples": float(self.total_samples),
            "total_time_ms": self.total_time,
            "avg_time_ms": float(self.total_time / self.total_samples)
            if self.total_samples > 0
            else 0.0,
            "current_entropy": self.entropy_tracker[-1]
            if self.entropy_tracker
            else 0.0,
            "entropy_samples": float(len(self.entropy_tracker)),
            "unique_measurements": float(len(self.freq_tracker)),
            "entropy_window_size": float(self.entropy_window_size),
            "measurement_window_utilization": (
                len(self.measurement_window) / self.entropy_window_size
            ),
        }
