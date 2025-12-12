"""
Baseline utility functions for entropy criterion calculations.

These are naive, unoptimized implementations that serve as ground truth
for verifying the optimized implementations in production.

Each function is implemented using the most straightforward algorithm
with O(n) complexity per call, prioritizing clarity over performance.
"""

import math
from typing import Dict, List, Tuple


def compute_entropy_naive(measurements: List[float]) -> float:
    """
    Compute Shannon entropy using the textbook formula.

    This is the ground truth implementation - O(n) per call,
    recalculates frequency distribution from scratch.

    Formula: H = -Σ(p_i * log2(p_i)) where p_i = count_i / n

    Args:
        measurements: List of measurement values

    Returns:
        Shannon entropy in bits
    """
    if len(measurements) == 0:
        return 0.0

    # Build frequency distribution from scratch
    freq: Dict[float, int] = {}
    for m in measurements:
        freq[m] = freq.get(m, 0) + 1

    n = len(measurements)
    entropy = 0.0

    # Standard Shannon entropy: H = -Σ(p_i * log2(p_i))
    for count in freq.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)

    # Protect against negative entropy due to floating-point errors
    return max(0.0, entropy)


def compute_linear_regression_naive(
    y_values: List[float],
) -> Tuple[float, float, float]:
    """
    Compute linear regression (slope, intercept, R²) from scratch.

    This is the ground truth implementation using standard least squares.
    X values are implicitly [0, 1, 2, ..., n-1].

    Args:
        y_values: List of y values (entropy values)

    Returns:
        Tuple of (slope, intercept, r2)
    """
    n = len(y_values)
    if n < 2:
        return 0.0, 0.0, 0.0

    # x values are positions: 0, 1, 2, ..., n-1
    x_values = list(range(n))

    # Compute means
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    # Compute sums for linear regression
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    sum_y2 = sum(y * y for y in y_values)

    # Compute slope: β = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
    numerator = sum_xy - n * mean_x * mean_y
    denominator = sum_x2 - n * mean_x * mean_x

    if abs(denominator) < 1e-12:
        return 0.0, mean_y, 0.0

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    # Compute R²
    ss_tot = sum_y2 - n * mean_y * mean_y

    if abs(ss_tot) < 1e-12:
        # All y values are identical -> perfect horizontal line
        return slope, intercept, 1.0

    # SS_res = Σ(y - ŷ)² where ŷ = slope * x + intercept
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))

    r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))

    return slope, intercept, r2


def check_convergence_naive(
    entropy_values: List[float],
    max_angle: float = 0.048,
    min_r2: float = 0.36,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if entropy values indicate convergence using naive calculations.

    This is the ground truth implementation for convergence checking.

    Args:
        entropy_values: List of entropy values
        max_angle: Maximum slope angle in degrees for convergence
        min_r2: Minimum R² value for convergence

    Returns:
        Tuple of (converged, stats_dict)
    """
    if len(entropy_values) < 2:
        return False, {"slope": 0.0, "slope_degrees": 0.0, "r2": 0.0}

    slope, intercept, r2 = compute_linear_regression_naive(entropy_values)
    slope_degrees = math.degrees(math.atan(slope))

    stats = {
        "slope": slope,
        "intercept": intercept,
        "slope_degrees": slope_degrees,
        "r2": r2,
        "mean_entropy": sum(entropy_values) / len(entropy_values),
        "num_samples": len(entropy_values),
    }

    # Check convergence criteria
    converged = slope_degrees <= max_angle and r2 >= min_r2

    return converged, stats


class BaselineEntropyCriterion:
    """
    A naive, unoptimized entropy criterion implementation.

    This class serves as the baseline/ground truth for testing
    the optimized EntropyCriterion implementation.

    All calculations are done from scratch (O(n) per operation)
    with no incremental updates or caching.
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
        self.total_samples: int = 0
        self.total_time: float = 0.0

        # Simple lists - no incremental updates
        self.measurements: List[float] = []
        self.entropy_values: List[float] = []

    def reset(self) -> None:
        """Reset all state."""
        self.total_samples = 0
        self.total_time = 0.0
        self.measurements = []
        self.entropy_values = []

    def add_measurement(self, measurement: float) -> None:
        """
        Add a measurement and update entropy tracking.

        This is O(n) per call - intentionally naive.
        """
        self.total_samples += 1
        self.total_time += measurement

        # Maintain sliding window for measurements
        self.measurements.append(measurement)
        if len(self.measurements) > self.entropy_window_size:
            self.measurements = self.measurements[-self.entropy_window_size :]

        # Compute entropy from scratch
        entropy = compute_entropy_naive(self.measurements)

        # Maintain sliding window for entropy values
        self.entropy_values.append(entropy)
        if len(self.entropy_values) > self.window_size:
            self.entropy_values = self.entropy_values[-self.window_size :]

    def is_finished(self) -> bool:
        """
        Check if benchmark should stop based on entropy convergence.

        Returns:
            True if convergence criteria are met
        """
        # Require minimum warmup samples
        if self.total_samples < self.min_warmup_samples:
            return False

        # Need at least 2 entropy samples for regression
        if len(self.entropy_values) < 2:
            return False

        # Only check on even samples (matches optimized implementation)
        if self.total_samples % 2 != 0:
            return False

        converged, stats = check_convergence_naive(
            self.entropy_values,
            max_angle=self.max_angle,
            min_r2=self.min_r2,
        )

        self._last_convergence_check = stats
        return converged

    def get_convergence_info(self) -> Dict[str, float]:
        """Get the last convergence check information."""
        return getattr(self, "_last_convergence_check", {})

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        unique_measurements = len(set(self.measurements))

        return {
            "total_samples": float(self.total_samples),
            "total_time_ms": self.total_time,
            "avg_time_ms": (
                self.total_time / self.total_samples if self.total_samples > 0 else 0.0
            ),
            "current_entropy": (
                self.entropy_values[-1] if self.entropy_values else 0.0
            ),
            "entropy_samples": float(len(self.entropy_values)),
            "unique_measurements": float(unique_measurements),
            "entropy_window_size": float(self.entropy_window_size),
            "measurement_window_utilization": (
                len(self.measurements) / self.entropy_window_size
            ),
        }


def compare_with_optimized(
    baseline: BaselineEntropyCriterion,
    optimized: object,  # EntropyCriterion from tritonbench.components.do_bench.entropy
    tolerance: float = 1e-9,
) -> Tuple[bool, List[Dict[str, object]]]:
    """
    Compare baseline and optimized implementations.

    Args:
        baseline: BaselineEntropyCriterion instance
        optimized: EntropyCriterion instance (the optimized one from production)
        tolerance: Maximum allowed difference

    Returns:
        Tuple of (all_match, divergences)
    """
    divergences: List[Dict[str, object]] = []

    baseline_stats = baseline.get_stats()
    optimized_stats = optimized.get_stats()

    metrics_to_compare = [
        "total_samples",
        "total_time_ms",
        "avg_time_ms",
        "current_entropy",
        "entropy_samples",
        "unique_measurements",
        "measurement_window_utilization",
    ]

    for metric in metrics_to_compare:
        baseline_val = baseline_stats.get(metric, 0.0)
        optimized_val = optimized_stats.get(metric, 0.0)

        # Handle both being near zero
        if abs(baseline_val) < 1e-15 and abs(optimized_val) < 1e-15:
            continue

        # Calculate relative difference
        if abs(baseline_val) > 1e-15:
            rel_diff = abs(optimized_val - baseline_val) / abs(baseline_val)
        else:
            rel_diff = abs(optimized_val - baseline_val)

        abs_diff = abs(optimized_val - baseline_val)

        if rel_diff > tolerance and abs_diff > tolerance:
            divergences.append(
                {
                    "metric": metric,
                    "baseline": baseline_val,
                    "optimized": optimized_val,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                }
            )

    all_match = len(divergences) == 0
    return all_match, divergences
