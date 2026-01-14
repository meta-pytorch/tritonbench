import math
import random
import unittest
from typing import Dict, List, NamedTuple

from tritonbench.components.do_bench.entropy.entropy_criterion import EntropyCriterion


class RegressionStats(NamedTuple):
    slope: float
    intercept: float
    r2: float
    n: int


def compute_regression_naive(y_values: list[float]) -> RegressionStats:
    """Naive O(n) linear regression. X values are [0, 1, 2, ..., n-1]."""
    n = len(y_values)
    if n < 2:
        return RegressionStats(slope=0.0, intercept=0.0, r2=0.0, n=n)

    x_values = list(range(n))
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    sum_y2 = sum(y * y for y in y_values)

    numerator = sum_xy - n * mean_x * mean_y
    denominator = sum_x2 - n * mean_x * mean_x

    if abs(denominator) < 1e-12:
        return RegressionStats(
            slope=float("nan"), intercept=float("nan"), r2=float("nan"), n=n
        )

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    ss_tot = sum_y2 - n * mean_y * mean_y
    if abs(ss_tot) < 1e-12:
        return RegressionStats(slope=slope, intercept=intercept, r2=1.0, n=n)

    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
    r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))

    return RegressionStats(slope=slope, intercept=intercept, r2=r2, n=n)


def compute_entropy_naive(measurements: List[float]) -> float:
    """Naive O(n) Shannon entropy."""
    if len(measurements) == 0:
        return 0.0

    freq: Dict[float, int] = {}
    for m in measurements:
        freq[m] = freq.get(m, 0) + 1

    n = len(measurements)
    entropy = 0.0

    for count in freq.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)

    return max(0.0, entropy)


class BaselineEntropyCriterion:
    """Naive O(n) entropy criterion for testing."""

    def __init__(
        self,
        window_size: int = 299,
        entropy_window_size: int = 500,
    ) -> None:
        self.window_size = window_size
        self.entropy_window_size = entropy_window_size
        self.total_samples: int = 0
        self.measurements: List[float] = []
        self.entropy_values: List[float] = []

    def add_measurement(self, measurement: float) -> None:
        self.total_samples += 1

        self.measurements.append(measurement)
        if len(self.measurements) > self.entropy_window_size:
            self.measurements = self.measurements[-self.entropy_window_size :]

        entropy = compute_entropy_naive(self.measurements)

        self.entropy_values.append(entropy)
        if len(self.entropy_values) > self.window_size:
            self.entropy_values = self.entropy_values[-self.window_size :]


def generate_data_stream(count: int = 100, seed: int = 42) -> List[float]:
    random.seed(seed)
    return [10.0 + random.gauss(0, 0.5) for _ in range(count)]


def generate_constant_data(count: int = 100, value: float = 10.0) -> List[float]:
    return [value] * count


class TestEntropyComparison(unittest.TestCase):
    """Verify EntropyCriterion matches naive implementation."""

    def setUp(self) -> None:
        self.tolerance = 1e-9

    def _values_match(self, a: float, b: float) -> bool:
        if abs(a) < 1e-15 and abs(b) < 1e-15:
            return True
        abs_diff = abs(a - b)
        if abs(a) > 1e-15:
            rel_diff = abs_diff / abs(a)
            return rel_diff <= self.tolerance or abs_diff <= self.tolerance
        return abs_diff <= self.tolerance

    def test_entropy_and_regression_match(self) -> None:
        baseline = BaselineEntropyCriterion()
        optimized = EntropyCriterion()

        data = generate_data_stream(count=600)
        mismatches: List[str] = []

        for step, measurement in enumerate(data, 1):
            baseline.add_measurement(measurement)
            optimized.add_measurement(measurement)

            b_entropy = baseline.entropy_values[-1] if baseline.entropy_values else 0.0
            o_entropy = (
                optimized.entropy_tracker[-1] if optimized.entropy_tracker else 0.0
            )
            if not self._values_match(b_entropy, o_entropy):
                mismatches.append(
                    f"Step {step}: entropy mismatch - "
                    f"baseline={b_entropy}, optimized={o_entropy}"
                )

            if len(baseline.entropy_values) >= 2:
                naive_stats = compute_regression_naive(list(baseline.entropy_values))
                opt_stats = optimized.get_regression_stats()

                if not self._values_match(naive_stats.slope, opt_stats["slope"]):
                    mismatches.append(
                        f"Step {step}: slope mismatch - "
                        f"baseline={naive_stats.slope}, optimized={opt_stats['slope']}"
                    )

                if not self._values_match(naive_stats.r2, opt_stats["r2"]):
                    mismatches.append(
                        f"Step {step}: R² mismatch - "
                        f"baseline={naive_stats.r2}, optimized={opt_stats['r2']}"
                    )

        if mismatches:
            self.fail(
                f"Found {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20])
            )

    def test_constant_data(self) -> None:
        baseline = BaselineEntropyCriterion()
        optimized = EntropyCriterion()

        data = generate_constant_data(count=300, value=10.0)
        mismatches: List[str] = []

        for step, measurement in enumerate(data, 1):
            baseline.add_measurement(measurement)
            optimized.add_measurement(measurement)

            b_entropy = baseline.entropy_values[-1] if baseline.entropy_values else 0.0
            o_entropy = (
                optimized.entropy_tracker[-1] if optimized.entropy_tracker else 0.0
            )
            if not self._values_match(b_entropy, o_entropy):
                mismatches.append(
                    f"Step {step}: entropy mismatch - "
                    f"baseline={b_entropy}, optimized={o_entropy}"
                )

            if len(baseline.entropy_values) >= 2:
                naive_stats = compute_regression_naive(list(baseline.entropy_values))
                opt_stats = optimized.get_regression_stats()

                if not self._values_match(naive_stats.slope, opt_stats["slope"]):
                    mismatches.append(
                        f"Step {step}: slope mismatch - "
                        f"baseline={naive_stats.slope}, optimized={opt_stats['slope']}"
                    )

                if not self._values_match(naive_stats.r2, opt_stats["r2"]):
                    mismatches.append(
                        f"Step {step}: R² mismatch - "
                        f"baseline={naive_stats.r2}, optimized={opt_stats['r2']}"
                    )

        if mismatches:
            self.fail(
                f"Found {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20])
            )
