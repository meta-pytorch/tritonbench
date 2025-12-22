import random
import unittest
from typing import List, NamedTuple

from tritonbench.components.do_bench.entropy.online_linear_regression import (
    OnlineLinearRegression,
)


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

    ss_res = sum(
        (y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values)
    )
    r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))

    return RegressionStats(slope=slope, intercept=intercept, r2=r2, n=n)


def generate_data_stream(count: int = 100, seed: int = 42) -> List[float]:
    random.seed(seed)
    return [10.0 + random.gauss(0, 0.5) for _ in range(count)]


def generate_constant_data(count: int = 100, value: float = 10.0) -> List[float]:
    return [value] * count


class TestOnlineLinearRegression(unittest.TestCase):
    """Verify OnlineLinearRegression matches naive O(n) implementation."""

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

    def test_regression_matches_at_each_step(self) -> None:
        window_size = 299
        online = OnlineLinearRegression(window_size=window_size)

        data = generate_data_stream(count=600)
        values: List[float] = []
        mismatches: List[str] = []

        for step, value in enumerate(data, 1):
            online.add_value(value)
            values.append(value)

            if len(values) > window_size:
                values = values[-window_size:]

            if len(values) < 2:
                continue

            naive_stats = compute_regression_naive(values)
            online_stats = online.get_stats()

            if not self._values_match(naive_stats.slope, online_stats.slope):
                mismatches.append(
                    f"Step {step}: slope mismatch - "
                    f"naive={naive_stats.slope}, online={online_stats.slope}"
                )

            if not self._values_match(naive_stats.r2, online_stats.r2):
                mismatches.append(
                    f"Step {step}: R² mismatch - "
                    f"naive={naive_stats.r2}, online={online_stats.r2}"
                )

        if mismatches:
            self.fail(
                f"Found {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20])
            )

    def test_constant_data(self) -> None:
        window_size = 299
        online = OnlineLinearRegression(window_size=window_size)

        data = generate_constant_data(count=300, value=10.0)
        values: List[float] = []
        mismatches: List[str] = []

        for step, value in enumerate(data, 1):
            online.add_value(value)
            values.append(value)

            if len(values) > window_size:
                values = values[-window_size:]

            if len(values) < 2:
                continue

            naive_stats = compute_regression_naive(values)
            online_stats = online.get_stats()

            if not self._values_match(naive_stats.slope, online_stats.slope):
                mismatches.append(
                    f"Step {step}: slope mismatch - "
                    f"naive={naive_stats.slope}, online={online_stats.slope}"
                )

            if not self._values_match(naive_stats.r2, online_stats.r2):
                mismatches.append(
                    f"Step {step}: R² mismatch - "
                    f"naive={naive_stats.r2}, online={online_stats.r2}"
                )

        if mismatches:
            self.fail(
                f"Found {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20])
            )
