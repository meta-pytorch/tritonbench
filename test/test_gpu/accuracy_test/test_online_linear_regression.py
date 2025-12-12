import random
import unittest
from typing import List

from tritonbench.components.do_bench.entropy.online_linear_regression import (
    OnlineLinearRegression,
)

from .linear_regression_test_util import compute_regression_naive


def generate_data_stream(count: int = 100, seed: int = 42) -> List[float]:
    """Generate a stream of simulated measurement data with Gaussian noise."""
    random.seed(seed)
    return [10.0 + random.gauss(0, 0.5) for _ in range(count)]


def generate_constant_data(count: int = 100, value: float = 10.0) -> List[float]:
    """Generate a constant data stream (all same value)."""
    return [value] * count


class TestOnlineLinearRegression(unittest.TestCase):
    """
    Test that OnlineLinearRegression matches naive O(n) implementation
    at every step of a data stream.

    Tolerance is set to 1e-9 as the numerical stable formulation could
    slightly deviate from the naive implementation.
    """

    def setUp(self) -> None:
        self.tolerance = 1e-9

    def _values_match(self, a: float, b: float) -> bool:
        """Check if two values match within tolerance."""
        if abs(a) < 1e-15 and abs(b) < 1e-15:
            return True
        abs_diff = abs(a - b)
        if abs(a) > 1e-15:
            rel_diff = abs_diff / abs(a)
            return rel_diff <= self.tolerance or abs_diff <= self.tolerance
        return abs_diff <= self.tolerance

    def test_regression_matches_at_each_step(self) -> None:
        """Stream 600 samples, verify slope/R² match at each step."""
        window_size = 299
        online = OnlineLinearRegression(window_size=window_size)

        data = generate_data_stream(count=600)
        values: List[float] = []
        mismatches: List[str] = []

        for step, value in enumerate(data, 1):
            online.add_value(value)
            values.append(value)

            # Keep only the window
            if len(values) > window_size:
                values = values[-window_size:]

            if len(values) < 2:
                continue

            # Naive computation from scratch
            naive_stats = compute_regression_naive(values)

            # Online computation
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
        """Test with constant data (all same value) - edge case for regression."""
        window_size = 299
        online = OnlineLinearRegression(window_size=window_size)

        data = generate_constant_data(count=300, value=10.0)
        values: List[float] = []
        mismatches: List[str] = []

        for step, value in enumerate(data, 1):
            online.add_value(value)
            values.append(value)

            # Keep only the window
            if len(values) > window_size:
                values = values[-window_size:]

            if len(values) < 2:
                continue

            # Naive computation from scratch
            naive_stats = compute_regression_naive(values)

            # Online computation
            online_stats = online.get_stats()

            # For constant data:
            # - slope should be 0 (or near 0)
            # - R² should be 1.0 (perfect horizontal fit) or handled specially
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


if __name__ == "__main__":
    unittest.main()
