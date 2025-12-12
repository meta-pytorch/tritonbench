import random
import unittest
from typing import List

from tritonbench.components.do_bench.entropy.entropy_criterion import EntropyCriterion

from .entropy_test_util import BaselineEntropyCriterion
from .linear_regression_test_util import compute_regression_naive


def generate_data_stream(count: int = 100, seed: int = 42) -> List[float]:
    """Generate a stream of simulated measurement data with Gaussian noise."""
    random.seed(seed)
    return [10.0 + random.gauss(0, 0.5) for _ in range(count)]


def generate_constant_data(count: int = 100, value: float = 10.0) -> List[float]:
    """Generate a constant data stream (all same value)."""
    return [value] * count


class TestEntropyComparison(unittest.TestCase):
    """
    Test that baseline and optimized EntropyCriterion implementations
    produce matching entropy, slope, and R² at every step.

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

    def test_entropy_and_regression_match(self) -> None:
        """Stream 600 samples, verify entropy/slope/R² match at each step."""
        baseline = BaselineEntropyCriterion()
        optimized = EntropyCriterion()

        data = generate_data_stream(count=600)
        mismatches: List[str] = []

        for step, measurement in enumerate(data, 1):
            baseline.add_measurement(measurement)
            optimized.add_measurement(measurement)

            # Compare entropy
            b_entropy = (
                baseline.entropy_values[-1] if baseline.entropy_values else 0.0
            )
            o_entropy = (
                optimized.entropy_tracker[-1] if optimized.entropy_tracker else 0.0
            )
            if not self._values_match(b_entropy, o_entropy):
                mismatches.append(
                    f"Step {step}: entropy mismatch - "
                    f"baseline={b_entropy}, optimized={o_entropy}"
                )

            # Compare slope and R² (need at least 2 samples)
            if len(baseline.entropy_values) >= 2:
                # Baseline: compute from scratch using naive function
                naive_stats = compute_regression_naive(list(baseline.entropy_values))
                b_slope = naive_stats.slope
                b_r2 = naive_stats.r2

                # Optimized: use the get_regression_stats() method
                opt_stats = optimized.get_regression_stats()
                o_slope = opt_stats["slope"]
                o_r2 = opt_stats["r2"]

                if not self._values_match(b_slope, o_slope):
                    mismatches.append(
                        f"Step {step}: slope mismatch - "
                        f"baseline={b_slope}, optimized={o_slope}"
                    )

                if not self._values_match(b_r2, o_r2):
                    mismatches.append(
                        f"Step {step}: R² mismatch - "
                        f"baseline={b_r2}, optimized={o_r2}"
                    )

        if mismatches:
            self.fail(
                f"Found {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20])
            )

    def test_constant_data(self) -> None:
        """Test with constant data (all same value) - edge case for entropy."""
        baseline = BaselineEntropyCriterion()
        optimized = EntropyCriterion()

        data = generate_constant_data(count=300, value=10.0)
        mismatches: List[str] = []

        for step, measurement in enumerate(data, 1):
            baseline.add_measurement(measurement)
            optimized.add_measurement(measurement)

            # Compare entropy
            # For constant data, entropy should be 0 (only one unique value)
            b_entropy = (
                baseline.entropy_values[-1] if baseline.entropy_values else 0.0
            )
            o_entropy = (
                optimized.entropy_tracker[-1] if optimized.entropy_tracker else 0.0
            )
            if not self._values_match(b_entropy, o_entropy):
                mismatches.append(
                    f"Step {step}: entropy mismatch - "
                    f"baseline={b_entropy}, optimized={o_entropy}"
                )

            # Compare slope and R² (need at least 2 samples)
            if len(baseline.entropy_values) >= 2:
                naive_stats = compute_regression_naive(list(baseline.entropy_values))
                b_slope = naive_stats.slope
                b_r2 = naive_stats.r2

                opt_stats = optimized.get_regression_stats()
                o_slope = opt_stats["slope"]
                o_r2 = opt_stats["r2"]

                if not self._values_match(b_slope, o_slope):
                    mismatches.append(
                        f"Step {step}: slope mismatch - "
                        f"baseline={b_slope}, optimized={o_slope}"
                    )

                if not self._values_match(b_r2, o_r2):
                    mismatches.append(
                        f"Step {step}: R² mismatch - "
                        f"baseline={b_r2}, optimized={o_r2}"
                    )

        if mismatches:
            self.fail(
                f"Found {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:20])
            )


if __name__ == "__main__":
    unittest.main()
