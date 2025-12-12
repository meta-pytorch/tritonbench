"""
Test utilities for linear regression verification.

Provides naive O(n) implementations as ground truth for testing
the optimized O(1) OnlineLinearRegression implementation.
"""

from typing import NamedTuple


class RegressionStats(NamedTuple):
    """Statistics from linear regression computation."""

    slope: float
    intercept: float
    r2: float
    n: int


def compute_regression_naive(y_values: list[float]) -> RegressionStats:
    """
    Compute linear regression using the naive O(n) approach.

    This is the ground truth implementation for testing.
    X values are implicitly [0, 1, 2, ..., n-1].

    Args:
        y_values: List of y values.

    Returns:
        RegressionStats with slope, intercept, r2, and sample count.
    """
    n = len(y_values)
    if n < 2:
        return RegressionStats(slope=0.0, intercept=0.0, r2=0.0, n=n)

    # x values are positions: 0, 1, 2, ..., n-1
    x_values = list(range(n))

    # Compute means
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    # Compute sums for linear regression
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    sum_y2 = sum(y * y for y in y_values)

    # Compute slope
    numerator = sum_xy - n * mean_x * mean_y
    denominator = sum_x2 - n * mean_x * mean_x

    # This is an edge case when x is constant, not expected in practice
    if abs(denominator) < 1e-12:
        return RegressionStats(slope=float("nan"), intercept=float("nan"), r2=float("nan"), n=n)

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    # Compute R²
    ss_tot = sum_y2 - n * mean_y * mean_y

    if abs(ss_tot) < 1e-12:
        return RegressionStats(slope=slope, intercept=intercept, r2=1.0, n=n)

    ss_res = sum(
        (y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values)
    )
    r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))

    return RegressionStats(slope=slope, intercept=intercept, r2=r2, n=n)
