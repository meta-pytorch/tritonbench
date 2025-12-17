import math
from collections import deque
from typing import NamedTuple, Tuple


class RegressionStats(NamedTuple):
    slope: float
    intercept: float
    r2: float
    n: int


class OnlineLinearRegression:
    """
    Incremental linear regression with sliding window support.

    Maintains running statistics for computing slope, intercept, and R²
    with O(1) updates.

    Args:
        window_size: Maximum number of samples to keep in the sliding window.
    """

    def __init__(self, window_size: int = 299) -> None:
        self.window_size = window_size
        self._values: deque[float] = deque(maxlen=window_size)

        # Running statistics for linear regression
        self._sum_x: float = 0.0
        self._sum_y: float = 0.0
        self._sum_xy: float = 0.0
        self._sum_x2: float = 0.0
        self._sum_y2: float = 0.0
        self._count: int = 0

    def reset(self) -> None:
        # Reset all state.
        self._values.clear()
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xy = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0
        self._count = 0

    def update(self, incoming: Tuple[float, float]) -> None:
        x, y = incoming
        self._sum_x += x
        self._sum_y += y
        self._sum_xy += x * y
        self._sum_x2 += x * x
        self._sum_y2 += y * y
        self._count += 1

    def update_replace(
        self, outgoing: Tuple[float, float], incoming: Tuple[float, float]
    ) -> None:
        """
        Remove an old point and add a new point.
        """
        x_out, y_out = outgoing
        self._sum_x -= x_out
        self._sum_y -= y_out
        self._sum_xy -= x_out * y_out
        self._sum_x2 -= x_out * x_out
        self._sum_y2 -= y_out * y_out

        x_in, y_in = incoming
        self._sum_x += x_in
        self._sum_y += y_in
        self._sum_xy += x_in * y_in
        self._sum_x2 += x_in * x_in
        self._sum_y2 += y_in * y_in

    def slide_window(self, y_out: float, y_in: float) -> None:
        """
        Optimized sliding window update for implicit x positions.

        This assumes x values are implicitly 0, 1, 2, ..., n-1 and the
        window is already full.
        """
        self._sum_y -= y_out
        self._sum_y += y_in
        self._sum_y2 -= y_out * y_out
        self._sum_y2 += y_in * y_in

        self._sum_xy -= self._sum_y - y_in
        self._sum_xy += (float(self._count) - 1.0) * y_in

    def add_value(self, y: float) -> None:
        """
        High-level API: Add a y value with implicit x positioning.

        This handles both the initial fill and sliding window phases:
        - Before window is full: uses update() with explicit (x, y)
        - After window is full: uses slide_window() for O(1) updates

        Args:
            y: The y value to add.
        """
        if len(self._values) == self.window_size:
            # Window is full - use optimized slide
            y_out = self._values[0]
            self.slide_window(y_out, y)
        else:
            # Window not full - use standard update with position
            x = float(self._count)
            self.update((x, y))

        self._values.append(y)

    @property
    def count(self) -> int:
        return self._count

    def mean_x(self) -> float:
        if self._count > 0:
            return self._sum_x / float(self._count)
        return 0.0

    def mean_y(self) -> float:
        if self._count > 0:
            return self._sum_y / float(self._count)
        return 0.0

    def slope(self) -> float:
        if self._count < 2:
            return float("nan")

        n = float(self._count)
        mean_x = self._sum_x / n
        mean_y = self._sum_y / n

        numerator = (self._sum_xy / n) - mean_x * mean_y
        denominator = (self._sum_x2 / n) - mean_x * mean_x

        if abs(denominator) < 1e-12:
            return float("nan")

        return numerator / denominator

    def intercept(self) -> float:
        if self._count < 2:
            return float("nan")

        current_slope = self.slope()

        if not math.isfinite(current_slope):
            return float("nan")

        return self.mean_y() - current_slope * self.mean_x()

    def r_squared(self) -> float:
        """
        R² value between 0 and 1, or NaN if insufficient data.
        """
        if self._count < 2:
            return float("nan")

        n = float(self._count)
        mean_y_v = self.mean_y()
        ss_tot = (self._sum_y2 / n) - mean_y_v * mean_y_v

        epsilon = 1e-12

        if ss_tot < epsilon:
            return 1.0

        slope_v = self.slope()
        intercept_v = self.intercept()

        if not math.isfinite(slope_v) or not math.isfinite(intercept_v):
            return float("nan")

        mean_xy_v = self._sum_xy / n
        mean_xx_v = self._sum_x2 / n
        mean_x_v = self._sum_x / n

        ss_tot_m_res = (
            slope_v
            * ((mean_xy_v - slope_v * mean_xx_v) + (mean_xy_v - intercept_v * mean_x_v))
            + intercept_v * (mean_y_v - slope_v * mean_x_v - intercept_v)
            + mean_y_v * (intercept_v - mean_y_v)
        )

        return min(max(ss_tot_m_res / ss_tot, 0.0), 1.0)

    def get_stats(self) -> RegressionStats:
        return RegressionStats(
            slope=self.slope(),
            intercept=self.intercept(),
            r2=self.r_squared(),
            n=self._count,
        )

    def get_slope_degrees(self) -> float:
        s = self.slope()
        if not math.isfinite(s):
            return float("nan")
        return math.degrees(math.atan(s))

    @property
    def sum_x(self) -> float:
        return self._sum_x

    @property
    def sum_y(self) -> float:
        return self._sum_y

    @property
    def sum_xy(self) -> float:
        return self._sum_xy

    @property
    def sum_x2(self) -> float:
        return self._sum_x2

    @property
    def sum_y2(self) -> float:
        return self._sum_y2

    @property
    def values(self) -> deque[float]:
        return self._values

    def __len__(self) -> int:
        return self._count
