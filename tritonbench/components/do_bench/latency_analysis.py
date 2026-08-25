"""Statistical analysis of raw latency samples.

``do_bench_wrapper`` returns a :class:`Latency` that keeps every per-iteration
measurement, not just the p50. This module turns those raw samples into

* descriptive statistics (min/max/mean/median/stddev/stderr, CV, IQR),
* confidence intervals for the mean (Student-t and bootstrap percentile), and
* when both sides of an A/B test are available, a hypothesis test with an
  effect size plus a percent change reported with a confidence interval.

The test is chosen from the data: Shapiro-Wilk decides whether both samples look
normal. If they do we use Welch's t-test (unequal variances) and Cohen's d,
otherwise the distribution-free Mann-Whitney U test with a rank-biserial
correlation. The percent change CI always comes from the bootstrap, which makes
no distributional assumption either way.

Everything here is pure stdlib (``math``/``statistics``/``random``) so latency
analysis never adds a scipy/numpy dependency to tritonbench.
"""

import logging
import math
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE = 0.95
# Below this many samples per side the intervals and tests are too unreliable to
# be worth reporting, so the analysis is skipped with a warning.
MIN_SAMPLE = 30
# Significance level for both the normality test and the A/B test.
DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
# Bootstrap cost is O(resamples * n). Very long sample vectors get fewer
# resamples so a full A/B matrix stays fast; 1000 resamples is still plenty for
# a percentile CI.
MIN_BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_WORK_BUDGET = 5_000_000
# Bootstrap and subsampling are seeded so repeated analyses of the same samples
# report identical numbers.
DEFAULT_SEED = 0x5EED
# Shapiro-Wilk is only defined up to a few thousand points, and with very large
# n it rejects normality on differences too small to matter. Beyond this we test
# an evenly spaced subsample.
SHAPIRO_MAX_SAMPLES = 5000
MIN_SAMPLES_FOR_SHAPIRO = 3


# ============================================================================
# Distribution helpers (stdlib only)
# ============================================================================

_NORMAL = statistics.NormalDist()


def _norm_cdf(x: float) -> float:
    return _NORMAL.cdf(x)


def _norm_ppf(p: float) -> float:
    return _NORMAL.inv_cdf(p)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction expansion used by the incomplete beta function."""
    tiny = 1e-300
    eps = 3e-14
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        # Even step of the recurrence.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # Odd step of the recurrence.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_sf_two_sided(t: float, df: float) -> float:
    """Two-sided tail probability P(|T| >= |t|) for a Student-t with df."""
    if df <= 0 or math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    return _betainc(0.5 * df, 0.5, df / (df + t * t))


def _t_cdf(t: float, df: float) -> float:
    tail = 0.5 * _t_sf_two_sided(t, df)
    return 1.0 - tail if t > 0 else tail


def _t_ppf(p: float, df: float) -> float:
    """Inverse Student-t CDF, by bisection on ``_t_cdf`` (monotonic)."""
    if df <= 0:
        return float("nan")
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    lo, hi = -1.0e3, 1.0e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, abs(lo)):
            break
    return 0.5 * (lo + hi)


def _poly(coeffs: Sequence[float], x: float) -> float:
    """Evaluate ``coeffs[0] + coeffs[1]*x + ...`` (Horner)."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


# ============================================================================
# Normality: Shapiro-Wilk (Royston 1995, AS R94)
# ============================================================================

_SW_C1 = (0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056)
_SW_C2 = (0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)
_SW_G = (-2.273, 0.459)
_SW_C3 = (0.5440, -0.39978, 0.025054, -0.0006714)
_SW_C4 = (1.3822, -0.77857, 0.062767, -0.0020322)
_SW_C5 = (-1.5861, -0.31082, -0.083751, 0.0038915)
_SW_C6 = (-0.4803, -0.082676, 0.0030302)


def shapiro_wilk(samples: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Shapiro-Wilk test for normality.

    Returns ``(W, p_value)``, or ``None`` when the test is not applicable
    (fewer than 3 samples, or all samples identical).
    """
    x = sorted(float(v) for v in samples)
    n = len(x)
    if n < MIN_SAMPLES_FOR_SHAPIRO:
        return None

    mean = statistics.fmean(x)
    ss = math.fsum((v - mean) ** 2 for v in x)
    if ss <= 0.0:
        # Zero variance: the sample is a constant, normality is undefined.
        return None

    # Expected values of the standard normal order statistics.
    m = [_norm_ppf((i + 1 - 0.375) / (n + 0.25)) for i in range(n)]
    ssumm2 = math.fsum(v * v for v in m)
    rsn = 1.0 / math.sqrt(n)

    a = [0.0] * n
    if n == 3:
        a[n - 1] = math.sqrt(0.5)
        a[0] = -a[n - 1]
    else:
        c = [v / math.sqrt(ssumm2) for v in m]
        a_last = c[n - 1] + _poly(_SW_C1, rsn)
        a[n - 1] = a_last
        a[0] = -a_last
        if n > 5:
            a_last1 = c[n - 2] + _poly(_SW_C2, rsn)
            a[n - 2] = a_last1
            a[1] = -a_last1
            phi = (ssumm2 - 2.0 * m[n - 1] ** 2 - 2.0 * m[n - 2] ** 2) / (
                1.0 - 2.0 * a_last**2 - 2.0 * a_last1**2
            )
            lo, hi = 2, n - 2
        else:
            phi = (ssumm2 - 2.0 * m[n - 1] ** 2) / (1.0 - 2.0 * a_last**2)
            lo, hi = 1, n - 1
        if phi <= 0.0:
            return None
        sqrt_phi = math.sqrt(phi)
        for i in range(lo, hi):
            a[i] = m[i] / sqrt_phi

    w = math.fsum(ai * xi for ai, xi in zip(a, x)) ** 2 / ss
    w = min(w, 1.0)

    # Royston's normalizing transforms for the null distribution of W.
    if n == 3:
        p = (6.0 / math.pi) * (math.asin(math.sqrt(w)) - math.asin(math.sqrt(0.75)))
    elif w >= 1.0:
        p = 1.0
    elif n <= 11:
        gamma = _poly(_SW_G, n)
        if gamma - math.log1p(-w) <= 0.0:
            return w, 0.0
        y = -math.log(gamma - math.log1p(-w))
        mu = _poly(_SW_C3, n)
        sigma = math.exp(_poly(_SW_C4, n))
        p = 1.0 - _norm_cdf((y - mu) / sigma)
    else:
        u = math.log(n)
        y = math.log1p(-w)
        mu = _poly(_SW_C5, u)
        sigma = math.exp(_poly(_SW_C6, u))
        p = 1.0 - _norm_cdf((y - mu) / sigma)

    return w, min(max(p, 0.0), 1.0)


def _subsample(samples: Sequence[float], max_n: int) -> List[float]:
    """Evenly spaced subsample, preserving order. Deterministic."""
    n = len(samples)
    if n <= max_n:
        return list(samples)
    step = n / max_n
    return [samples[int(i * step)] for i in range(max_n)]


# ============================================================================
# Result types
# ============================================================================


@dataclass
class SampleStats:
    """Descriptive statistics for one set of latency samples (in ms)."""

    n: int
    min: float
    max: float
    mean: float
    median: float
    stddev: float  # sample standard deviation (ddof=1)
    stderr: float  # standard error of the mean
    cv: float  # coefficient of variation, as a percentage
    q1: float
    q3: float
    iqr: float
    confidence: float
    mean_ci: Tuple[float, float]  # Student-t CI for the mean
    bootstrap_mean_ci: Tuple[float, float]  # percentile bootstrap CI for the mean
    bootstrap_median_ci: Tuple[float, float]  # percentile bootstrap CI for the median


@dataclass
class NormalityResult:
    statistic: float
    p_value: float
    normal: bool
    n_tested: int  # may be < n when the sample was subsampled
    n_total: int


@dataclass
class ComparisonResult:
    """Outcome of comparing side B against side A."""

    normality_a: Optional[NormalityResult]
    normality_b: Optional[NormalityResult]
    normal: bool  # both sides passed Shapiro-Wilk
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    dof: Optional[float]  # Welch's degrees of freedom, None for Mann-Whitney
    effect_name: str
    effect_size: float
    effect_magnitude: str
    pct_change: float  # (mean_b - mean_a) / mean_a * 100
    pct_change_ci: Tuple[float, float]  # bootstrap percentile CI
    confidence: float


@dataclass
class LatencyAnalysis:
    side_a: SampleStats
    side_b: Optional[SampleStats]
    comparison: Optional[ComparisonResult]


# ============================================================================
# Descriptive statistics
# ============================================================================


def _percentile_ci(values: List[float], confidence: float) -> Tuple[float, float]:
    """Percentile interval of an already sorted list of bootstrap estimates."""
    if not values:
        return (float("nan"), float("nan"))
    tail = (1.0 - confidence) / 2.0
    lo_idx = max(0, min(len(values) - 1, int(math.floor(tail * len(values)))))
    hi_idx = max(
        0, min(len(values) - 1, int(math.ceil((1.0 - tail) * len(values))) - 1)
    )
    return (values[lo_idx], values[hi_idx])


def _effective_resamples(n: int, requested: int) -> int:
    """Trim the resample count for long sample vectors (see the budget above)."""
    if n <= 0:
        return requested
    return min(requested, max(MIN_BOOTSTRAP_RESAMPLES, BOOTSTRAP_WORK_BUDGET // n))


def _bootstrap_cis(
    samples: Sequence[float],
    confidence: float,
    resamples: int,
    rng: random.Random,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Percentile bootstrap CIs for the mean and the median."""
    n = len(samples)
    resamples = _effective_resamples(n, resamples)
    if n < 2 or resamples < 2:
        value = float(samples[0]) if n else float("nan")
        return (value, value), (value, value)
    data = list(samples)
    means = []
    medians = []
    for _ in range(resamples):
        draw = rng.choices(data, k=n)
        means.append(statistics.fmean(draw))
        medians.append(statistics.median(draw))
    means.sort()
    medians.sort()
    return _percentile_ci(means, confidence), _percentile_ci(medians, confidence)


def compute_sample_stats(
    samples: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Optional[SampleStats]:
    """Descriptive statistics, dispersion and confidence intervals for one side."""
    data = [float(v) for v in samples]
    n = len(data)
    if n == 0:
        return None

    mean = statistics.fmean(data)
    median = statistics.median(data)
    if n >= 2:
        stddev = statistics.stdev(data)
        stderr = stddev / math.sqrt(n)
        # Inclusive quantiles (linear interpolation between order statistics),
        # matching the usual numpy.percentile convention.
        q1, _, q3 = statistics.quantiles(data, n=4, method="inclusive")
        # Student-t CI for the mean.
        t_crit = _t_ppf(0.5 + confidence / 2.0, n - 1)
        half_width = t_crit * stderr
        mean_ci = (mean - half_width, mean + half_width)
    else:
        stddev = 0.0
        stderr = 0.0
        q1 = q3 = data[0]
        mean_ci = (mean, mean)

    rng = random.Random(seed)
    bootstrap_mean_ci, bootstrap_median_ci = _bootstrap_cis(
        data, confidence, bootstrap_resamples, rng
    )

    return SampleStats(
        n=n,
        min=min(data),
        max=max(data),
        mean=mean,
        median=median,
        stddev=stddev,
        stderr=stderr,
        cv=(stddev / mean * 100.0) if mean else float("nan"),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        confidence=confidence,
        mean_ci=mean_ci,
        bootstrap_mean_ci=bootstrap_mean_ci,
        bootstrap_median_ci=bootstrap_median_ci,
    )


# ============================================================================
# Hypothesis tests
# ============================================================================


def welch_t_test(
    samples_a: Sequence[float], samples_b: Sequence[float]
) -> Tuple[float, float, float]:
    """Welch's unequal-variance t-test. Returns ``(t, dof, two_sided_p)``."""
    na, nb = len(samples_a), len(samples_b)
    mean_a = statistics.fmean(samples_a)
    mean_b = statistics.fmean(samples_b)
    var_a = statistics.variance(samples_a)
    var_b = statistics.variance(samples_b)
    se_a = var_a / na
    se_b = var_b / nb
    se = math.sqrt(se_a + se_b)
    if se == 0.0:
        return 0.0, float(na + nb - 2), 1.0
    t = (mean_b - mean_a) / se
    denom = 0.0
    if na > 1:
        denom += se_a**2 / (na - 1)
    if nb > 1:
        denom += se_b**2 / (nb - 1)
    dof = (se_a + se_b) ** 2 / denom if denom > 0 else float(na + nb - 2)
    return t, dof, _t_sf_two_sided(t, dof)


def cohens_d(samples_a: Sequence[float], samples_b: Sequence[float]) -> float:
    """Cohen's d for B relative to A, using the pooled standard deviation."""
    na, nb = len(samples_a), len(samples_b)
    if na < 2 or nb < 2:
        return float("nan")
    var_a = statistics.variance(samples_a)
    var_b = statistics.variance(samples_b)
    pooled = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled == 0.0:
        return 0.0
    return (statistics.fmean(samples_b) - statistics.fmean(samples_a)) / pooled


def _ranks_with_ties(values: Sequence[float]) -> Tuple[List[float], float]:
    """Average ranks (1-based) plus the tie correction ``sum(t^3 - t)``."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_correction = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Ranks i+1 .. j+1 are tied; they all get the average rank.
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_correction += t**3 - t
        i = j + 1
    return ranks, tie_correction


def mann_whitney_u(
    samples_a: Sequence[float], samples_b: Sequence[float]
) -> Tuple[float, float, float]:
    """Mann-Whitney U test (normal approximation, tie- and continuity-corrected).

    Returns ``(U_a, two_sided_p, rank_biserial)`` where ``U_a`` is the statistic
    for side A and the rank-biserial correlation is positive when side B tends
    to be slower than side A.
    """
    na, nb = len(samples_a), len(samples_b)
    ranks, tie_correction = _ranks_with_ties(list(samples_a) + list(samples_b))
    rank_sum_a = math.fsum(ranks[:na])
    u_a = rank_sum_a - na * (na + 1) / 2.0
    rank_biserial = 1.0 - 2.0 * u_a / (na * nb)

    total = na + nb
    mu = na * nb / 2.0
    variance = (na * nb / 12.0) * (
        (total + 1) - tie_correction / (total * (total - 1.0))
    )
    if variance <= 0.0:
        return u_a, 1.0, rank_biserial
    diff = u_a - mu
    # Continuity correction, applied toward the mean.
    corrected = max(abs(diff) - 0.5, 0.0)
    z = corrected / math.sqrt(variance)
    p = 2.0 * (1.0 - _norm_cdf(z))
    return u_a, min(max(p, 0.0), 1.0), rank_biserial


def _magnitude(value: float, thresholds: Tuple[float, float, float]) -> str:
    v = abs(value)
    if math.isnan(v):
        return "unknown"
    small, medium, large = thresholds
    if v < small:
        return "negligible"
    if v < medium:
        return "small"
    if v < large:
        return "medium"
    return "large"


def _bootstrap_pct_change_ci(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    confidence: float,
    resamples: int,
    rng: random.Random,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the percent change in mean latency."""
    na, nb = len(samples_a), len(samples_b)
    resamples = _effective_resamples(max(na, nb), resamples)
    if na < 2 or nb < 2 or resamples < 2:
        return (float("nan"), float("nan"))
    data_a = list(samples_a)
    data_b = list(samples_b)
    changes = []
    for _ in range(resamples):
        mean_a = statistics.fmean(rng.choices(data_a, k=na))
        mean_b = statistics.fmean(rng.choices(data_b, k=nb))
        if mean_a == 0.0:
            continue
        changes.append((mean_b - mean_a) / mean_a * 100.0)
    if not changes:
        return (float("nan"), float("nan"))
    changes.sort()
    return _percentile_ci(changes, confidence)


def compare_samples(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Optional[ComparisonResult]:
    """Compare two sets of latency samples, picking the test from the data."""
    data_a = [float(v) for v in samples_a]
    data_b = [float(v) for v in samples_b]
    if len(data_a) < 2 or len(data_b) < 2:
        return None

    # Step 1: normality of each side (Shapiro-Wilk).
    def _normality(data: List[float]) -> Optional[NormalityResult]:
        tested = _subsample(data, SHAPIRO_MAX_SAMPLES)
        result = shapiro_wilk(tested)
        if result is None:
            return None
        w, p = result
        return NormalityResult(
            statistic=w,
            p_value=p,
            normal=p > alpha,
            n_tested=len(tested),
            n_total=len(data),
        )

    normality_a = _normality(data_a)
    normality_b = _normality(data_b)
    normal = bool(
        normality_a and normality_b and normality_a.normal and normality_b.normal
    )

    # Step 2: parametric when both sides look normal, rank-based otherwise.
    if normal:
        statistic, dof, p_value = welch_t_test(data_a, data_b)
        test_name = "Welch's t-test"
        effect_name = "Cohen's d"
        effect_size = cohens_d(data_a, data_b)
        effect_magnitude = _magnitude(effect_size, (0.2, 0.5, 0.8))
    else:
        statistic, p_value, effect_size = mann_whitney_u(data_a, data_b)
        dof = None
        test_name = "Mann-Whitney U test"
        effect_name = "rank-biserial correlation"
        effect_magnitude = _magnitude(effect_size, (0.1, 0.3, 0.5))

    # Step 3: percent change, always with a bootstrap CI.
    mean_a = statistics.fmean(data_a)
    mean_b = statistics.fmean(data_b)
    pct_change = (mean_b - mean_a) / mean_a * 100.0 if mean_a else float("nan")
    rng = random.Random(seed)
    pct_change_ci = _bootstrap_pct_change_ci(
        data_a, data_b, confidence, bootstrap_resamples, rng
    )

    return ComparisonResult(
        normality_a=normality_a,
        normality_b=normality_b,
        normal=normal,
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        significant=p_value < alpha,
        dof=dof,
        effect_name=effect_name,
        effect_size=effect_size,
        effect_magnitude=effect_magnitude,
        pct_change=pct_change,
        pct_change_ci=pct_change_ci,
        confidence=confidence,
    )


def _has_enough_samples(samples: Sequence[float], label: str) -> bool:
    """Warn and return False if a side has fewer than ``MIN_SAMPLE`` samples."""
    n = len(samples)
    if n < MIN_SAMPLE:
        logger.warning(
            f"Skipping latency analysis, {label} has only {n} "
            f"sample(s) (need at least {MIN_SAMPLE})"
        )
        return False
    return True


def analyze_latency(
    samples_a: Sequence[float],
    samples_b: Optional[Sequence[float]] = None,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Optional[LatencyAnalysis]:
    """Analyze one or two sets of latency samples.

    With only ``samples_a`` this reports descriptive statistics and confidence
    intervals for that side. With both sides it additionally runs the normality
    test, the matching hypothesis test, and the percent change with its CI.

    Returns None (after warning) when a side has fewer than ``MIN_SAMPLE``
    samples: too few measurements to say anything useful about the spread.
    """
    if not _has_enough_samples(samples_a, "side A"):
        return None
    if samples_b is not None and not _has_enough_samples(samples_b, "side B"):
        return None

    stats_a = compute_sample_stats(samples_a, confidence, bootstrap_resamples, seed)
    if stats_a is None:
        return None
    stats_b = None
    comparison = None
    if samples_b:
        stats_b = compute_sample_stats(samples_b, confidence, bootstrap_resamples, seed)
        if stats_b is not None:
            comparison = compare_samples(
                samples_a, samples_b, confidence, alpha, bootstrap_resamples, seed
            )
    return LatencyAnalysis(side_a=stats_a, side_b=stats_b, comparison=comparison)


# ============================================================================
# Reporting
# ============================================================================


def _fmt(value: float, precision: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{precision}f}"


def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def format_sample_stats(stats: SampleStats, label: str, indent: str = "") -> List[str]:
    """Human-readable descriptive statistics for one side (values in ms)."""
    pct = int(round(stats.confidence * 100))
    return [
        f"{indent}{label} (n={stats.n}):",
        f"{indent}  min={_fmt(stats.min)}  max={_fmt(stats.max)}  "
        f"mean={_fmt(stats.mean)}  median={_fmt(stats.median)}",
        f"{indent}  stddev={_fmt(stats.stddev)}  stderr={_fmt(stats.stderr)}  "
        f"CV={_fmt(stats.cv, 2)}%  IQR={_fmt(stats.iqr)} "
        f"[Q1={_fmt(stats.q1)}, Q3={_fmt(stats.q3)}]",
        f"{indent}  {pct}% CI (t): [{_fmt(stats.mean_ci[0])}, {_fmt(stats.mean_ci[1])}]  "
        f"bootstrap mean: [{_fmt(stats.bootstrap_mean_ci[0])}, {_fmt(stats.bootstrap_mean_ci[1])}]  "
        f"bootstrap median: [{_fmt(stats.bootstrap_median_ci[0])}, {_fmt(stats.bootstrap_median_ci[1])}]",
    ]


def format_latency_analysis(
    analysis: LatencyAnalysis,
    label_a: str = "Side A",
    label_b: str = "Side B",
    indent: str = "",
) -> List[str]:
    """Render a :class:`LatencyAnalysis` as printable lines."""
    lines = format_sample_stats(analysis.side_a, label_a, indent)
    if analysis.side_b is not None:
        lines.extend(format_sample_stats(analysis.side_b, label_b, indent))

    comparison = analysis.comparison
    if comparison is None:
        return lines

    def _normality_str(result: Optional[NormalityResult], label: str) -> str:
        if result is None:
            return f"{label}: n/a"
        subsampled = (
            f", subsampled {result.n_tested}/{result.n_total}"
            if result.n_tested < result.n_total
            else ""
        )
        verdict = "normal" if result.normal else "non-normal"
        return f"{label}: W={_fmt(result.statistic)}, p={_fmt_p(result.p_value)} ({verdict}{subsampled})"

    lines.append(
        f"{indent}  Shapiro-Wilk: "
        f"{_normality_str(comparison.normality_a, label_a)}; "
        f"{_normality_str(comparison.normality_b, label_b)}"
    )
    stat_label = "t" if comparison.dof is not None else "U"
    dof_str = f", dof={_fmt(comparison.dof, 1)}" if comparison.dof is not None else ""
    verdict = "significant" if comparison.significant else "not significant"
    lines.append(
        f"{indent}  {comparison.test_name}: {stat_label}={_fmt(comparison.statistic, 3)}{dof_str}, "
        f"p={_fmt_p(comparison.p_value)} ({verdict} at alpha={comparison.alpha})"
    )
    lines.append(
        f"{indent}  {comparison.effect_name}: {_fmt(comparison.effect_size, 3)} "
        f"({comparison.effect_magnitude})"
    )
    pct = int(round(comparison.confidence * 100))
    lines.append(
        f"{indent}  Percent change ({label_b} vs {label_a}): "
        f"{comparison.pct_change:+.2f}% "
        f"[{pct}% CI: {comparison.pct_change_ci[0]:+.2f}%, {comparison.pct_change_ci[1]:+.2f}%]"
    )
    return lines
