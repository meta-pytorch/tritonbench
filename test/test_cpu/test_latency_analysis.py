import math
import random
import statistics
import unittest

from tritonbench.components.do_bench.latency_analysis import (
    _t_ppf,
    analyze_latency,
    cohens_d,
    compare_samples,
    compute_sample_stats,
    mann_whitney_u,
    MIN_SAMPLE,
    shapiro_wilk,
    welch_t_test,
)


class TestDistributionHelpers(unittest.TestCase):
    def test_t_ppf_matches_published_table(self):
        # Two-sided 95% critical values from a standard Student-t table.
        for df, expected in [(1, 12.7062), (10, 2.2281), (30, 2.0423), (100, 1.9840)]:
            self.assertAlmostEqual(_t_ppf(0.975, df), expected, places=4)
        # Large df converges to the normal quantile.
        self.assertAlmostEqual(_t_ppf(0.975, 10**7), 1.9600, places=4)


class TestSampleStats(unittest.TestCase):
    def test_descriptive_statistics(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_sample_stats(data)
        self.assertEqual(stats.n, 5)
        self.assertEqual(stats.min, 1.0)
        self.assertEqual(stats.max, 5.0)
        self.assertAlmostEqual(stats.mean, 3.0)
        self.assertAlmostEqual(stats.median, 3.0)
        self.assertAlmostEqual(stats.stddev, statistics.stdev(data))
        self.assertAlmostEqual(stats.stderr, statistics.stdev(data) / math.sqrt(5))
        self.assertAlmostEqual(stats.cv, statistics.stdev(data) / 3.0 * 100)
        # Inclusive quartiles of [1..5] are 2 and 4.
        self.assertAlmostEqual(stats.q1, 2.0)
        self.assertAlmostEqual(stats.q3, 4.0)
        self.assertAlmostEqual(stats.iqr, 2.0)

    def test_confidence_intervals_bracket_the_mean(self):
        rng = random.Random(0)
        data = [rng.gauss(1.0, 0.05) for _ in range(300)]
        stats = compute_sample_stats(data)
        for low, high in (
            stats.mean_ci,
            stats.bootstrap_mean_ci,
            stats.bootstrap_median_ci,
        ):
            self.assertLess(low, high)
        self.assertLess(stats.mean_ci[0], stats.mean)
        self.assertGreater(stats.mean_ci[1], stats.mean)
        # The bootstrap CI of the mean should closely track the t interval.
        self.assertAlmostEqual(stats.mean_ci[0], stats.bootstrap_mean_ci[0], places=3)
        self.assertAlmostEqual(stats.mean_ci[1], stats.bootstrap_mean_ci[1], places=3)

    def test_bootstrap_is_deterministic(self):
        rng = random.Random(1)
        data = [rng.gauss(1.0, 0.05) for _ in range(100)]
        self.assertEqual(
            compute_sample_stats(data).bootstrap_mean_ci,
            compute_sample_stats(data).bootstrap_mean_ci,
        )

    def test_degenerate_samples(self):
        self.assertIsNone(compute_sample_stats([]))
        stats = compute_sample_stats([2.5])
        self.assertEqual(stats.n, 1)
        self.assertEqual(stats.stddev, 0.0)
        self.assertEqual(stats.mean_ci, (2.5, 2.5))


class TestNormality(unittest.TestCase):
    def test_shapiro_wilk_reference_values(self):
        # Textbook example; R's shapiro.test reports W = 0.7888, p = 0.0067.
        data = [148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236]
        w, p = shapiro_wilk(data)
        self.assertAlmostEqual(w, 0.7888, places=4)
        self.assertAlmostEqual(p, 0.0067, places=4)

    def test_shapiro_wilk_on_normal_scores_is_one(self):
        # Feeding back the expected normal order statistics is a perfect fit.
        for n in (4, 5, 6, 11, 50):
            w, p = shapiro_wilk(
                [
                    statistics.NormalDist().inv_cdf((i + 1 - 0.375) / (n + 0.25))
                    for i in range(n)
                ]
            )
            self.assertGreater(w, 0.99)
            self.assertGreater(p, 0.5)

    def test_shapiro_wilk_rejects_exponential(self):
        rng = random.Random(2)
        _, p = shapiro_wilk([rng.expovariate(1.0) for _ in range(200)])
        self.assertLess(p, 1e-6)

    def test_shapiro_wilk_undefined_cases(self):
        self.assertIsNone(shapiro_wilk([1.0, 2.0]))
        self.assertIsNone(shapiro_wilk([1.0, 1.0, 1.0, 1.0]))


class TestHypothesisTests(unittest.TestCase):
    def test_welch_t_test_reference_values(self):
        # Wikipedia's Welch t-test example: t = 2.46, dof = 24.9, p = 0.021.
        a = [
            27.5,
            21,
            19,
            23.6,
            17,
            17.9,
            16.9,
            20.1,
            21.9,
            22.6,
            23.1,
            19.6,
            19,
            21.7,
            21.4,
        ]
        b = [
            27.1,
            22,
            20.8,
            23.4,
            23.4,
            23.5,
            25.8,
            22,
            24.8,
            20.2,
            21.9,
            22.1,
            22.9,
            20.5,
            24.4,
        ]
        t, dof, p = welch_t_test(a, b)
        self.assertAlmostEqual(t, 2.455, places=3)
        self.assertAlmostEqual(dof, 24.98, places=1)
        self.assertAlmostEqual(p, 0.0214, places=4)

    def test_cohens_d_sign_and_magnitude(self):
        rng = random.Random(3)
        a = [rng.gauss(1.0, 0.1) for _ in range(500)]
        b = [x + 0.1 for x in a]
        # A one-stddev shift upward is a d of about 1.0.
        self.assertAlmostEqual(cohens_d(a, b), 1.0, delta=0.15)
        self.assertLess(cohens_d(b, a), 0)

    def test_mann_whitney_u_fully_separated(self):
        u, p, rank_biserial = mann_whitney_u([1, 2, 3, 4], [5, 6, 7, 8])
        self.assertEqual(u, 0.0)
        self.assertAlmostEqual(rank_biserial, 1.0)
        self.assertLess(p, 0.05)

    def test_mann_whitney_u_identical_samples(self):
        u, p, rank_biserial = mann_whitney_u([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(u, 4.5)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(rank_biserial, 0.0)


def _normal_sample(n, mean, stddev):
    """A sample that follows the normal distribution exactly.

    Drawing pseudo-random gaussians would make the normality branch flaky:
    Shapiro-Wilk rejects a genuinely normal sample 5% of the time.
    """
    dist = statistics.NormalDist(mean, stddev)
    return [dist.inv_cdf((i + 0.5) / n) for i in range(n)]


class TestComparison(unittest.TestCase):
    def test_normal_samples_use_welch(self):
        a = _normal_sample(300, 1.0, 0.02)
        b = _normal_sample(300, 1.05, 0.02)
        result = compare_samples(a, b)
        self.assertTrue(result.normal)
        self.assertEqual(result.test_name, "Welch's t-test")
        self.assertEqual(result.effect_name, "Cohen's d")
        self.assertIsNotNone(result.dof)
        self.assertTrue(result.significant)
        self.assertAlmostEqual(result.pct_change, 5.0, delta=1.0)
        low, high = result.pct_change_ci
        self.assertLess(low, result.pct_change)
        self.assertGreater(high, result.pct_change)

    def test_skewed_samples_use_mann_whitney(self):
        rng = random.Random(5)
        a = [1.0 + rng.expovariate(1.0) for _ in range(300)]
        b = [x * 1.5 for x in a]
        result = compare_samples(a, b)
        self.assertFalse(result.normal)
        self.assertEqual(result.test_name, "Mann-Whitney U test")
        self.assertEqual(result.effect_name, "rank-biserial correlation")
        self.assertIsNone(result.dof)
        self.assertTrue(result.significant)
        # Positive effect and percent change mean side B is slower.
        self.assertGreater(result.effect_size, 0)
        self.assertGreater(result.pct_change, 0)

    def test_identical_samples_are_not_significant(self):
        rng = random.Random(8)
        a = [1.0 + rng.expovariate(1.0) for _ in range(200)]
        result = compare_samples(a, list(a))
        self.assertFalse(result.significant)
        self.assertAlmostEqual(result.effect_size, 0.0)
        self.assertAlmostEqual(result.pct_change, 0.0)

    def test_too_few_samples(self):
        self.assertIsNone(compare_samples([1.0], [2.0]))


class TestAnalyzeLatency(unittest.TestCase):
    def test_single_side(self):
        rng = random.Random(6)
        analysis = analyze_latency([rng.gauss(1.0, 0.02) for _ in range(100)])
        self.assertIsNotNone(analysis.side_a)
        self.assertIsNone(analysis.side_b)
        self.assertIsNone(analysis.comparison)

    def test_both_sides(self):
        rng = random.Random(7)
        a = [rng.gauss(1.0, 0.02) for _ in range(200)]
        b = [rng.gauss(0.9, 0.02) for _ in range(200)]
        analysis = analyze_latency(a, b)
        self.assertIsNotNone(analysis.side_b)
        self.assertIsNotNone(analysis.comparison)
        # Side B is faster, so the percent change is negative.
        self.assertLess(analysis.comparison.pct_change, 0)
        self.assertLess(analysis.comparison.pct_change_ci[1], 0)

    def test_empty_samples(self):
        self.assertIsNone(analyze_latency([]))

    def test_too_few_samples_is_skipped_with_a_warning(self):
        rng = random.Random(9)
        short = [rng.gauss(1.0, 0.02) for _ in range(MIN_SAMPLE - 1)]
        enough = [rng.gauss(1.0, 0.02) for _ in range(MIN_SAMPLE)]
        log_name = "tritonbench.components.do_bench.latency_analysis"

        with self.assertLogs(log_name, level="WARNING") as logs:
            self.assertIsNone(analyze_latency(short))
        self.assertIn("side A", logs.output[0])

        # A short side B skips the analysis too, even though side A is fine.
        with self.assertLogs(log_name, level="WARNING") as logs:
            self.assertIsNone(analyze_latency(enough, short))
        self.assertIn("side B", logs.output[0])

        # Exactly MIN_SAMPLE samples is enough.
        self.assertIsNotNone(analyze_latency(enough))


if __name__ == "__main__":
    unittest.main()
