import unittest

from tritonbench.components.do_bench.run import (
    MAX_CUDAGRAPH_REPEAT,
    _get_cudagraph_n_repeat,
)


class CudaGraphRepeatTest(unittest.TestCase):
    def test_caps_repeat_count(self):
        self.assertEqual(
            _get_cudagraph_n_repeat(rep=100, estimate_ms=0.01),
            MAX_CUDAGRAPH_REPEAT,
        )

    def test_preserves_repeat_count_below_cap(self):
        self.assertEqual(_get_cudagraph_n_repeat(rep=100, estimate_ms=0.2), 500)

    def test_handles_zero_estimate(self):
        self.assertEqual(
            _get_cudagraph_n_repeat(rep=100, estimate_ms=0),
            MAX_CUDAGRAPH_REPEAT,
        )
