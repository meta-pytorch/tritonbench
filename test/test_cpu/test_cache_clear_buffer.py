import unittest
from unittest.mock import MagicMock, patch

import tritonbench.components.do_bench.utils as do_bench_utils
from tritonbench.components.do_bench.utils import prime_cache_clear_buffer


class PrimeCacheClearBufferTest(unittest.TestCase):
    def setUp(self):
        # The primed flag is process-wide; reset it so each test starts cold.
        self._saved = do_bench_utils._cache_clear_buffer_primed
        do_bench_utils._cache_clear_buffer_primed = False

    def tearDown(self):
        do_bench_utils._cache_clear_buffer_primed = self._saved

    def _fake_triton(self):
        triton = MagicMock()
        triton.runtime.driver.active.get_empty_cache_for_benchmark.return_value = (
            "cache"
        )
        return triton

    def test_flushes_once_then_is_a_noop(self):
        triton = self._fake_triton()
        driver = triton.runtime.driver.active
        with patch.dict("sys.modules", {"triton": triton}), patch.object(
            do_bench_utils, "get_device_module"
        ):
            for _ in range(3):
                prime_cache_clear_buffer()
        # Only the first call may touch the buffer, otherwise priming would
        # itself perturb the estimate it is meant to protect.
        driver.clear_cache.assert_called_once_with("cache")
        self.assertTrue(do_bench_utils._cache_clear_buffer_primed)

    def test_unsupported_backend_is_swallowed_and_not_retried(self):
        triton = self._fake_triton()
        triton.runtime.driver.active.get_empty_cache_for_benchmark.side_effect = (
            RuntimeError("no benchmark cache on this backend")
        )
        with patch.dict("sys.modules", {"triton": triton}):
            prime_cache_clear_buffer()
            prime_cache_clear_buffer()
        self.assertEqual(
            triton.runtime.driver.active.get_empty_cache_for_benchmark.call_count, 1
        )


if __name__ == "__main__":
    unittest.main()
