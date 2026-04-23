import unittest

import torch

from tritonbench.utils.env_utils import (
    IS_BLACKWELL,
    IS_BLACKWELL_ANY,
    IS_BLACKWELL_CONSUMER,
    is_blackwell,
    is_blackwell_consumer,
)


class BlackwellDetectionTest(unittest.TestCase):
    def test_constants_are_bool(self):
        self.assertIsInstance(IS_BLACKWELL, bool)
        self.assertIsInstance(IS_BLACKWELL_CONSUMER, bool)
        self.assertIsInstance(IS_BLACKWELL_ANY, bool)

    def test_functions_return_bool(self):
        self.assertIsInstance(is_blackwell(), bool)
        self.assertIsInstance(is_blackwell_consumer(), bool)

    def test_mutually_exclusive(self):
        # A single GPU cannot be both datacenter (sm_100) and consumer
        # (sm_120) Blackwell simultaneously.
        self.assertFalse(IS_BLACKWELL and IS_BLACKWELL_CONSUMER)

    def test_any_is_disjunction(self):
        self.assertEqual(IS_BLACKWELL_ANY, IS_BLACKWELL or IS_BLACKWELL_CONSUMER)

    def test_cpu_only_returns_false(self):
        if not torch.cuda.is_available():
            self.assertFalse(IS_BLACKWELL)
            self.assertFalse(IS_BLACKWELL_CONSUMER)
            self.assertFalse(IS_BLACKWELL_ANY)


if __name__ == "__main__":
    unittest.main()
