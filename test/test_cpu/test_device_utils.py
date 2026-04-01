import unittest
from unittest.mock import patch

from parameterized import parameterized
from tritonbench.utils.device_utils import (
    compute_input_shards,
    parse_device_range,
    validate_device_ids,
)


class ParseDeviceRangeTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("single_device", "0", [0]),
            ("single_high_device", "7", [7]),
            ("range", "0-2", [0, 1, 2]),
            ("range_same", "0-0", [0]),
            ("range_and_single", "0-2,5", [0, 1, 2, 5]),
            ("multiple_singles", "7,3,1", [7, 3, 1]),
            ("range_and_singles", "0-1,4,6-8", [0, 1, 4, 6, 7, 8]),
            ("large_range", "0-7", [0, 1, 2, 3, 4, 5, 6, 7]),
            ("spaces_around", " 0 , 2 ", [0, 2]),
        ]
    )
    def test_valid_inputs(self, _name, device_str, expected):
        self.assertEqual(parse_device_range(device_str), expected)

    @parameterized.expand(
        [
            ("empty_string", ""),
            ("whitespace_only", "   "),
            ("non_integer", "abc"),
            ("negative_id", "-1"),
            ("reversed_range", "3-1"),
            ("negative_range_start", "-2-1"),
            ("empty_segment", "0,,2"),
        ]
    )
    def test_invalid_inputs(self, _name, device_str):
        with self.assertRaises(ValueError):
            parse_device_range(device_str)


class ComputeInputShardsTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "even_split",
                100,
                4,
                [(0, 25), (25, 25), (50, 25), (75, 25)],
            ),
            (
                "uneven_split",
                10,
                3,
                [(0, 4), (4, 3), (7, 3)],
            ),
            (
                "more_devices_than_inputs",
                1,
                4,
                [(0, 1)],
            ),
            (
                "single_device",
                50,
                1,
                [(0, 50)],
            ),
            (
                "inputs_equal_devices",
                4,
                4,
                [(0, 1), (1, 1), (2, 1), (3, 1)],
            ),
            (
                "remainder_distributed",
                7,
                3,
                [(0, 3), (3, 2), (5, 2)],
            ),
        ]
    )
    def test_valid_shards_no_min(self, _name, total_inputs, num_devices, expected):
        self.assertEqual(
            compute_input_shards(total_inputs, num_devices, min_inputs_per_device=1),
            expected,
        )

    def test_all_inputs_covered(self):
        total = 100
        shards = compute_input_shards(total, 4)
        self.assertEqual(sum(size for _, size in shards), total)

    def test_shards_are_contiguous(self):
        shards = compute_input_shards(100, 4)
        for i in range(1, len(shards)):
            prev_start, prev_size = shards[i - 1]
            curr_start, _ = shards[i]
            self.assertEqual(prev_start + prev_size, curr_start)

    def test_zero_total_inputs(self):
        with self.assertRaises(ValueError):
            compute_input_shards(0, 2)

    def test_zero_devices(self):
        with self.assertRaises(ValueError):
            compute_input_shards(10, 0)

    def test_negative_total_inputs(self):
        with self.assertRaises(ValueError):
            compute_input_shards(-5, 2)

    def test_negative_devices(self):
        with self.assertRaises(ValueError):
            compute_input_shards(10, -1)

    @parameterized.expand(
        [
            (
                "enough_inputs_for_all_devices",
                100,
                4,
                10,
                [(0, 25), (25, 25), (50, 25), (75, 25)],
            ),
            (
                "cap_to_2_devices",
                25,
                4,
                10,
                [(0, 13), (13, 12)],
            ),
            (
                "cap_to_1_device",
                15,
                4,
                10,
                [(0, 15)],
            ),
            (
                "exactly_at_min_boundary",
                40,
                4,
                10,
                [(0, 10), (10, 10), (20, 10), (30, 10)],
            ),
            (
                "fewer_than_min_still_uses_1_device",
                5,
                4,
                10,
                [(0, 5)],
            ),
            (
                "cap_to_3_devices",
                35,
                8,
                10,
                [(0, 12), (12, 12), (24, 11)],
            ),
        ]
    )
    def test_min_inputs_per_device(
        self, _name, total_inputs, num_devices, min_per_device, expected
    ):
        self.assertEqual(
            compute_input_shards(total_inputs, num_devices, min_per_device), expected
        )

    def test_min_inputs_per_device_all_inputs_covered(self):
        total = 25
        shards = compute_input_shards(total, 4, min_inputs_per_device=10)
        self.assertEqual(sum(size for _, size in shards), total)

    def test_min_inputs_per_device_each_shard_at_least_min(self):
        shards = compute_input_shards(100, 8, min_inputs_per_device=10)
        for _, size in shards:
            self.assertGreaterEqual(size, 10)

    def test_default_min_is_10(self):
        shards_default = compute_input_shards(15, 4)
        shards_explicit = compute_input_shards(15, 4, min_inputs_per_device=10)
        self.assertEqual(shards_default, shards_explicit)


class ValidateDeviceIdsTest(unittest.TestCase):
    @patch("tritonbench.utils.device_utils.torch.cuda.is_available", return_value=False)
    def test_cuda_not_available(self, _mock):
        with self.assertRaises(RuntimeError) as ctx:
            validate_device_ids([0, 1])
        self.assertIn("CUDA is not available", str(ctx.exception))

    @patch("tritonbench.utils.device_utils.torch.cuda.device_count", return_value=4)
    @patch("tritonbench.utils.device_utils.torch.cuda.is_available", return_value=True)
    def test_all_devices_valid(self, _mock_avail, _mock_count):
        validate_device_ids([0, 1, 2, 3])

    @patch("tritonbench.utils.device_utils.torch.cuda.device_count", return_value=4)
    @patch("tritonbench.utils.device_utils.torch.cuda.is_available", return_value=True)
    def test_device_out_of_range(self, _mock_avail, _mock_count):
        with self.assertRaises(RuntimeError) as ctx:
            validate_device_ids([0, 5])
        self.assertIn("[5]", str(ctx.exception))
        self.assertIn("4 GPU(s)", str(ctx.exception))

    @patch("tritonbench.utils.device_utils.torch.cuda.device_count", return_value=2)
    @patch("tritonbench.utils.device_utils.torch.cuda.is_available", return_value=True)
    def test_multiple_devices_out_of_range(self, _mock_avail, _mock_count):
        with self.assertRaises(RuntimeError) as ctx:
            validate_device_ids([0, 2, 5])
        self.assertIn("[2, 5]", str(ctx.exception))

    @patch("tritonbench.utils.device_utils.torch.cuda.device_count", return_value=8)
    @patch("tritonbench.utils.device_utils.torch.cuda.is_available", return_value=True)
    def test_single_valid_device(self, _mock_avail, _mock_count):
        validate_device_ids([7])
