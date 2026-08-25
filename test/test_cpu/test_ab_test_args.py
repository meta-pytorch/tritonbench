import unittest

from tritonbench.utils.ab_test import base_global_args, merge_global_args


class BaseGlobalArgsTest(unittest.TestCase):
    def test_keeps_globals_and_drops_side_selectors(self):
        argv = ["--op", "softmax", "--num-inputs", "1", "--side-a=--rep 1000"]
        self.assertEqual(
            base_global_args(argv), ["--op", "softmax", "--num-inputs", "1"]
        )

    def test_drops_operator_args(self):
        # --M is a softmax operator arg, not a tritonbench global one.
        argv = ["--op", "softmax", "--M", "4096"]
        self.assertEqual(base_global_args(argv), ["--op", "softmax"])

    def test_no_globals(self):
        self.assertEqual(base_global_args([]), [])


class MergeGlobalArgsTest(unittest.TestCase):
    def test_side_args_are_appended(self):
        self.assertEqual(
            merge_global_args(["--op", "softmax"], ["--rep", "1000"]),
            ["--op", "softmax", "--rep", "1000"],
        )

    def test_side_arg_overrides_base_in_place(self):
        self.assertEqual(
            merge_global_args(["--op", "softmax", "--rep", "100"], ["--rep", "1000"]),
            ["--op", "softmax", "--rep", "1000"],
        )

    def test_equals_form_overrides_space_form(self):
        self.assertEqual(
            merge_global_args(["--rep", "100"], ["--rep=1000"]), ["--rep=1000"]
        )

    def test_valueless_flags_are_preserved(self):
        self.assertEqual(
            merge_global_args(["--op", "softmax"], ["--cudagraph"]),
            ["--op", "softmax", "--cudagraph"],
        )

    def test_empty_sides(self):
        self.assertEqual(merge_global_args([], []), [])
        self.assertEqual(merge_global_args(["--op", "softmax"], []), ["--op", "softmax"])


if __name__ == "__main__":
    unittest.main()
