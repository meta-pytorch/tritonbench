import json
import os
import subprocess
import sys
import unittest
from unittest import mock

from tritonbench.utils.path_utils import add_ld_library_path


class AddLdLibraryPathTest(unittest.TestCase):
    def setUp(self):
        # Keep cleanup safe even if the context replaces os.environ.
        self.environ = os.environ
        self.addCleanup(setattr, os, "environ", self.environ)
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop("LD_LIBRARY_PATH", None)

    def test_restores_missing_path(self):
        with add_ld_library_path("/temporary/lib"):
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/temporary/lib")
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)

    def test_restores_existing_path(self):
        for original in ("", "/existing/lib"):
            with self.subTest(original=original):
                os.environ["LD_LIBRARY_PATH"] = original
                with add_ld_library_path("/temporary/lib"):
                    expected = (
                        f"{original}:/temporary/lib" if original else "/temporary/lib"
                    )
                    self.assertEqual(os.environ["LD_LIBRARY_PATH"], expected)
                self.assertEqual(os.environ["LD_LIBRARY_PATH"], original)

    def test_preserves_environment_object(self):
        with add_ld_library_path("/temporary/lib"):
            pass
        self.assertIs(os.environ, self.environ)

    def test_restores_path_after_exception(self):
        os.environ["LD_LIBRARY_PATH"] = "/existing/lib"
        with self.assertRaisesRegex(ImportError, "optional backend"):
            with add_ld_library_path("/temporary/lib"):
                raise ImportError("optional backend")
        self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/existing/lib")

    def test_nested_contexts_restore_each_path(self):
        with add_ld_library_path("/outer/lib"):
            with add_ld_library_path("/inner/lib"):
                self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/outer/lib:/inner/lib")
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], "/outer/lib")
        self.assertNotIn("LD_LIBRARY_PATH", os.environ)

    def test_preserves_unrelated_environment_updates(self):
        with add_ld_library_path("/temporary/lib"):
            os.environ["TRITONBENCH_PATH_UTILS_TEST"] = "updated"
        self.assertEqual(os.environ["TRITONBENCH_PATH_UTILS_TEST"], "updated")

    def test_subprocess_inherits_restored_environment(self):
        for original in (None, "", "/existing/lib"):
            with self.subTest(original=original):
                if original is None:
                    os.environ.pop("LD_LIBRARY_PATH", None)
                else:
                    os.environ["LD_LIBRARY_PATH"] = original
                os.environ.pop("TRITONBENCH_PATH_UTILS_TEST", None)
                with add_ld_library_path("/temporary/lib"):
                    pass
                os.environ["TRITONBENCH_PATH_UTILS_TEST"] = "after-context"
                # No explicit env: verify the real process environment, not a
                # replacement Python mapping passed directly to the child.
                output = subprocess.check_output(
                    [
                        sys.executable,
                        "-c",
                        "import json, os; print(json.dumps(["
                        "os.environ.get('LD_LIBRARY_PATH'), "
                        "os.environ.get('TRITONBENCH_PATH_UTILS_TEST')]))",
                    ],
                    text=True,
                )
                self.assertEqual(json.loads(output), [original, "after-context"])


if __name__ == "__main__":
    unittest.main()
