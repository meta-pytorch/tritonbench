import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


class TestDependencyGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with PYPROJECT_PATH.open("rb") as f:
            cls.groups = tomllib.load(f)["dependency-groups"]

    def test_ci_group_matches_uploader_dependencies(self):
        self.assertEqual(
            set(self.groups["ci"]),
            {"boto3", "pyyaml", "requests"},
        )

    def test_dev_nvidia_group_includes_runtime_yaml_dependency(self):
        self.assertTrue(
            {
                "packaging",
                "psutil",
                "tabulate",
                "matplotlib",
                "pyyaml",
                "nvidia-ml-py",
                "transformers==5.0.0rc3",
            }.issubset(set(self.groups["dev-nvidia"]))
        )

    def test_non_nvidia_groups_keep_runtime_import_dependencies(self):
        for group_name in ("dev-amd", "dev-cpu"):
            with self.subTest(group=group_name):
                self.assertIn("pyyaml", self.groups[group_name])
                self.assertIn("transformers==5.0.0rc3", self.groups[group_name])

    def test_dev_numpy_group_keeps_expected_pins(self):
        self.assertEqual(
            set(self.groups["dev-numpy"]),
            {
                "numpy==2.0.2; python_version < '3.10'",
                "numpy==2.1.0; python_version >= '3.10' and python_version < '3.14'",
                "numpy==2.3.4; python_version >= '3.14'",
            },
        )
