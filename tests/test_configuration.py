"""Tests for configuration utilities."""

import tempfile
import unittest
from pathlib import Path

from src.configuration import load_configuration


class TestConfiguration(unittest.TestCase):
    """Tests for configuration loading."""

    def test_load_configuration(self) -> None:
        """Load configuration values from a YAML file."""
        yaml_content = """
problem:
  name: test_problem

penalties:
  assignment: 10.0
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yml"
            config_path.write_text(yaml_content, encoding="utf-8")

            config = load_configuration(config_path)

        self.assertEqual(config["problem"]["name"], "test_problem")
        self.assertEqual(config["penalties"]["assignment"], 10.0)
