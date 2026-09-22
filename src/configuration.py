"""Configuration loading utilities."""

from pathlib import Path

import yaml


def load_configuration(config_path: Path) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
