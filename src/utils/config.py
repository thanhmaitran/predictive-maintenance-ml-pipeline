"""
Small helper for loading config/config.yaml so paths, random seeds and
column definitions live in one place instead of being hardcoded across
every module.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(path: str = str(DEFAULT_CONFIG_PATH)) -> Dict[str, Any]:
    """Load the project YAML config.

    Args:
        path: Path to config.yaml.

    Returns:
        Parsed config as a dict.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
