"""Helpers for loading configuration files (queries, prompts, patterns)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from funding_extractor.exceptions import ConfigurationError


def _base_config_dir(custom_config_dir: Optional[str] = None) -> Path:
    if custom_config_dir:
        return Path(custom_config_dir)
    return Path(__file__).resolve().parents[2] / "configs"


def get_config_path(config_type: str, filename: str, custom_config_dir: Optional[str] = None) -> Path:
    base_dir = _base_config_dir(custom_config_dir)
    return base_dir / config_type / filename


def load_queries(queries_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> Dict[str, str]:
    if queries_file:
        config_path = Path(queries_file)
    else:
        config_path = get_config_path("queries", "default.yaml", custom_config_dir)

    if not config_path.exists():
        raise ConfigurationError(
            f"Query configuration file not found at '{config_path}'. "
            "Provide a valid path via --queries or place default.yaml under configs/queries."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
        return data.get("queries", {})


def load_extraction_prompt(prompt_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> str:
    if prompt_file:
        config_path = Path(prompt_file)
    else:
        config_path = get_config_path("prompts", "extraction_prompt.txt", custom_config_dir)

    if not config_path.exists():
        raise ConfigurationError(
            f"Extraction prompt file not found at '{config_path}'. "
            "Provide --prompt-file or place extraction_prompt.txt under configs/prompts."
        )

    return config_path.read_text(encoding="utf-8")


def load_extraction_examples(
    examples_file: Optional[str] = None, custom_config_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    if examples_file:
        config_path = Path(examples_file)
    else:
        config_path = get_config_path("prompts", "extraction_examples.json", custom_config_dir)

    if not config_path.exists():
        raise ConfigurationError(
            f"Extraction examples file not found at '{config_path}'. "
            "Provide --examples-file or place extraction_examples.json under configs/prompts."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_funding_patterns(patterns_file: Optional[str] = None, custom_config_dir: Optional[str] = None) -> List[str]:
    if patterns_file:
        config_path = Path(patterns_file)
    else:
        config_path = get_config_path("patterns", "funding_patterns.yaml", custom_config_dir)

    if not config_path.exists():
        raise ConfigurationError(
            f"Funding patterns file not found at '{config_path}'. "
            "Provide --patterns-file or place funding_patterns.yaml under configs/patterns."
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
        return data.get("patterns", [])
