from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from funding_statement_extractor.exceptions import ConfigurationError


def _base_config_dir(custom_config_dir: Optional[str] = None) -> Path:
    if custom_config_dir:
        return Path(custom_config_dir)
    return Path(__file__).resolve().parents[1] / "configs"


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


def load_funding_patterns(
    patterns_file: Optional[str] = None, custom_config_dir: Optional[str] = None
) -> Tuple[List[str], List[str]]:
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
        return data.get("patterns", []), data.get("negative_patterns", [])
