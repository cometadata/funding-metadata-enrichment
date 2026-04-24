import json
import logging
import os
from pathlib import Path
from typing import Optional

from funding_statement_extractor.statements.models import ProcessingResults

logger = logging.getLogger(__name__)


def load_existing_results(output_file: Path) -> Optional[ProcessingResults]:
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return ProcessingResults.from_dict(data)
        except Exception as exc:
            logger.warning("Warning: Could not load existing results: %s", exc)
    return None


def save_results(results: ProcessingResults, output_file: Path) -> None:
    temp_file = str(output_file) + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as fh:
        json.dump(results.to_dict(), fh, indent=2, ensure_ascii=False)
    os.replace(temp_file, output_file)
