import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from funding_extractor.models import ProcessingResults


def read_statements_by_document(input_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            document_id = record.get("document_id", "unknown")
            grouped[document_id].append(record)
    return dict(grouped)


def write_results_json(results: ProcessingResults, output_path: Path) -> None:
    temp_file = str(output_path) + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as fh:
        json.dump(results.to_dict(), fh, indent=2, ensure_ascii=False)
    os.replace(temp_file, output_path)


def load_results_json(input_path: Path) -> ProcessingResults:
    with open(input_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return ProcessingResults.from_dict(data)
