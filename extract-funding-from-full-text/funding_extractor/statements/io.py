import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

from funding_extractor.statements.models import FundingStatement


def write_statements_jsonl(
    output_path: Path,
    statements: Iterable[Tuple[str, FundingStatement]],
) -> int:
    count = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for document_id, stmt in statements:
            record = {
                "document_id": document_id,
                "statement": stmt.statement,
                "score": stmt.score,
                "query": stmt.query,
                "paragraph_idx": stmt.paragraph_idx,
                "is_problematic": stmt.is_problematic,
                "original": stmt.original,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_statements_jsonl(input_path: Path) -> Iterator[Dict]:
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
