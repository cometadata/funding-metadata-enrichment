"""Checkpoint persistence utilities."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from funding_extractor.exceptions import CheckpointError


def get_file_hash(file_path: str) -> str:
    return hashlib.md5(file_path.encode()).hexdigest()


class CheckpointRepository:
    """Repository for reading and writing checkpoint data."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.data: Dict[str, Any] = {
            "processed_files": {},
            "last_update": None,
            "total_processed": 0,
        }

    def load(self, resume: bool = False) -> Dict[str, Any]:
        if resume and self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
                    self.data = json.load(fh)
            except Exception as exc:  # pylint: disable=broad-except
                raise CheckpointError(f"Could not load checkpoint: {exc}") from exc
        return self.data

    def save(self) -> None:
        self.data["last_update"] = datetime.now().isoformat()
        self.data["total_processed"] = len(self.data.get("processed_files", {}))

        temp_file = str(self.checkpoint_path) + ".tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2)
            os.replace(temp_file, self.checkpoint_path)
        except Exception as exc:  # pylint: disable=broad-except
            raise CheckpointError(f"Failed to write checkpoint: {exc}") from exc

    def is_processed(self, doc_hash: str) -> bool:
        return doc_hash in self.data.get("processed_files", {})

    def record(self, doc_hash: str, metadata: Dict[str, Any]) -> None:
        if "processed_files" not in self.data:
            self.data["processed_files"] = {}
        self.data["processed_files"][doc_hash] = metadata
