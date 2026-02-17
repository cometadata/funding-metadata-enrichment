import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from funding_extractor.exceptions import DocumentLoadError


def find_markdown_files(directory: str) -> List[str]:
    md_files: List[str] = []
    path = Path(directory)
    if not path.exists():
        raise DocumentLoadError(f"Directory {directory} does not exist")

    for file_path in path.rglob("*.md"):
        md_files.append(str(file_path))
    return sorted(md_files)


@dataclass
class DocumentPayload:
    document_id: str
    checkpoint_key: str
    file_path: Optional[str] = None
    content: Optional[str] = None

    def load_text(self) -> str:
        if self.content is not None:
            return self.content
        if self.file_path:
            return Path(self.file_path).read_text(encoding="utf-8")
        raise DocumentLoadError(f"No textual content available for {self.document_id}")


def determine_input_format(input_path: Path, requested_format: Optional[str]) -> str:
    if requested_format:
        return requested_format

    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix == ".jsonl":
            return "jsonl"
        if suffix == ".parquet":
            return "parquet"
        return "markdown"

    try:
        next(input_path.rglob("*.md"))
        return "markdown"
    except StopIteration:
        pass
    except PermissionError:
        pass

    try:
        next(input_path.rglob("*.jsonl"))
        return "jsonl"
    except StopIteration:
        pass
    except PermissionError:
        pass

    try:
        next(input_path.rglob("*.parquet"))
        return "parquet"
    except StopIteration:
        return "markdown"
    except PermissionError:
        return "markdown"


def build_markdown_documents(input_path: Path) -> List[DocumentPayload]:
    if input_path.is_file():
        md_files = [str(input_path)]
    else:
        md_files = find_markdown_files(str(input_path))

    documents: List[DocumentPayload] = []
    for file_path in md_files:
        filename = os.path.basename(file_path)
        documents.append(
            DocumentPayload(
                document_id=filename,
                checkpoint_key=file_path,
                file_path=file_path,
            )
        )
    return documents


def _resolve_column(
    schema_names: List[str],
    requested: Optional[str],
    fallback_candidates: Optional[List[str]],
    required: bool,
    role: str,
) -> Tuple[Optional[str], bool]:
    normalized = {name.lower(): name for name in schema_names}
    search_order: List[str] = []

    if requested:
        search_order.append(requested)
    if fallback_candidates:
        search_order.extend(fallback_candidates)

    for name in search_order:
        match = normalized.get(name.lower())
        if match:
            if requested and match.lower() == requested.lower():
                return match, False
            return match, True

    if required:
        available = ", ".join(schema_names)
        raise DocumentLoadError(
            f"Column '{requested}' not found in parquet schema for {role}. Available columns: {available}"
        )

    return None, False


def stream_parquet_documents(
    input_path: Path,
    text_column: Optional[str],
    id_column: Optional[str],
    batch_size: int,
    fallback_text_columns: Optional[List[str]] = None,
    fallback_id_columns: Optional[List[str]] = None,
) -> Tuple[Iterator[DocumentPayload], Optional[int]]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise DocumentLoadError(
            "pyarrow is required to process parquet inputs. Install it via `pip install pyarrow`."
        ) from exc

    dataset = ds.dataset(str(input_path), format="parquet")
    schema = dataset.schema
    schema_names = list(getattr(schema, "names", [])) or [field.name for field in schema]

    resolved_text_column, text_was_inferred = _resolve_column(
        schema_names=schema_names,
        requested=text_column,
        fallback_candidates=fallback_text_columns or [],
        required=True,
        role="text content",
    )

    resolved_id_column, id_was_inferred = _resolve_column(
        schema_names=schema_names,
        requested=id_column,
        fallback_candidates=fallback_id_columns or [],
        required=False,
        role="identifier",
    )

    if text_was_inferred:
        print(f"Auto-selected parquet text column '{resolved_text_column}'. Use --parquet-text-column to override.")
    if resolved_id_column and id_was_inferred:
        print(f"Auto-selected parquet id column '{resolved_id_column}'. Use --parquet-id-column to override.")

    try:
        total_rows = dataset.count_rows()
    except Exception:
        total_rows = None

    requested_columns = [resolved_text_column]
    if resolved_id_column and resolved_id_column != resolved_text_column:
        requested_columns.append(resolved_id_column)

    def generator() -> Iterator[DocumentPayload]:
        row_index = 0
        for batch in dataset.to_batches(columns=requested_columns, batch_size=batch_size):
            schema = batch.schema
            text_idx = schema.get_field_index(resolved_text_column)
            if text_idx == -1:
                raise DocumentLoadError(f"Column '{resolved_text_column}' not found in parquet schema")

            id_idx = schema.get_field_index(resolved_id_column) if resolved_id_column else None

            text_array = batch.column(text_idx)
            id_array = batch.column(id_idx) if id_idx is not None else None

            for i in range(batch.num_rows):
                text_value = text_array[i].as_py()
                if text_value is None:
                    row_index += 1
                    continue

                if isinstance(text_value, bytes):
                    text_value = text_value.decode("utf-8", errors="ignore")

                text_str = str(text_value).strip()
                if not text_str:
                    row_index += 1
                    continue

                if id_array is not None:
                    raw_id = id_array[i].as_py()
                    document_id = str(raw_id) if raw_id is not None else None
                else:
                    document_id = None

                if not document_id:
                    base = input_path.stem if input_path.is_file() else input_path.name
                    document_id = f"{base}-row-{row_index}"

                checkpoint_key = f"{input_path}:{document_id}"
                yield DocumentPayload(
                    document_id=document_id,
                    checkpoint_key=checkpoint_key,
                    content=text_str,
                )
                row_index += 1

    return generator(), total_rows


def stream_jsonl_documents(
    input_path: Path,
    text_column: Optional[str],
    id_column: Optional[str],
    fallback_text_columns: Optional[List[str]] = None,
    fallback_id_columns: Optional[List[str]] = None,
) -> Tuple[Iterator[DocumentPayload], Optional[int]]:
    with open(input_path, "r", encoding="utf-8") as fh:
        first_line = fh.readline()
    if not first_line.strip():
        return iter([]), 0

    first_obj = json.loads(first_line)
    field_names = list(first_obj.keys())

    resolved_text_column, text_was_inferred = _resolve_column(
        schema_names=field_names,
        requested=text_column,
        fallback_candidates=fallback_text_columns or [],
        required=True,
        role="text content",
    )

    resolved_id_column, id_was_inferred = _resolve_column(
        schema_names=field_names,
        requested=id_column,
        fallback_candidates=fallback_id_columns or [],
        required=False,
        role="identifier",
    )

    if text_was_inferred:
        print(f"Auto-selected JSONL text field '{resolved_text_column}'. Use --text-column to override.")
    if resolved_id_column and id_was_inferred:
        print(f"Auto-selected JSONL id field '{resolved_id_column}'. Use --id-column to override.")

    total_rows = 0
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                total_rows += 1

    def generator() -> Iterator[DocumentPayload]:
        row_index = 0
        with open(input_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                text_value = obj.get(resolved_text_column)
                if text_value is None:
                    row_index += 1
                    continue

                if isinstance(text_value, bytes):
                    text_value = text_value.decode("utf-8", errors="ignore")

                text_str = str(text_value).strip()
                if not text_str:
                    row_index += 1
                    continue

                if resolved_id_column is not None:
                    raw_id = obj.get(resolved_id_column)
                    document_id = str(raw_id) if raw_id is not None else None
                else:
                    document_id = None

                if not document_id:
                    base = input_path.stem if input_path.is_file() else input_path.name
                    document_id = f"{base}-row-{row_index}"

                checkpoint_key = f"{input_path}:{document_id}"
                yield DocumentPayload(
                    document_id=document_id,
                    checkpoint_key=checkpoint_key,
                    content=text_str,
                )
                row_index += 1

    return generator(), total_rows
