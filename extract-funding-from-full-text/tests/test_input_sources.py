import sys
import types
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funding_statement_extractor.io.loaders import determine_input_format, stream_parquet_documents, stream_jsonl_documents


class FakeScalar:
    def __init__(self, value):
        self._value = value

    def as_py(self):
        return self._value


class FakeColumn:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        return FakeScalar(self._values[idx])


class FakeSchema:
    def __init__(self, names):
        self.names = names

    def get_field_index(self, name):
        if not name:
            return -1
        try:
            return self.names.index(name)
        except ValueError:
            return -1


class FakeBatch:
    def __init__(self, rows, column_names):
        self._rows = rows
        self._schema = FakeSchema(column_names)

    @property
    def schema(self):
        return self._schema

    @property
    def num_rows(self):
        return len(self._rows)

    def column(self, idx):
        column_name = self._schema.names[idx]
        return FakeColumn([row.get(column_name) for row in self._rows])


class FakeDataset:
    def __init__(self, batches):
        self._batches = batches
        column_names: List[str] = []
        for batch in batches:
            for name in batch.schema.names:
                if name not in column_names:
                    column_names.append(name)
        self.schema = FakeSchema(column_names)

    def count_rows(self):
        return sum(batch.num_rows for batch in self._batches)

    def to_batches(self, columns, batch_size):
        yield from self._batches


def install_fake_pyarrow(monkeypatch, dataset):
    fake_dataset_module = types.ModuleType("pyarrow.dataset")
    fake_dataset_module.dataset = lambda *args, **kwargs: dataset

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_pyarrow.__path__ = []
    fake_pyarrow.dataset = fake_dataset_module

    monkeypatch.setitem(sys.modules, 'pyarrow', fake_pyarrow)
    monkeypatch.setitem(sys.modules, 'pyarrow.dataset', fake_dataset_module)


def test_determine_input_format_detects_types(tmp_path):
    markdown_dir = tmp_path / "markdown_docs"
    markdown_dir.mkdir()
    (markdown_dir / "doc.md").write_text("# Title\ncontent", encoding="utf-8")
    assert determine_input_format(markdown_dir, requested_format=None) == 'markdown'

    parquet_dir = tmp_path / "parquet_docs"
    parquet_dir.mkdir()
    (parquet_dir / "chunk.parquet").write_text("placeholder", encoding="utf-8")
    assert determine_input_format(parquet_dir, requested_format=None) == 'parquet'


def test_stream_parquet_documents_skips_empty_rows(tmp_path, monkeypatch):
    rows = [
        {'markdown': 'Funding round A', 'source_id': 'row-a'},
        {'markdown': 'Funding round B', 'source_id': None},
        {'markdown': '  ', 'source_id': 'row-c'}
    ]
    batches = [FakeBatch(rows, ['markdown', 'source_id'])]
    dataset = FakeDataset(batches)
    install_fake_pyarrow(monkeypatch, dataset)

    parquet_path = tmp_path / "sample.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    iterator, total_rows = stream_parquet_documents(
        parquet_path,
        text_column='markdown',
        id_column='source_id',
        batch_size=2
    )

    documents = list(iterator)
    assert total_rows == 3
    assert len(documents) == 2
    assert documents[0].document_id == 'row-a'
    assert documents[0].content == 'Funding round A'

    fallback_doc = documents[1]
    assert fallback_doc.document_id.startswith(f"{parquet_path.stem}-row-")
    assert fallback_doc.content == 'Funding round B'
    assert fallback_doc.checkpoint_key.endswith(fallback_doc.document_id)


def test_stream_parquet_documents_infers_text_column(tmp_path, monkeypatch):
    rows = [
        {'content': 'Funding round C', 'file_name': 'doc1.md'}
    ]
    batches = [FakeBatch(rows, ['content', 'file_name'])]
    dataset = FakeDataset(batches)
    install_fake_pyarrow(monkeypatch, dataset)

    parquet_path = tmp_path / "sample2.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")

    iterator, total_rows = stream_parquet_documents(
        parquet_path,
        text_column=None,
        id_column=None,
        batch_size=1,
        fallback_text_columns=['content'],
        fallback_id_columns=['file_name']
    )

    documents = list(iterator)
    assert total_rows == 1
    assert len(documents) == 1
    assert documents[0].document_id == 'doc1.md'
    assert documents[0].content == 'Funding round C'


import json


def test_determine_input_format_detects_jsonl(tmp_path):
    jsonl_file = tmp_path / "data.jsonl"
    jsonl_file.write_text('{"text": "hello"}\n', encoding="utf-8")
    assert determine_input_format(jsonl_file, requested_format=None) == "jsonl"

    jsonl_dir = tmp_path / "jsonl_docs"
    jsonl_dir.mkdir()
    (jsonl_dir / "data.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")
    assert determine_input_format(jsonl_dir, requested_format=None) == "jsonl"


def test_stream_jsonl_documents_basic(tmp_path):
    rows = [
        {"markdown": "Funding round A", "doi": "10.1234/a"},
        {"markdown": "Funding round B", "doi": "10.1234/b"},
    ]
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    iterator, total_rows = stream_jsonl_documents(
        jsonl_path,
        text_column="markdown",
        id_column="doi",
    )

    documents = list(iterator)
    assert total_rows == 2
    assert len(documents) == 2
    assert documents[0].document_id == "10.1234/a"
    assert documents[0].content == "Funding round A"
    assert documents[1].document_id == "10.1234/b"
    assert documents[1].content == "Funding round B"
    assert documents[0].checkpoint_key == f"{jsonl_path}:10.1234/a"


def test_stream_jsonl_documents_skips_empty_rows(tmp_path):
    rows = [
        {"markdown": "Funding round A", "doi": "10.1234/a"},
        {"markdown": None, "doi": "10.1234/b"},
        {"markdown": "", "doi": "10.1234/c"},
        {"markdown": "   ", "doi": "10.1234/d"},
        {"markdown": "Funding round E", "doi": "10.1234/e"},
    ]
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    iterator, total_rows = stream_jsonl_documents(
        jsonl_path,
        text_column="markdown",
        id_column="doi",
    )

    documents = list(iterator)
    assert total_rows == 5
    assert len(documents) == 2
    assert documents[0].document_id == "10.1234/a"
    assert documents[1].document_id == "10.1234/e"


def test_stream_jsonl_documents_infers_columns(tmp_path):
    rows = [
        {"content": "Funding round C", "file_name": "doc1.md"},
    ]
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    iterator, total_rows = stream_jsonl_documents(
        jsonl_path,
        text_column=None,
        id_column=None,
        fallback_text_columns=["content"],
        fallback_id_columns=["file_name"],
    )

    documents = list(iterator)
    assert total_rows == 1
    assert len(documents) == 1
    assert documents[0].document_id == "doc1.md"
    assert documents[0].content == "Funding round C"


def test_stream_jsonl_documents_auto_generates_ids(tmp_path):
    rows = [
        {"markdown": "Funding round A"},
        {"markdown": "Funding round B"},
    ]
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    iterator, total_rows = stream_jsonl_documents(
        jsonl_path,
        text_column="markdown",
        id_column=None,
    )

    documents = list(iterator)
    assert total_rows == 2
    assert len(documents) == 2
    assert documents[0].document_id == "sample-row-0"
    assert documents[1].document_id == "sample-row-1"
