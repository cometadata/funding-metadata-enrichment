# Extract Funding Statements from Full-Text

Python utility for extracting funding statements from markdown documents using semantic search / late-interaction models. Built with [pylate](https://github.com/lightonai/pylate).


## Installation

### Using uv (recommended)
```bash
git clone https://github.com/yourusername/extract-funding-from-full-text.git
cd extract-funding-from-full-text

uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

uv pip install -e .
```

### Using pip
```bash
git clone https://github.com/yourusername/extract-funding-from-full-text.git
cd extract-funding-from-full-text

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e .
```

This installs the `extract-funding-statements` command-line tool and all dependencies. Activate the virtual environment before running the tool.

## Quick Start

```bash
# Process a single markdown file with default settings
extract-funding-statements -i document.md -o results.json

# Process a directory of markdown files
extract-funding-statements -i /path/to/documents -o results.json

# Stream a directory of parquet chunks (text column auto-detects, prefers "markdown")
extract-funding-statements -i /path/to/parquet-chunks --input-format parquet \
  --parquet-text-column markdown --parquet-id-column source_id -o results.json
```

You can also run the module directly:
```bash
python -m funding_statement_extractor -i document.md -o results.json
```

## Configuration

Default configuration files ship inside the package at `funding_statement_extractor/configs/`:

### Query Configuration
`funding_statement_extractor/configs/queries/default.yaml` defines the semantic search queries for finding funding statements.

### Pattern Configuration
`funding_statement_extractor/configs/patterns/funding_patterns.yaml` provides regular expressions for identifying paragraphs with funding details.

Custom queries and patterns can be supplied via command-line arguments.

## Usage Examples

### Basic Extraction
```bash
extract-funding-statements -i paper.md -o funding.json
```

### With Text Normalization
```bash
extract-funding-statements -i docs/ -o results.json --normalize
```

### With Pattern-Based Rescue and Post-Filtering
```bash
extract-funding-statements -i docs/ -o results.json \
  --enable-pattern-rescue --enable-post-filter
```

### With Paragraph Pre-filtering (faster)
```bash
extract-funding-statements -i docs/ -o results.json \
  --enable-paragraph-prefilter
```

## Command-Line Options

### Required Arguments
- `-i, --input` Input markdown file, directory, or parquet dataset
- `-o, --output` Output JSON file (default: funding_results.json)

### Configuration Options
- `-q, --queries` Custom queries YAML file
- `--config-dir` Custom configuration directory
- `--patterns-file` Custom funding patterns YAML

### Processing Options
- `--normalize` Enable text normalization
- `--skip-extraction` Skip semantic extraction (reuse existing results file)
- `--enable-pattern-rescue` Catch likely funding paragraphs that fall outside the top-k
- `--enable-post-filter` Filter lower-confidence statements using regex-based scoring
- `--enable-paragraph-prefilter` Pre-filter paragraphs by funding keywords before encoding (~6-9x speedup, may affect recall)
- `--batch-size` Documents between fsync/checkpoint flushes (default: 50)
- `--workers` Number of parallel worker processes (auto-detected by default)

### ColBERT Options
- `--colbert-model` ColBERT model (default: [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1))
- `--threshold` Minimum relevance score (default: 10.0)
- `--top-k` Top paragraphs per query (default: 5)

### Batch Engine Options
The default extraction path is a streaming batch engine that overlaps I/O, paragraph preparation, and GPU encoding.

- `--legacy-engine` Use the per-document ProcessPoolExecutor loop instead of the batch engine
- `--paragraphs-per-batch` Pipeline batch size in paragraphs for the GPU consumer (default: 4096)
- `--encode-batch-size` Sub-batch size passed to model.encode (default: 512)
- `--queue-depth` Inter-stage queue depth in the batch engine (default: 128)
- `--dtype` Model dtype for the batch engine: auto, fp32, fp16, bf16 (default: auto)

### Input Source Options
- `--input-format` Force selection between markdown and parquet inputs
- `--parquet-text-column` Column containing markdown text (default: auto-detect, preferring `markdown`)
- `--parquet-id-column` Optional identifier column for parquet rows (auto-detects common names)
- `--parquet-batch-size` Batch size when streaming parquet rows (default: 64)

### Checkpoint Options
- `--checkpoint-file` Custom checkpoint file path
- `--resume` Resume from previous checkpoint
- `--force` Force reprocessing all files
- `--retry-failed` Re-process documents marked failed in the checkpoint

### Other Options
- `-v, --verbose` Enable verbose output

## Output Format

The tool generates a JSON file with the following structure:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "parameters": {
    "input_path": "/path/to/documents",
    "input_format": "markdown",
    "normalize": true,
    "threshold": 10.0,
    "top_k": 5
  },
  "results": {
    "document.md": {
      "funding_statements": [
        {
          "statement": "This work was supported by NSF grant 12345 and NIH grant 67890.",
          "original": "Original text before normalization",
          "score": 35.2,
          "query": "funding_statement",
          "is_problematic": false
        }
      ]
    }
  },
  "summary": {
    "total_files": 10,
    "files_with_funding": 8,
    "total_statements": 25
  }
}
```

## Advanced Features

### Checkpoint and Resume
Progress is automatically saved for large batch operations:
- Checkpoints are saved every `--batch-size` document completions
- Use `--resume` to continue from the last checkpoint
- Use `--force` to reprocess all files
- Use `--retry-failed` to re-run documents that previously failed

### Text Normalization
The `--normalize` flag attempts to normalize the extracted funding statement by:
- Correcting malformed accents and special characters
- Removing line numbers
- Cleaning whitespace issues
- Detecting and flagging funding statements that appear to contain errant content (tables, etc.)

## Operational scripts

Batch extraction, orchestration, profiling, and HF Jobs runners live outside the library:

- `../hf_jobs/funding_statement_extractor/`: PEP 723 self-contained scripts run via `hf jobs uv run` (baseline benchmark, Tier-2 extraction worker, h200 profiler).
- `../ops/funding_statement_extractor/`: local helpers run with this project's venv (orchestrator that dispatches HF Jobs, dataset builder, local profiler, prediction diff tool).

Each directory has its own README and tests.

## Notes on Input and Markdown Conversion

This tool processes markdown files as input. When converting PDF documents to markdown, the quality and structure of the output can vary significantly depending on the conversion library used and the source document format. Some options:

### Markdown Conversion Libraries

- [pdfplumber](https://github.com/jsvine/pdfplumber): good for basic text extraction from PDFs with decent support for complex layouts and tables.
  ```python
  import pdfplumber
  with pdfplumber.open("document.pdf") as pdf:
      text = '\n'.join(page.extract_text() for page in pdf.pages)
  ```

- [Docling](https://github.com/DS4SD/docling): advanced document conversion with robust support for complex layouts, tables, and figures. Works well for scientific papers.
  ```python
  from docling.document_converter import DocumentConverter
  converter = DocumentConverter()
  result = converter.convert("document.pdf")  # supports local path or URL
  markdown = result.document.export_to_markdown()
  ```

- Vision-language models (VLMs): multimodal LLMs can transcribe complex layouts, tables, and figures directly from page images. This is often the most effective option for challenging or scanned documents. See [cometadata/pdf-ocr](https://github.com/cometadata/pdf-ocr) for our reference pipeline that runs PDFs through VLM-based OCR and emits markdown.
