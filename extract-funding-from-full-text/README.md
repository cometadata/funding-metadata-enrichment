# Extract Funding Statements from Full-Text

Python utility for extracting funding statements from markdown documents using semantic search / late-interaction models. Built with [pylate](https://github.com/lightonai/pylate).


## Installation

### Using uv (recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/extract-funding-from-full-text.git
cd extract-funding-from-full-text

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with uv
uv pip install -e .
```

### Using pip
```bash
# Clone the repository
git clone https://github.com/yourusername/extract-funding-from-full-text.git
cd extract-funding-from-full-text

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .
```

This will install the `extract-funding-statements` command-line tool and all dependencies.

**Note:** Make sure to activate your virtual environment before running the tool. The `extract-funding-statements` command will be available in your PATH when the virtual environment is activated.

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

Alternatively, you can run the module directly:
```bash
python -m funding_statement_extractor -i document.md -o results.json
```

## Configuration

Default configuration files ship inside the package at `funding_statement_extractor/configs/`:

### Query Configuration
`funding_statement_extractor/configs/queries/default.yaml` - Defines the semantic search queries for finding funding statements

### Pattern Configuration
`funding_statement_extractor/configs/patterns/funding_patterns.yaml` - Regular expressions for identifying paragraphs with funding details

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

## Command-Line Options

### Required Arguments
- `-i, --input` - Input markdown file, directory, or parquet dataset
- `-o, --output` - Output JSON file (default: funding_results.json)

### Configuration Options
- `-q, --queries` - Custom queries YAML file
- `--config-dir` - Custom configuration directory
- `--patterns-file` - Custom funding patterns YAML

### Processing Options
- `--normalize` - Enable text normalization
- `--skip-extraction` - Skip semantic extraction (reuse existing results file)
- `--enable-pattern-rescue` - Catch likely funding paragraphs that fall outside the top-k
- `--enable-post-filter` - Filter lower-confidence statements using regex-based scoring
- `--batch-size` - Documents per batch for checkpointing (default: 10)
- `--workers` - Number of parallel worker processes (auto-detected by default)

### ColBERT Options
- `--colbert-model` - ColBERT model (default: [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1))
- `--threshold` - Minimum relevance score (default: 10.0)
- `--top-k` - Top paragraphs per query (default: 5)

### Input Source Options
- `--input-format` - Force auto-detection between markdown and parquet inputs
- `--parquet-text-column` - Column containing markdown text (default: auto-detect, preferring `markdown`)
- `--parquet-id-column` - Optional identifier column for parquet rows (auto-detects common names)
- `--parquet-batch-size` - Batch size when streaming parquet rows (default: 64)

### Checkpoint Options
- `--checkpoint-file` - Custom checkpoint file path
- `--resume` - Resume from previous checkpoint
- `--force` - Force reprocessing all files

### Other Options
- `-v, --verbose` - Enable verbose output

## Output Format

The tool generates a JSON file with the following structure:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "parameters": {
    "input_path": "/path/to/documents",
    "input_format": "markdown",
    "normalize": true,
    "threshold": 28.0,
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
- Checkpoints are saved after each batch
- Use `--resume` to continue from last checkpoint
- Use `--force` to reprocess all files

### Text Normalization
The `--normalize` flag attempts to normalize the extracted funding statement by:
- Correcting malformed accents and special characters
- Removing line numbers
- Cleaning whitespace issues
- Detecting and flagging funding statements that appear to contain errant content (tables, etc.)

## Notes on Input and Markdown Conversion

This tool processes markdown files as input. When converting PDF documents to markdown, the quality and structure of the output can vary significantly depending on the conversion library used and the source document format. Some options:

### Markdown Conversion Libraries

- [pdfplumber](https://github.com/jsvine/pdfplumber) - Good for basic text extract from PDFs with decent support for complex layouts and tables.
  ```python
  import pdfplumber
  with pdfplumber.open("document.pdf") as pdf:
      text = '\n'.join(page.extract_text() for page in pdf.pages)
  ```

- [Docling](https://github.com/DS4SD/docling) - Advanced document conversion with robust support for complex layouts, tables, and figures. Works great for scientific papers.
  ```python
  from docling.document_converter import DocumentConverter
  converter = DocumentConverter()
  result = converter.convert("document.pdf")  # supports local path or URL
  markdown = result.document.export_to_markdown()
  ```

- [DOTS OCR](https://github.com/rednote-hilab/dots.ocr) - Vision-language model based conversion using multimodal LLMs. Exciting new approach for challenging documents!
  ```python
  from transformers import AutoModelForCausalLM, AutoProcessor
  model = AutoModelForCausalLM.from_pretrained("DotsOCR", trust_remote_code=True)
  processor = AutoProcessor.from_pretrained("DotsOCR", trust_remote_code=True)
  # Process images with model to extract text/layout
  ```
