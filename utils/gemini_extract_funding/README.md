# Gemini Funding Extraction

Extract funding and financial support statements from markdown files using Google's Gemini API with structured outputs.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Process a single file:
```bash
python gemini_extract_funding.py path/to/file.md
```

Process a directory:
```bash
python gemini_extract_funding.py path/to/markdown/directory --output results.jsonl
```

Specify a model:
```bash
python gemini_extract_funding.py path/to/file.md --model gemini-2.5-pro
```

## Output

Results are written as JSON Lines (one JSON object per line) to stdout or the specified output file:

```json
{"found": true, "statements": ["This work was supported by NSF grant 12345."], "notes": null, "file": "path/to/file.md"}
{"found": false, "statements": [], "notes": "No funding statement found.", "file": "path/to/other.md"}
```

## Options

- `path`: Markdown file or directory containing markdown files
- `--model`: Gemini model to use (default: `gemini-2.5-flash`)
- `--api-key`: Gemini API key (overrides environment variables)
- `--output`: Output file path (defaults to stdout)
