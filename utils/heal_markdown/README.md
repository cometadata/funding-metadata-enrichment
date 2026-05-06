# Heal Markdown

Script for healing markdown files from PDF conversions to resemble something like a more human-readable format. 


## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Process parquet file to parquet output
```bash
python heal_markdown.py input.parquet --out-dir output --output-format parquet
```

### Process parquet to markdown files
```bash
python heal_markdown.py input.parquet --out-dir output --output-format markdown
```

### Process parquet to both formats
```bash
python heal_markdown.py input.parquet --out-dir output --output-format both
```

### Process individual markdown files (backward compatible)
```bash
python heal_markdown.py document.md --out-dir output
python heal_markdown.py directory/ --out-dir output
```

## Column Auto-Detection

For parquet input, automatically detects columns:
- **Filename**: `file_name`, `filename`, `relative_path`, `name`, `path` (fallback: row index)
- **Content**: `content`, `text`, `markdown`, `md`, `body` (required)

## Output Files

### Cleaned Parquet Schema
- `file_name`: Document identifier
- `content`: Cleaned markdown
- `warnings`: Processing warnings
- `success`: Success flag
- `original_size`, `output_size`: Size metrics

### Failed Parquet Schema
- `file_name`: Document identifier
- `original_content`: Original text
- `cleaned_content`: Best-effort cleaned text
- `failure_category`: `unrecoverable`, `partial`, `needs_review`, etc.
- `issues`: Detailed issue descriptions
- `recovery_attempted`, `recovery_confidence`: Recovery metadata
- `recoverable`: Manual recovery feasibility
- `validation_error`: Original error message

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-format` | `markdown`, `parquet`, or `both` | Same as input |
| `--out-dir` | Output directory | Required for parquet input |
| `--failed-output` | Failed records parquet path | `{out-dir}/failed.parquet` |
| `--in-place` | Overwrite original files | False |
| `--page-lines` | Lines per page for header detection | 50 |
| `--repeat-threshold` | Frequency threshold for repeated lines | 0.8 |
| `--skip-tables` | Disable table reconstruction | False |
| `--strip-citations` | Remove `[12]` style citations | False |
| `--exclude-failed` | Skip empty conversions | False |


**Output:**
- `cleaned_output/test_sample_cleaned.parquet` - Successfully cleaned documents
- `cleaned_output/failed.parquet` - Documents with issues requiring review
