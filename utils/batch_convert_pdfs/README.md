# Batch Convert PDFs

Batch converts PDF files to multiple formats using Docling.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python batch_convert_pdfs.py [-i INPUT_DIR] [-o OUTPUT_DIR] [options]
```

## Arguments

- `-i, --input`: Input directory (default: `pdfs`)
- `-o, --output`: Output directory (default: `output`)
- `-f, --formats`: Output formats (default: all)
- `-r, --ocr`: Enable OCR processing
- `-R, --resume`: Skip already processed files
- `-P, --pages`: Page range to process
- `--workers`: Number of parallel processes (default: CPU cores - 1)

## Examples

```bash
# Convert all PDFs to markdown
python batch_convert_pdfs.py -f markdown

# Convert with OCR enabled, specific pages
python batch_convert_pdfs.py -r -P "1-10"

# Resume previous batch with 4 workers
python batch_convert_pdfs.py -R --workers 4
```

## Output Structure

```
output/
├── document1/
│   ├── document1.md
│   ├── document1.html
│   ├── metadata/
│   │   ├── document1.json
│   │   └── document1.yaml
│   └── images/
│       └── document1_page_001.png
```