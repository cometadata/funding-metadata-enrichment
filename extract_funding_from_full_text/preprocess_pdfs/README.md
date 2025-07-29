# Preprocess PDFs

Batch PDF preprocessor that removes line numbers and headers from works before text extraction.

## Usage

```bash
python preprocess_pdfs.py -r line-numbers headers -i input_dir -o output_dir
```

## Features

- Removes marginal line numbers from PDF pages
- Detects and removes repeating header text across pages
- Parallel processing with configurable workers

## Options

- `-r, --remove`: Elements to remove (line-numbers, headers, or both) [required]
- `-i, --input`: Input directory (default: pdfs)
- `-o, --output`: Output directory (default: output)
- `-w, --workers`: Number of parallel processes (default: CPU cores - 1)
- `-c, --copy`: Copy unmodified PDFs to output
- `-f, --force`: Reprocess existing files
- `--header-text-threshold`: Text similarity threshold for header detection (0.0-1.0, default: 0.8)

## Output

Processed files are saved with `_preprocessed.pdf` suffix in the output directory.