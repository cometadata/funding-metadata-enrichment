# Convert and Normalize Ranking Extraction JSON

Post-processes outputs from `extract_funding_w_reranker.py` to create clean JSON or CSV files with DOI/filename-funding statement mappings.

## Usage

```bash
python convert_normalize_ranking_extraction_json.py -i input.json [options]
```

## Required Arguments

- `-i, --input`: Input JSON file from extract_funding_w_reranker.py

## Optional Arguments

- `-o, --output`: Output filename (default: `[input]_converted.json` or `.csv`)
- `-n, --normalize`: Clean funding statements:
  - Fix whitespace and line breaks
  - Correct accented characters (é, á, ñ, etc.)
  - Remove markdown escaping
  - Remove sequential line numbers
- `-e, --exclude-problematic`: Exclude entries with table/formatting issues from main output
- `--csv`: Output in CSV format instead of JSON
- `--use-doi`: Use DOI instead of filename in output (requires `-d` for validation)
- `-d, --dois`: CSV file containing DOIs for validation (required only with `--use-doi`)

## Output Formats

### JSON Output (default)
```json
[
  {
    "doi": "10.1234/example",
    "funding_statements": [
      "This work was supported by NSF grant 12345.",
      "Additional funding from NIH grant 67890."
    ]
  }
]
```

### CSV Output with Filenames (default with `--csv`)
```csv
filename,funding_statement
10_1234_example.md,"This work was supported by NSF grant 12345."
10_1234_example.md,"Additional funding from NIH grant 67890."
```

### Using DOIs (with `--use-doi`)
```json
[
  {
    "doi": "10.1234/example",
    "funding_statements": ["..."]
  }
]
```

## Output Files

- Main output: Clean funding statements mapped to DOIs or filenames
  - `*_converted.json` (default) or `*_converted.csv` (with `--csv`)
- Problematic output: Entries with formatting issues (always JSON)
  - `*_problematic.json` - Contains tables, excessive pipes, or other formatting problems


## Normalization

When using `-n, --normalize`, the script:

1. Converts multiple spaces/newlines to single spaces
2. Fixes separated accents (e.g., `e´` → `é`, `n~` → `ñ`)
3. Removes backslashes from `\_`, `\*`, `\[`, `\]`
4. Attempts to detect and removes sequential line numbers (e.g., "text 123 124 125" → "text")
5. Ensures consistent character encoding

## Problematic Statement Detection

The script identifies problematic funding statement entries with:
- Table structures (`|-----|` or similar)
- Excessive pipe characters (> 4 pipes)
- Multiple lines with pipes (> 3 lines)
- Numbered table cells (`| 1 |`, `| 2 |`, etc.)

These entries are saved into a separate file for manual review.

## Input File Format

JSON structure expected is that returned from `extract_funding_w_reranker.py`:
```json
{
  "results_by_document": [
    {
      "filename": "10_1234_example.md",
      "funding_statements": [
        {
          "full_paragraph": "Funding statement text here..."
        }
      ]
    }
  ]
}
```
