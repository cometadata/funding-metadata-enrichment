# Convert and Normalize Ranking Extraction JSON

Post-processes outputs from `extract_funding_w_reranker.py` to create clean JSON with DOI-funding statement mappings.

## Usage

```bash
python convert_normalize_ranking_extraction_json.py -i input.json -d dois.csv [-o output.json] [-n] [-e]
```

## Options

- `-i, --input`: Input JSON file from extract_funding_w_reranker.py
- `-d, --dois`: CSV file containing DOIs for validation
- `-o, --output`: Output filename (default: input_converted.json)
- `-n, --normalize`: Clean funding statements (fix whitespace, accents, markdown escaping, remove line numbers)
- `-e, --exclude-problematic`: Exclude entries with table/formatting issues from main output

## Outputs

- `*_converted.json`: Clean funding statements mapped to DOIs
- `*_problematic.json`: Entries with formatting issues (tables, excessive pipes)