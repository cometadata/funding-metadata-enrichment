# Funding Statement Extraction with Reranker

Tool for extracting funding acknowledgements from documents using semantic search and pattern matching.

## How It Works

The extraction process occurs in two stages:

1. A ColBERT model (`lightonai/GTE-ModernColBERT-v1`) compares each paragraph in a document against a set of funding-related queries, scoring text sections by semantic relevance and returns the top candidates.

2. The highest-scoring text sections then are processed with a regex pattern to confirm they contain language related to funding (e.g., "funded by", "grant number", "supported by").

A section is deemed a funding statement only if it passes both stages - achieving a minimum semantic score and matching at least one funding pattern.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python extract_funding_w_reranker.py -d /path/to/markdown/files -q queries.yaml -o results.json
```

### Arguments

- `-d, --directory`: Directory containing markdown files to search
- `-q, --query-file`: YAML file with search queries
- `-o, --output`: Output JSON file (default: funding_extractions.json)
- `--resume`: Resume from previous checkpoint
- `-k, --top-k`: Top paragraphs to analyze per query (default: 5)
- `-t, --threshold`: Minimum semantic score (default: 28.0)
- `-w, --workers`: Number of parallel workers (auto-detected by default)

### Query File Format

```yaml
queries:
  funding_statement: "funding statement acknowledgement grant support"
  acknowledgements: "acknowledgements section funding support"
```

## Output

As output, we return a JSON file containing:
- Document-level results with extracted funding statements
- Semantic scores and matched queries for each statement