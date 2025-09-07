# API Example Scripts

This directory contains example scripts demonstrating how to use the COMET Funding Statement Extraction API.

## Directory Structure

```
example_files/
├── README.md                       # This file
├── sample_documents/              # Sample markdown documents
│   ├── 10.1101.2020.02.24.20027490.md
│   ├── 10.1101.2020.03.10.985721.md
│   ├── 10.1101.2020.03.19.999441.md
│   └── 10.1101.2020.03.23.002899.md
├── basic_api_example.py           # Simple synchronous API usage
├── batch_processing_example.py    # Process multiple files concurrently
└── websocket_example.py          # Real-time progress tracking with WebSocket
```

## Prerequisites

1. Start the API server (from the api directory):
   ```bash
   cd ..
   uvicorn funding_api:app --reload
   ```

2. Install dependencies:
   ```bash
   pip install requests aiohttp websockets
   ```

## Example Scripts

### 1. Basic API Example (`basic_api_example.py`)

Demonstrates basic synchronous API usage with a single file.


Usage:
```bash
# Process with default normalization
python basic_api_example.py sample_documents/10.1101.2020.02.24.20027490.md

# Process without normalization
python basic_api_example.py sample_documents/10.1101.2020.02.24.20027490.md --no-normalize

# Run with automatic file selection
python basic_api_example.py
```

### 2. Batch Processing Example (`batch_processing_example.py`)

Process multiple files concurrently with comprehensive reporting.



Usage:
```bash
# Process all files in sample_documents
python batch_processing_example.py sample_documents/

# Process specific files
python batch_processing_example.py sample_documents/*.md

# Custom output directory
python batch_processing_example.py sample_documents/ --output-dir my_reports/
```

Output:
- `batch_report_YYYYMMDD_HHMMSS.txt` - Detailed processing report
- `statements_export_YYYYMMDD_HHMMSS.csv` - All statements in CSV format

### 3. WebSocket Example (`websocket_example.py`)

Real-time progress tracking using WebSocket connection.


### Usage
```bash
# Process a single file with progress tracking
python websocket_example.py sample_documents/10.1101.2020.02.24.20027490.md

# Run with automatic file selection
python websocket_example.py
```

## Sample Documents

The `sample_documents/` directory contains some sample academic works in markdown format with various funding acknowledgment formats for testing.

## Common Patterns

### Error Handling
All examples include proper error handling:
```python
try:
    response = requests.post(url, files=files, data=data)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
```

### File Validation
Examples validate markdown files before processing:
```python
if not file_path.suffix == ".md":
    print("Error: File must be a markdown (.md) file")
    sys.exit(1)
```

### Progress Tracking
The async examples show different progress tracking methods:
- Polling: Check `/task/{task_id}` endpoint periodically
- WebSocket: Real-time updates via WebSocket connection


## Troubleshooting

1. Ensure the API server is started before running examples
2. File paths: Use relative paths from the example_files directory
3. Normalization: Enable normalization for cleaner output
4. Batch size: For large batches, adjust `max_concurrent` in batch processing

`Connection refused` error:
- Ensure the API is running on `http://localhost:8000`
- Check if the port is correct

`File not found`:
- Use paths relative to the example_files directory
- Check if sample_documents directory exists

`WebSocket connection failed`:
- Ensure uvicorn is installed with WebSocket support: `pip install uvicorn[standard]`
- Check if the API has WebSocket support enabled