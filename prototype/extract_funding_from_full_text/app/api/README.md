# Funding Statement Extraction API

FastAPI service that uses ColBERT semantic search to extract funding acknowledgements from markdown files.


## Installation
   ```bash
   pip install -r requirements.txt
   ```


## Start the API server:
   ```bash
   uvicorn funding_api:app --reload
   ```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, API documentation is availableat:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints

#### Health Check
```http
GET /
```
Returns server status and model loading state.

#### Synchronous Extraction
```http
POST /extract-funding
Content-Type: multipart/form-data

file: <markdown-file>
normalize: true
exclude_problematic: false
top_k: 5
threshold: 28.0
```

#### Asynchronous Extraction with Progress
```http
POST /extract-funding-async
Content-Type: multipart/form-data

file: <markdown-file>
normalize: true
```

Returns:
```json
{
  "task_id": "uuid",
  "status": "processing",
  "websocket_url": "/ws/progress/{task_id}",
  "polling_url": "/task/{task_id}"
}
```

#### WebSocket Progress
```
WS /ws/progress/{task_id}
```

Receive real-time progress updates:
```json
{
  "task_id": "uuid",
  "status": "processing",
  "percentage": 0.5,
  "message": "Processing query: grant_numbers",
  "completed": false
}
```

#### Task Status (Polling)
```http
GET /task/{task_id}
```

### Response Format

Successful extraction returns:
```json
{
  "funding_statements": [
    {
      "statement": "This work was supported by NSF grant 1234567.",
      "score": 45.2,
      "query": "grant_numbers",
      "normalized": true
    }
  ],
  "metadata": {
    "num_paragraphs": 150,
    "processing_time": 2.34,
    "processed_at": "2024-01-01T12:00:00"
  },
  "summary": {
    "total_statements": 5,
    "unique_statements": 3,
    "statements_by_query": {
      "grant_numbers": 2,
      "funding_acknowledgment": 3
    }
  },
  "problematic_statements": []
}
```

## Configuration

### Environment Variables

Configure the API using environment variables with the prefix `FUNDING_API_`:

```bash
# Model configuration
FUNDING_API_MODEL_NAME="lightonai/GTE-ModernColBERT-v1"

# Processing parameters
FUNDING_API_TOP_K=5
FUNDING_API_THRESHOLD=28.0
FUNDING_API_BATCH_SIZE=32

# Statement normalization
FUNDING_API_NORMALIZE_STATEMENTS=true
FUNDING_API_MIN_STATEMENT_LENGTH=20

# API metadata
FUNDING_API_TITLE="Funding Statement Extraction API"
FUNDING_API_VERSION="1.0.0"
```

### Funding Queries

Customize search queries by editing `funding_queries.yaml`:

```yaml
grant_numbers:
  query: "grant number award number funding code"
  description: "Searches for grant and award numbers"

funding_acknowledgment:
  query: "supported by funded by acknowledged"
  description: "General funding acknowledgments"
```

## Advanced Usage

### Custom Model Selection

Use a different ColBERT model:
```python
export FUNDING_API_MODEL_NAME="your-org/your-colbert-model"
```

### Batch Processing

Process multiple documents:
```python
import asyncio
import aiohttp

async def process_documents(file_paths):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for path in file_paths:
            with open(path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=path)
                data.add_field('normalize', 'true')
                
                task = session.post(
                    'http://localhost:8000/extract-funding-async',
                    data=data
                )
                tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]
```

### WebSocket Client Example

```python
import asyncio
import websockets
import json

async def track_progress(task_id):
    uri = f"ws://localhost:8000/ws/progress/{task_id}"
    
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            update = json.loads(message)
            
            print(f"Progress: {update['percentage']*100:.1f}% - {update['message']}")
            
            if update['completed']:
                return update['result']
```

## Architecture

### Components

1. `funding_api.py`
   - FastAPI application and endpoint definitions
   - Request/response models
   - WebSocket and async task management

2. `funding_processor.py`
   - ColBERT model initialization and management
   - Document processing pipeline
   - Semantic search implementation

3. `funding_normalizer.py`
   - Text normalization and cleaning
   - Deduplication logic
   - Statement validation

4. `progress_tracking.py`
   - Real-time progress management
   - WebSocket connection handling
   - Task state persistence

5. `config.py`
   - Environment variable management
   - Query loading from YAML
   - Regex pattern definitions


## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [ColBERT](https://github.com/stanford-futuredata/ColBERT) semantic search
- Uses [PyLate](https://github.com/lightonai/pylate) for model implementation

---
