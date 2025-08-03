# Funding Extractor

Extract funding information from funding statements using [LangExtract](https://github.com/google/langextract).


## Installation

```bash
uv pip install -e ".[dev,test]"
```

## Supported Providers

The funding extractor supports multiple LLM providers:

| Provider | Description | API Key Required | Default Model | Notes |
|----------|-------------|------------------|---------------|-------|
| `gemini` | Google Gemini models | Yes (GEMINI_API_KEY) | gemini-2.5-flash-lite | Default provider |
| `openai` | OpenAI GPT models | Yes (OPENAI_API_KEY) | gpt-4o-mini | Official OpenAI API |
| `ollama` | Local Ollama models | No | Varies | Runs locally |
| `openai` + custom URL | OpenAI-compatible APIs | Varies | Varies | LM Studio, vLLM, etc. |

## Usage

### CLI

```bash
# Process a JSON file with funding statements (default: Gemini)
funding-extractor input.json -o output.json

# Use a different model
funding-extractor input.json --model gemini-2.5-pro

# With API key
funding-extractor input.json --api-key YOUR_API_KEY

# Or set environment variable
export GEMINI_API_KEY=YOUR_API_KEY
funding-extractor input.json

# Incremental processing with batch saves
funding-extractor input.json -o output.json --batch-size 5

# Resume processing (skips already processed DOIs)
funding-extractor input.json -o output.json  # Will load existing results

# Verbose mode for progress tracking
funding-extractor input.json -o output.json --verbose

# Skip model validation (useful for custom models)
funding-extractor input.json --skip-model-validation

# Set custom timeout (default: 60 seconds)
funding-extractor input.json --timeout 120  # 2 minutes

# Combine options for robust processing
funding-extractor input.json -o output.json --timeout 90 --batch-size 5 --verbose
```

### Using Local LLMs with Ollama

Funding extractor supports local LLM inference using Ollama via LangExtract's Ollama interfaces. This allows you to run extraction entirely on your own hardware without sending data to external APIs.

#### Setup Ollama

1. Install Ollama from https://ollama.ai
2. Pull a model (e.g., Qwen 3 modles):
```bash
ollama pull qwen3:8b
```

#### Run Extraction with Ollama

```bash
# Use Ollama with default settings
funding-extractor input.json --provider ollama -o output.json

# Use a specific Ollama model
funding-extractor input.json --provider ollama --model qwen3:8b

# Use Ollama on a different host
funding-extractor input.json --provider ollama --model-url http://localhost:11434

# Or set environment variable for Ollama host
export OLLAMA_HOST=http://localhost:11434
funding-extractor input.json --provider ollama
```

### Using OpenAI Models

```bash
# Use OpenAI with default model (gpt-4o-mini)
funding-extractor input.json --provider openai -o output.json

# Use a specific OpenAI model
funding-extractor input.json --provider openai --model gpt-4o

# With API key (or use OPENAI_API_KEY environment variable)
funding-extractor input.json --provider openai --api-key YOUR_OPENAI_KEY
```

### Using OpenAI-Compatible Endpoints

You can use any OpenAI-compatible API endpoint (e.g., local models via [llama.cpp](https://github.com/ggml-org/llama.cpp), models serve through [OpenRouter](https://openrouter.ai/), or other services).

```bash
# Use a custom OpenAI-compatible endpoint
funding-extractor input.json --provider openai --model qwen3:8b \
  --base-url http://192.168.1.206:1234 --api-key dummy

# Alternative: use --model-url instead of --base-url
funding-extractor input.json --provider openai --model mixtral:8x7b \
  --model-url http://localhost:8000 --api-key dummy

# For endpoints that don't validate model names, you can use any model
funding-extractor input.json --provider openai --model custom-model-7b \
  --base-url http://your-server:8080 --api-key your-key
```

**Notes for OpenAI-Compatible Endpoints**:
- Model validation is automatically skipped for custom endpoints (non-api.openai.com URLs)
- Many local endpoints don't require a real API key - you can use "dummy" or any placeholder
- The `--base-url` and `--model-url` options are aliases and work the same way
- Response format compatibility may vary between different OpenAI-compatible services, so monitor for any errors


## Input Format

The input JSON should have this structure:

```json
[
  {
    "doi": "10.1234/example",
    "funding_statements": [
      "Supported by the National Science Foundation grant 1234567"
    ]
  }
]
```

## Output Format

The output JSON will contain extracted entities with support for multiple grants per funder:

```json
[
  {
    "doi": "10.1234/example",
    "funding_statement": "Supported by NIH grants R01 HL092774 and T32 AT007180",
    "entities": [
      {
        "funder": "NIH",
        "grants": [
          {
            "grant_id": "HL092774",
            "program": "R01"
          },
          {
            "grant_id": "AT007180",
            "program": "T32"
          }
        ],
        "programs": [],
        "extraction_texts": ["NIH", "HL092774", "AT007180"]
      }
    ]
  }
]
```