# Funding Metadata Extraction Application

This application is a web-based version of the funding extraction pipeline described in the parent directory. It performs the same extraction process but:

- Expects markdown files instead of PDFs (no conversion step)
- Provides a demo REST API web UI instead of command-line tools

## Components

- API (`/api`): FastAPI service for funding extraction with progress tracking
- UI (`/ui`): Svelte web interface for uploading documents and viewing results

For PDF processing and batch operations, use the pipeline tools in the parent directory or one of myriad other conversion libraries.