# Funding Statement Extractor UI

Demo web app for extracting funding acknowledgements from markdown documents using the COMET Funding Statement Extraction API.


## Prerequisites

- Node.js 18+ and npm
- The COMET Funding Statement Extraction API running on `http://localhost:8000`

## Quick Start

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the dev server:
   ```bash
   npm run dev
   ```

3. Navigate to `http://localhost:3000`


## Configuration

### API Endpoint

The application is configured to proxy API requests through Vite's dev server. To modify the API endpoint, edit `vite.config.js`:

```javascript
proxy: {
  '/api': {
    target: 'http://127.0.0.1:8000',  // Change this to your API URL
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, '')
  }
}
```

### Environment Variables

For production deployments, you can configure the API endpoint via environment variables:

```bash
VITE_API_URL=https://your-api-server.com
```

## Architecture


### Components

```
src/
├── App.svelte                    # Main application component
├── lib/
│   ├── api.js                   # API communication layer
│   ├── stores/
│   │   └── taskStore.js         # Global state management
│   └── components/
│       ├── Uploader.svelte      # File upload interface
│       ├── Progress.svelte      # Progress bar with animations
│       ├── ResultCard.svelte    # Individual funding statement display
│       ├── ResultsSummary.svelte # Processing statistics
│       └── ProblematicStatements.svelte # Formatting warnings
```

### State Management

The application uses Svelte stores for reactive state management:

```javascript
{
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error',
  file: File | null,
  taskId: string | null,
  percentage: number,
  message: string,
  result: ExtractionResult | null,
  error: string | null,
  websocket: WebSocket | null
}
```


## Usage Guide

### Basic Workflow

1. Select a File:
   - Drag and drop a `.md` file onto the upload area
   - Or click to browse and select a file

2. Configure Options:
   - Check/uncheck "Normalize statements" based on your needs

3. Start Extraction:
   - Click "Extract Statements" to begin processing

4. Monitor Progress:
   - Watch the real-time progress bar
   - See status messages as different queries are processed

5. Review Results:
   - View extracted funding statements with relevance scores
   - Check problematic statements if any were found
   - Process another file or refine your document

### Understanding Results

Each extracted statement includes:
- Statement Text: The funding acknowledgement text
- Score: Relevance score from the AI model (higher is better)
- Query: Which search query found this statement
- Normalized: Indicates if text cleanup was applied
