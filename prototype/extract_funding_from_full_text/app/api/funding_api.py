import io
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings, load_queries
from funding_processor import FundingExtractor
from funding_normalizer import extract_unique_funding_statements, normalize_funding_statement
from progress_tracking import progress_tracker, TaskProgress


class FundingStatement(BaseModel):
    statement: str = Field(..., description="The extracted funding statement")
    score: float = Field(..., description="Relevance score from the model")
    query: str = Field(..., description="Query that matched this statement")
    normalized: bool = Field(False, description="Whether the statement has been normalized")
    full_paragraph: Optional[str] = Field(None, description="Full paragraph containing the statement")


class ProcessingMetadata(BaseModel):
    num_paragraphs: int = Field(..., description="Total number of paragraphs in document")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    processed_at: str = Field(..., description="ISO timestamp of processing")


class FundingSummary(BaseModel):
    total_statements: int = Field(..., description="Total funding statements found")
    unique_statements: int = Field(..., description="Number of unique statements")
    statements_by_query: Dict[str, int] = Field(..., description="Count of statements per query")


class ExtractionResponse(BaseModel):
    funding_statements: List[FundingStatement] = Field(..., description="List of extracted funding statements")
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    summary: FundingSummary = Field(..., description="Summary statistics")
    problematic_statements: Optional[List[str]] = Field(None, description="Statements with formatting issues")


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    queries_loaded: bool
    timestamp: str


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and queries
funding_extractor: Optional[FundingExtractor] = None
queries: Optional[Dict[str, str]] = None


@app.on_event("startup")
async def startup_event():
    global funding_extractor, queries
    
    try:
        funding_extractor = FundingExtractor(load_async=True)
        
        queries = load_queries()
        
        await progress_tracker.start()
        
        print(f"API started successfully")
        print(f"Loaded {len(queries)} queries")
        print(f"Model will load asynchronously on first request")
        
        asyncio.create_task(funding_extractor.initialize_model_async())
        
    except Exception as e:
        print(f"Failed to initialize API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    await progress_tracker.stop()


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        model_loaded=funding_extractor is not None and funding_extractor.model_loaded,
        queries_loaded=queries is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/extract-funding", response_model=ExtractionResponse)
async def extract_funding(
    file: UploadFile = File(..., description="Markdown file to process"),
    normalize: Optional[bool] = Form(True, description="Normalize funding statements"),
    exclude_problematic: Optional[bool] = Form(False, description="Exclude problematic statements"),
    top_k: Optional[int] = Form(None, description="Top paragraphs to analyze per query"),
    threshold: Optional[float] = Form(None, description="Minimum score threshold")
):
    """
    Extract funding statements from a markdown file.
    
    - file: Markdown file containing the document text
    - normalize: Apply normalization to remove line numbers and fix formatting
    - exclude_problematic: Exclude statements with formatting issues
    - top_k: Number of top paragraphs to analyze (default: 5)
    - threshold: Minimum relevance score (default: 28.0)
    """
    if not file.filename.endswith('.md'):
        raise HTTPException(status_code=400, detail="File must be a markdown (.md) file")
    
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        results = funding_extractor.process_document(
            content=text,
            queries=queries,
            top_k=top_k,
            threshold=threshold
        )
        
        clean_statements, problematic_statements = extract_unique_funding_statements(
            results['funding_statements'],
            normalize=normalize,
            exclude_problematic=exclude_problematic
        )
        
        funding_statements = []
        
        for statement in clean_statements:
            for record in results['funding_statements']:
                normalized_record = normalize_funding_statement(record['full_paragraph']) if normalize else record['full_paragraph']
                if normalized_record == statement or record['full_paragraph'] == statement:
                    funding_statements.append(FundingStatement(
                        statement=statement,
                        score=record['score'],
                        query=record['query'],
                        normalized=normalize,
                        full_paragraph=record['full_paragraph'] if not normalize else None
                    ))
                    break

        if not exclude_problematic and problematic_statements:
            for statement in problematic_statements:
                for record in results['funding_statements']:
                    if record['full_paragraph'] == statement:
                        funding_statements.append(FundingStatement(
                            statement=statement,
                            score=record['score'],
                            query=record['query'],
                            normalized=False,
                            full_paragraph=record['full_paragraph']
                        ))
                        break
        
        statements_by_query = {}
        for stmt in funding_statements:
            statements_by_query[stmt.query] = statements_by_query.get(stmt.query, 0) + 1
        
        summary = FundingSummary(
            total_statements=len(funding_statements),
            unique_statements=len(set(stmt.statement for stmt in funding_statements)),
            statements_by_query=statements_by_query
        )
        
        metadata = ProcessingMetadata(
            num_paragraphs=results['num_paragraphs'],
            processing_time=results['processing_time'],
            processed_at=results['processed_at']
        )
        
        return ExtractionResponse(
            funding_statements=funding_statements,
            metadata=metadata,
            summary=summary,
            problematic_statements=problematic_statements if problematic_statements else None
        )
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding error. Please ensure the file is UTF-8 encoded.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/extract-funding-text", response_model=ExtractionResponse)
async def extract_funding_from_text(
    text: str = Form(..., description="Markdown text to process"),
    normalize: bool = Form(True, description="Normalize funding statements"),
    exclude_problematic: bool = Form(False, description="Exclude problematic statements"),
    top_k: Optional[int] = Form(None, description="Top paragraphs to analyze per query"),
    threshold: Optional[float] = Form(None, description="Minimum score threshold")
):
    """
    Extract funding statements from markdown text directly.
    
    This endpoint accepts text input instead of a file upload.
    """
    try:
        results = funding_extractor.process_document(
            content=text,
            queries=queries,
            top_k=top_k,
            threshold=threshold
        )
        
        clean_statements, problematic_statements = extract_unique_funding_statements(
            results['funding_statements'],
            normalize=normalize,
            exclude_problematic=exclude_problematic
        )
        
        funding_statements = []
        
        for statement in clean_statements:
            for record in results['funding_statements']:
                normalized_record = normalize_funding_statement(record['full_paragraph']) if normalize else record['full_paragraph']
                if normalized_record == statement or record['full_paragraph'] == statement:
                    funding_statements.append(FundingStatement(
                        statement=statement,
                        score=record['score'],
                        query=record['query'],
                        normalized=normalize,
                        full_paragraph=record['full_paragraph'] if not normalize else None
                    ))
                    break
        
        if not exclude_problematic and problematic_statements:
            for statement in problematic_statements:
                for record in results['funding_statements']:
                    if record['full_paragraph'] == statement:
                        funding_statements.append(FundingStatement(
                            statement=statement,
                            score=record['score'],
                            query=record['query'],
                            normalized=False,
                            full_paragraph=record['full_paragraph']
                        ))
                        break
        
        statements_by_query = {}
        for stmt in funding_statements:
            statements_by_query[stmt.query] = statements_by_query.get(stmt.query, 0) + 1
        
        summary = FundingSummary(
            total_statements=len(funding_statements),
            unique_statements=len(set(stmt.statement for stmt in funding_statements)),
            statements_by_query=statements_by_query
        )
        
        metadata = ProcessingMetadata(
            num_paragraphs=results['num_paragraphs'],
            processing_time=results['processing_time'],
            processed_at=results['processed_at']
        )
        
        return ExtractionResponse(
            funding_statements=funding_statements,
            metadata=metadata,
            summary=summary,
            problematic_statements=problematic_statements if problematic_statements else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/queries")
async def get_queries():
    return {"queries": queries}


class AsyncExtractionRequest(BaseModel):
    normalize: bool = True
    exclude_problematic: bool = False
    top_k: Optional[int] = None
    threshold: Optional[float] = None


class AsyncExtractionResponse(BaseModel):
    task_id: str
    status: str = "processing"
    message: str = "Extraction started"
    websocket_url: str
    polling_url: str


@app.post("/extract-funding-async", response_model=AsyncExtractionResponse)
async def extract_funding_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Markdown file to process"),
    normalize: bool = Form(True),
    exclude_problematic: bool = Form(False),
    top_k: Optional[int] = Form(None),
    threshold: Optional[float] = Form(None)
):
    """
    Extract funding statements asynchronously with progress tracking.
    
    Returns a task_id that can be used to track progress via WebSocket or polling.
    """
    if not file.filename.endswith('.md'):
        raise HTTPException(status_code=400, detail="File must be a markdown (.md) file")
    
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        task_id = progress_tracker.create_task()
        
        background_tasks.add_task(
            _process_document_async,
            task_id,
            text,
            normalize,
            exclude_problematic,
            top_k,
            threshold
        )
        
        return AsyncExtractionResponse(
            task_id=task_id,
            websocket_url=f"/ws/progress/{task_id}",
            polling_url=f"/task/{task_id}"
        )
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding error. Please ensure the file is UTF-8 encoded.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


async def _process_document_async(
    task_id: str,
    text: str,
    normalize: bool,
    exclude_problematic: bool,
    top_k: Optional[int],
    threshold: Optional[float]
):
    """Background task to process document with progress tracking."""
    try:
        async def update_progress(message: str, percentage: float):
            await progress_tracker.update_progress(task_id, "processing", percentage, message)
        
        results = await funding_extractor.process_document_async(
            content=text,
            queries=queries,
            top_k=top_k,
            threshold=threshold,
            progress_callback=update_progress
        )
        
        await progress_tracker.update_progress(task_id, "processing", 0.9, "Normalizing statements")
        
        clean_statements, problematic_statements = extract_unique_funding_statements(
            results['funding_statements'],
            normalize=normalize,
            exclude_problematic=exclude_problematic
        )
        
        funding_statements = []
        
        for statement in clean_statements:
            for record in results['funding_statements']:
                normalized_record = normalize_funding_statement(record['full_paragraph']) if normalize else record['full_paragraph']
                if normalized_record == statement or record['full_paragraph'] == statement:
                    funding_statements.append({
                        'statement': statement,
                        'score': record['score'],
                        'query': record['query'],
                        'normalized': normalize,
                        'full_paragraph': record['full_paragraph'] if not normalize else None
                    })
                    break
        
        if not exclude_problematic and problematic_statements:
            for statement in problematic_statements:
                for record in results['funding_statements']:
                    if record['full_paragraph'] == statement:
                        funding_statements.append({
                            'statement': statement,
                            'score': record['score'],
                            'query': record['query'],
                            'normalized': False,
                            'full_paragraph': record['full_paragraph']
                        })
                        break
        
        statements_by_query = {}
        for stmt in funding_statements:
            statements_by_query[stmt['query']] = statements_by_query.get(stmt['query'], 0) + 1
        
        final_result = {
            'funding_statements': funding_statements,
            'metadata': {
                'num_paragraphs': results['num_paragraphs'],
                'processing_time': results['processing_time'],
                'processed_at': results['processed_at']
            },
            'summary': {
                'total_statements': len(funding_statements),
                'unique_statements': len(set(stmt['statement'] for stmt in funding_statements)),
                'statements_by_query': statements_by_query
            },
            'problematic_statements': problematic_statements if problematic_statements else None
        }
        
        await progress_tracker.complete_task(task_id, result=final_result)
        
    except Exception as e:
        await progress_tracker.complete_task(task_id, error=str(e))


@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await progress_tracker.connect_websocket(task_id, websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await progress_tracker.disconnect_websocket(task_id, websocket)


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get the current status of an async task (polling endpoint)."""
    task = progress_tracker.get_task_progress(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task.task_id,
        "status": task.status,
        "percentage": task.percentage,
        "message": task.message,
        "completed": task.completed,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "result": task.result,
        "error": task.error
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)