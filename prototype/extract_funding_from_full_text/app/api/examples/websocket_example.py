#!/usr/bin/env python3
"""
WebSocket client example for real-time progress tracking.

This script demonstrates:
- Uploading a file for async processing
- Connecting to WebSocket for real-time updates
- Displaying progress with a visual progress bar
- Handling results when processing completes
"""

import asyncio
import aiohttp
import websockets
import json
import sys
from pathlib import Path
from typing import Optional
import time

# API configuration
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"


class ProgressBar:
    """Simple console progress bar."""
    
    def __init__(self, width: int = 50):
        self.width = width
        self.last_message = ""
    
    def update(self, percentage: float, message: str = ""):
        """Update the progress bar display."""
        filled = int(self.width * percentage)
        bar = "█" * filled + "░" * (self.width - filled)
        percent_str = f"{percentage * 100:5.1f}%"
        
        # Clear previous line and print new progress
        print(f"\r[{bar}] {percent_str} {message}", end="", flush=True)
        self.last_message = message
    
    def complete(self):
        """Mark progress as complete."""
        print()  # New line after progress bar


async def upload_file_async(file_path: Path) -> Optional[dict]:
    """
    Upload a file for async processing.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Task info with task_id and websocket_url
    """
    async with aiohttp.ClientSession() as session:
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=file_path.name)
            data.add_field('normalize', 'true')
            
            try:
                async with session.post(f"{API_BASE_URL}/extract-funding-async", data=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        print(f"Error uploading file: HTTP {response.status} - {error_text}")
                        return None
            except Exception as e:
                print(f"Error uploading file: {e}")
                return None


async def track_progress_websocket(task_id: str, progress_bar: ProgressBar) -> Optional[dict]:
    """
    Connect to WebSocket and track progress.
    
    Args:
        task_id: Task ID from upload response
        progress_bar: Progress bar instance
        
    Returns:
        Final result when processing completes
    """
    ws_url = f"{WS_BASE_URL}/ws/progress/{task_id}"
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"Connected to WebSocket for task {task_id}")
            
            while True:
                try:
                    # Receive progress update
                    message = await websocket.recv()
                    update = json.loads(message)
                    
                    # Update progress display
                    progress_bar.update(
                        update['percentage'],
                        update.get('message', '')
                    )
                    
                    # Check if completed
                    if update.get('completed', False):
                        progress_bar.complete()
                        
                        if update.get('error'):
                            print(f"\n❌ Error: {update['error']}")
                            return None
                        
                        return update.get('result')
                        
                except websockets.exceptions.ConnectionClosed:
                    print("\nWebSocket connection closed")
                    break
                    
    except Exception as e:
        print(f"\nWebSocket error: {e}")
        return None


def display_results(result: dict):
    """Display extraction results in a formatted way."""
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60 + "\n")
    
    # Summary
    summary = result['summary']
    metadata = result['metadata']
    
    print(f"📊 Summary:")
    print(f"   - Total statements: {summary['total_statements']}")
    print(f"   - Unique statements: {summary['unique_statements']}")
    print(f"   - Processing time: {metadata['processing_time']:.2f} seconds")
    print(f"   - Paragraphs analyzed: {metadata['num_paragraphs']}")
    
    # Statements by query
    if summary['statements_by_query']:
        print(f"\n📋 Statements by query:")
        for query, count in summary['statements_by_query'].items():
            print(f"   - {query}: {count}")
    
    # Funding statements
    statements = result['funding_statements']
    if statements:
        print(f"\n💰 Funding Statements:")
        print("-" * 60)
        for i, stmt in enumerate(statements[:10], 1):  # Show first 10
            print(f"\n{i}. {stmt['statement']}")
            print(f"   Score: {stmt['score']:.2f} | Query: {stmt['query']}")
        
        if len(statements) > 10:
            print(f"\n... and {len(statements) - 10} more statements")
    else:
        print("\n❌ No funding statements found")


async def process_file_with_progress(file_path: Path):
    """
    Process a file with real-time progress tracking.
    
    Args:
        file_path: Path to the markdown file
    """
    print(f"📁 Processing: {file_path.name}")
    print("-" * 60)
    
    # Step 1: Upload file
    print("Uploading file...")
    task_info = await upload_file_async(file_path)
    
    if not task_info:
        print("Failed to upload file")
        return
    
    task_id = task_info['task_id']
    print(f"✓ File uploaded successfully (Task ID: {task_id})")
    
    # Step 2: Track progress via WebSocket
    print("\nProcessing file...")
    progress_bar = ProgressBar()
    
    result = await track_progress_websocket(task_id, progress_bar)
    
    # Step 3: Display results
    if result:
        display_results(result)
    else:
        print("\nFailed to get results")


async def main():
    """Main function."""
    print("Funding Statement Extraction API - WebSocket Progress Example")
    print("=" * 60)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python websocket_example.py <markdown_file>")
        print("\nExample:")
        print("  python websocket_example.py sample_documents/10.1101.2020.02.24.20027490.md")
        
        # Find example files
        sample_dir = Path("sample_documents")
        if sample_dir.exists():
            md_files = list(sample_dir.glob("*.md"))
            if md_files:
                print(f"\nFound sample markdown files:")
                for f in md_files[:5]:
                    print(f"  - {f}")
            else:
                print("\nNo sample files found in sample_documents/")
        else:
            print("\nNo sample_documents directory found")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    # Validate file
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    if not file_path.suffix == ".md":
        print(f"Error: File must be a markdown (.md) file")
        sys.exit(1)
    
    # Process file
    start_time = time.time()
    await process_file_with_progress(file_path)
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    # Install required packages reminder
    try:
        import websockets
        import aiohttp
    except ImportError:
        print("This example requires additional packages:")
        print("  pip install websockets aiohttp")
        sys.exit(1)
    
    # Run async main
    asyncio.run(main())