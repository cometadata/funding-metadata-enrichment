#!/usr/bin/env python3
"""
Batch processing example for the Funding Statement Extraction API.

This script demonstrates:
- Processing multiple markdown files concurrently
- Using async endpoints for better performance
- Tracking progress for multiple tasks
- Generating a summary report
"""

import asyncio
import aiohttp
from pathlib import Path
import json
import sys
from datetime import datetime
from typing import List, Dict, Any
import csv

# API configuration
API_BASE_URL = "http://localhost:8000"


async def process_file_async(session: aiohttp.ClientSession, file_path: Path) -> Dict[str, Any]:
    """
    Process a single file asynchronously.
    
    Args:
        session: aiohttp client session
        file_path: Path to the markdown file
        
    Returns:
        Dictionary with file path and results or error
    """
    try:
        # Upload file
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=file_path.name)
            data.add_field('normalize', 'true')
            
            async with session.post(f"{API_BASE_URL}/extract-funding-async", data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return {
                        'file': str(file_path),
                        'status': 'error',
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                
                task_info = await response.json()
                task_id = task_info['task_id']
        
        # Poll for results
        max_attempts = 60  # 60 seconds timeout
        for attempt in range(max_attempts):
            async with session.get(f"{API_BASE_URL}/task/{task_id}") as response:
                task_status = await response.json()
                
                if task_status['completed']:
                    if task_status.get('error'):
                        return {
                            'file': str(file_path),
                            'status': 'error',
                            'error': task_status['error']
                        }
                    
                    return {
                        'file': str(file_path),
                        'status': 'success',
                        'result': task_status['result']
                    }
            
            await asyncio.sleep(1)
        
        return {
            'file': str(file_path),
            'status': 'error',
            'error': 'Processing timeout'
        }
        
    except Exception as e:
        return {
            'file': str(file_path),
            'status': 'error',
            'error': str(e)
        }


async def process_batch(file_paths: List[Path], max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """
    Process multiple files concurrently.
    
    Args:
        file_paths: List of file paths to process
        max_concurrent: Maximum concurrent requests
        
    Returns:
        List of results for each file
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(session: aiohttp.ClientSession, file_path: Path):
        async with semaphore:
            print(f"Processing: {file_path.name}")
            result = await process_file_async(session, file_path)
            print(f"Completed: {file_path.name} - {result['status']}")
            return result
    
    # Process all files
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_with_semaphore(session, file_path)
            for file_path in file_paths
        ]
        return await asyncio.gather(*tasks)


def generate_report(results: List[Dict[str, Any]], output_dir: Path):
    """
    Generate summary report and CSV export.
    
    Args:
        results: List of processing results
        output_dir: Directory to save reports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Summary statistics
    total_files = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    total_statements = 0
    all_statements = []
    
    # Collect all statements
    for result in results:
        if result['status'] == 'success':
            statements = result['result']['funding_statements']
            total_statements += len(statements)
            
            for stmt in statements:
                all_statements.append({
                    'file': Path(result['file']).name,
                    'statement': stmt['statement'],
                    'score': stmt['score'],
                    'query': stmt['query'],
                    'normalized': stmt.get('normalized', False)
                })
    
    # Write summary report
    report_path = output_dir / f"batch_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("FUNDING STATEMENT EXTRACTION - BATCH REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"API URL: {API_BASE_URL}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total statements found: {total_statements}\n\n")
        
        # File-by-file results
        f.write("FILE RESULTS\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"\n{Path(result['file']).name}:\n")
            if result['status'] == 'success':
                summary = result['result']['summary']
                f.write(f"  - Statements found: {summary['total_statements']}\n")
                f.write(f"  - Unique statements: {summary['unique_statements']}\n")
                f.write(f"  - Processing time: {result['result']['metadata']['processing_time']:.2f}s\n")
            else:
                f.write(f"  - ERROR: {result['error']}\n")
        
        # Failed files details
        if failed > 0:
            f.write("\n\nFAILED FILES\n")
            f.write("-" * 30 + "\n")
            for result in results:
                if result['status'] == 'error':
                    f.write(f"\n{Path(result['file']).name}:\n")
                    f.write(f"  Error: {result['error']}\n")
    
    # Write CSV export
    if all_statements:
        csv_path = output_dir / f"statements_export_{timestamp}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'statement', 'score', 'query', 'normalized'])
            writer.writeheader()
            writer.writerows(all_statements)
    
    print(f"\n📄 Report saved to: {report_path}")
    if all_statements:
        print(f"📊 CSV export saved to: {csv_path}")


def main():
    """Main function to run batch processing."""
    print("Funding Statement Extraction API - Batch Processing Example")
    print("=" * 60)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\nUsage: python batch_processing_example.py <directory_or_files> [--output-dir <dir>]")
        print("\nExamples:")
        print("  python batch_processing_example.py sample_documents/")
        print("  python batch_processing_example.py sample_documents/*.md")
        print("  python batch_processing_example.py sample_documents/10.1101.2020.02.24.20027490.md")
        print("  python batch_processing_example.py sample_documents/ --output-dir reports/")
        
        # Check if sample_documents exists
        sample_dir = Path("sample_documents")
        if sample_dir.exists() and any(sample_dir.glob("*.md")):
            print(f"\nSample documents directory found with {len(list(sample_dir.glob('*.md')))} files")
            print("Run: python batch_processing_example.py sample_documents/")
        sys.exit(1)
    
    # Determine output directory
    output_dir = Path("batch_results")
    if "--output-dir" in sys.argv:
        idx = sys.argv.index("--output-dir")
        if idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[idx + 1])
            # Remove these from args
            sys.argv.pop(idx)
            sys.argv.pop(idx)
    
    # Collect markdown files
    file_paths = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_dir():
            # Find all .md files in directory
            file_paths.extend(path.glob("*.md"))
        elif path.is_file() and path.suffix == ".md":
            file_paths.append(path)
        else:
            print(f"Warning: Skipping '{arg}' (not a markdown file or directory)")
    
    if not file_paths:
        print("Error: No markdown files found")
        sys.exit(1)
    
    print(f"\nFound {len(file_paths)} markdown files to process")
    
    print("\nStarting batch processing...")
    print("-" * 60)
    
    results = asyncio.run(process_batch(file_paths))
    
    print("\n" + "-" * 60)
    print("Generating report...")
    generate_report(results, output_dir)
    
    print("\n Batch processing complete!")


if __name__ == "__main__":
    main()