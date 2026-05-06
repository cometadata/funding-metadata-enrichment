#!/usr/bin/env python3
"""
Basic example of using the Funding Statement Extraction API.

This script demonstrates:
- Uploading a markdown file
- Extracting funding statements
- Displaying results
"""

import requests
import json
from pathlib import Path
import sys

# API configuration
API_BASE_URL = "http://localhost:8000"

def extract_funding_sync(file_path: str, normalize: bool = True):
    """
    Extract funding statements from a markdown file using the synchronous endpoint.
    
    Args:
        file_path: Path to the markdown file
        normalize: Whether to normalize the statements
    """
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found")
        return
    
    # Prepare the request
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'text/markdown')}
        data = {
            'normalize': str(normalize).lower(),
            'exclude_problematic': 'false',
            'top_k': '5',
            'threshold': '28.0'
        }
        
        # Make the request
        print(f"Uploading {file_path} to API...")
        try:
            response = requests.post(
                f"{API_BASE_URL}/extract-funding",
                files=files,
                data=data
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return
    
    # Parse the response
    result = response.json()
    
    # Display results
    print(f"\n{'='*60}")
    print("EXTRACTION RESULTS")
    print(f"{'='*60}\n")
    
    # Summary
    summary = result['summary']
    metadata = result['metadata']
    print(f"📊 Summary:")
    print(f"   - Total statements found: {summary['total_statements']}")
    print(f"   - Unique statements: {summary['unique_statements']}")
    print(f"   - Processing time: {metadata['processing_time']:.2f} seconds")
    print(f"   - Paragraphs analyzed: {metadata['num_paragraphs']}")
    
    # Statements by query
    if summary['statements_by_query']:
        print(f"\n📋 Statements by query type:")
        for query, count in summary['statements_by_query'].items():
            print(f"   - {query}: {count}")
    
    # Display funding statements
    statements = result['funding_statements']
    if statements:
        print(f"\n💰 Funding Statements ({len(statements)} found):")
        print("-" * 60)
        for i, stmt in enumerate(statements, 1):
            print(f"\n{i}. {stmt['statement']}")
            print(f"   Score: {stmt['score']:.2f} | Query: {stmt['query']}")
            if stmt.get('normalized'):
                print("   ✓ Normalized")
    else:
        print("\n❌ No funding statements found")
    
    # Problematic statements
    if result.get('problematic_statements'):
        print(f"\n⚠️  Problematic Statements ({len(result['problematic_statements'])} found):")
        print("-" * 60)
        for stmt in result['problematic_statements']:
            print(f"\n{stmt}")


def main():
    """Main function to run the example."""
    print("Funding Statement Extraction API - Basic Example")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python basic_api_example.py <markdown_file> [--no-normalize]")
        print("\nExample:")
        print("  python basic_api_example.py sample_documents/10.1101.2020.02.24.20027490.md")
        print("  python basic_api_example.py sample_documents/10.1101.2020.02.24.20027490.md --no-normalize")
        
        # Use a default file if available
        sample_dir = Path("sample_documents")
        if sample_dir.exists():
            default_files = list(sample_dir.glob("*.md"))
            if default_files:
                print(f"\nFound sample markdown files:")
                for f in default_files[:5]:
                    print(f"  - {f}")
                print(f"\nUsing '{default_files[0]}' as example...")
                file_path = str(default_files[0])
            else:
                print("\nNo sample files found in sample_documents/")
                sys.exit(1)
        else:
            print("\nNo sample_documents directory found")
            sys.exit(1)
    else:
        file_path = sys.argv[1]
    
    # Check for normalization flag
    normalize = "--no-normalize" not in sys.argv
    
    # Extract funding statements
    extract_funding_sync(file_path, normalize=normalize)


if __name__ == "__main__":
    main()